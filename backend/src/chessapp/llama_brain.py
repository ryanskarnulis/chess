"""llama-server brain: the `Brain` implementation over a `ChatProvider`.

The brain owns orchestration — prompt assembly, the bounded tool loop, the
thinking-toggle policy — and delegates the wire to a `ChatProvider`
(`provider.py`), which speaks the OpenAI chat API to llama-server over plain
httpx, and execution to a `ToolDispatcher` (the registry). This module is the
only place that knows the model is Gemma-4 behind llama.cpp; everything else
sees the `Brain` protocol.

One turn is **two phases** (`docs/planner-narrator.md`, audit item 15):

- The **planner** is the loop below. It runs on `planner_prompt` — the compact,
  persona-free tool contract — because on a 12B a page of tone competes with
  the tool decision for attention. Its first turn with no tool calls ends the
  loop, and that turn's text is an internal handoff note, not commentary: the
  planner never speaks to the player.
- The **narrator** is one further call on `system_prompt` — the full Glitch
  personality — offered **no tools**, given the utterance, the turn's tool
  results and the planner's note. Its text is the reply. `narrate()`, the fast
  path's commentary turn, is the same call with a different brief; both run
  through `_speak`. Because the phase that talks holds no tools, the closing
  pass is tool-free by construction rather than by the model declining.

A budget stop (`max_iterations` / `correction_limit`) reaches no narrator: with
nothing verified to speak from, the turn ends silent and the pipeline answers
with its canned stuck reply.

The loop is the fleet's standard shape (`../agent-standard/STANDARD.md` §3,
reference: `project-command-center/backend/app/ai/loop.py`): call the model
with tools, append its turn, dispatch each call, append each result as a
`role: "tool"` message, repeat. Termination is structural — at most
`max_iterations` model turns — so the model can read a tool result while it
still holds tools (that is what makes `get_best_moves` → `make_move` possible)
without ever being able to spin.

Failures, and why they are not all the same:

- **Domain rejections are results, not errors.** An illegal move comes back
  `legal: false`, a bad save name comes back `ok: false`; both are fed back as
  ordinary tool results for the model to react to inside the iteration budget.
  This is how one illegal-move guess self-corrects instead of ending the turn.
- **Schema-level failures get a separate, smaller correction budget** — an
  unknown tool name or arguments that violate the schema. They are still fed
  back as tool results (the model sees exactly what it got wrong), but they
  also burn a correction, so a model that cannot form a valid call stops early
  rather than wasting the whole iteration budget.
- **Unparseable arguments are the one case with nowhere to attach.** The
  provider raises `ToolCallArgumentsError` *before* returning a result, so
  there is no valid `assistant(tool_calls)` turn to append and therefore no
  turn a `role: "tool"` message could answer. That correction goes back as a
  user-role message instead, and burns a correction too.

Model-specific quirks, split across the two layers:

- Gemma emits its chain-of-thought in a separate `reasoning_content` field.
  The provider drops it, so `ChatResult.content` is final answers only and
  thought blocks never leak into commentary or back into history (BRIEF).
- Thinking is toggled per request via the provider's `enable_thinking` flag.
  Planner turns never think — picking or declining a tool is a parse, whatever
  is in context. The narrator is the phase that reasons in words, so it alone
  flips ON, and only when an analysis tool answered during the run it closes
  (the turn that comments on an evaluation is analysis work, the turn that
  parses "knight f3" is not). One thinking turn per analysis question.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jsonschema

from chessapp.brain import AgentResponse, Narration, ToolDispatcher, _RunState
from chessapp.personality import planner_prompt_for, system_prompt_for
from chessapp.provider import (
    ChatProvider,
    ChatResult,
    LlamaCppProvider,
    ToolCallArgumentsError,
    Usage,
)
from chessapp.provider import ToolCall as ProviderToolCall

# How many model turns one command gets, and how many of those may be spent
# correcting a malformed tool call. Chess's tool schemas are small and closed
# (the eval baseline records zero schema corrections on the passing scenarios)
# and every turn is a local-12B round trip, so both budgets are deliberately
# tighter than PCC's 10/3.
_DEFAULT_MAX_ITERATIONS = 4
_DEFAULT_MAX_CORRECTIONS = 2

# Once one of these has answered, the rest of the run is analysis work and
# thinking goes ON (BRIEF: thinking OFF for fast move parsing, ON for analysis).
_ANALYSIS_TOOLS = frozenset(
    {"evaluate_position", "get_best_moves", "analyze_last_move"}
)


@dataclass
class LlamaBrain:
    """A `Brain` backed by a `ChatProvider` (llama-server behind llama-swap).

    The provider is injected so tests exercise the loop without a live LLM;
    `create_llama_brain` builds the real one. `dispatcher` is what actually
    runs a tool call — the validated registry — and `tool_definitions` are that
    registry's OpenAI-style schemas: the single source of truth for what the
    agent may call, and what a call is validated against before it is run.
    """

    provider: ChatProvider
    dispatcher: ToolDispatcher
    tool_definitions: list[dict[str, Any]]
    system_prompt: str | Callable[[], str]
    # The loop's own prompt. Defaults to the shipped planner contract so a
    # caller that only cares about the persona still gets the split.
    planner_prompt: str | Callable[[], str] = planner_prompt_for()
    enable_thinking: bool = False
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    max_corrections: int = _DEFAULT_MAX_CORRECTIONS
    # Per-phase sampling: the planner may run cooler than the narrator, which
    # keeps the provider's default. None means "whatever the provider samples at".
    planner_temperature: float | None = None

    def _resolve_system_prompt(self) -> str:
        """The narrator's system prompt for this request. A callable is
        re-resolved every call, so a live settings change (verbosity/hints
        mutating what the provider reads) takes effect on the next command; a
        plain string is a fixed prompt."""
        return _resolve(self.system_prompt)

    def _resolve_planner_prompt(self) -> str:
        """The loop's system prompt for this request, resolved per call for the
        same reason (hints mode changes what the planner is told)."""
        return _resolve(self.planner_prompt)

    def get_agent_response(
        self,
        board_state: dict[str, Any],
        command: str,
        transcript: Sequence[dict[str, str]] = (),
    ) -> AgentResponse:
        messages = self._messages(board_state, command, transcript)
        run = _RunState()
        corrections = 0

        for _ in range(self.max_iterations):
            try:
                # Planner turns never think: picking (or declining) a tool is a
                # parse, even when an analysis result is in context — the phase
                # that *reasons* about that result is the narrator, and it
                # inherits the thinking flip in `_close`. One thinking turn per
                # analysis question, not two.
                result = self._complete(messages)
            except ToolCallArgumentsError as exc:
                # The model was still called and the loop pays for it, so the
                # round trip counts (with no tokens — nothing came back to read).
                run.count_call()
                # Nothing to attach a tool result to (see module docstring):
                # correct with a user-role message and drop the unusable turn.
                corrections += 1
                if corrections > self.max_corrections:
                    return run.response("", "correction_limit")
                messages.append({"role": "user", "content": _wire_correction(exc)})
                continue
            run.count_call(*_usage_ints(result.usage))

            if not result.tool_calls:
                # The planner is done. Its text is a handoff note, never the
                # reply — the narrator turns the turn's verified results into
                # what the player actually reads.
                return self._close(run, command, result.content or "", transcript)

            messages.append(result.to_message())
            schema_error = False
            for call in result.tool_calls:
                payload, bad_schema = self._dispatch(call)
                schema_error = schema_error or bad_schema
                run.record(call.name, call.arguments, payload)
                messages.append(_tool_message(call.id, payload))
            if schema_error:
                corrections += 1
                if corrections > self.max_corrections:
                    return run.response("", "correction_limit")

        return run.response("", "max_iterations")

    def narrate(
        self,
        board_state: dict[str, Any],
        changes: list[dict[str, Any]],
        transcript: Sequence[dict[str, str]] = (),
    ) -> Narration:
        # The fast path's narrator turn: it reads the new board and what
        # changed, never the raw utterance. Same phase as the loop's closer —
        # same prompt, same absence of tools — with its own brief, because here
        # the move is already on the board and there is no planner note.
        return self._speak(
            _fast_path_brief(board_state, changes),
            transcript,
            thinking=self.enable_thinking,
        )

    def _close(
        self,
        run: _RunState,
        command: str,
        note: str,
        transcript: Sequence[dict[str, str]],
    ) -> AgentResponse:
        """The narrator phase: speak as Glitch from what the turn actually did.

        The board is not re-read here — the brain has no session, by design —
        so the tool results are the record of what changed, exactly as they are
        for `narrate`. The round trip is counted on the turn, so the split's
        extra call shows up in the trace and the eval baseline.
        """
        narration = self._speak(
            _closing_brief(command, run.tool_results, note),
            transcript,
            thinking=self._thinking(run),
        )
        run.count_call(narration.prompt_tokens, narration.completion_tokens)
        return run.response(narration.text, "completed")

    def _speak(
        self,
        brief: str,
        transcript: Sequence[dict[str, str]],
        *,
        thinking: bool,
    ) -> Narration:
        """One narrator round trip: the persona prompt, the conversation, a
        brief describing what happened — and no tools, so this phase cannot
        act on anything it reads."""
        messages = [
            {"role": "system", "content": self._resolve_system_prompt()},
            *transcript,
            {"role": "user", "content": brief},
        ]
        result = self.provider.chat(messages, tools=None, enable_thinking=thinking)
        prompt_tokens, completion_tokens = _usage_ints(result.usage)
        return Narration(
            text=result.content or "",
            model_calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _dispatch(self, call: ProviderToolCall) -> tuple[dict[str, Any], bool]:
        """Run one tool call; return its result and whether it failed at the
        *schema* level (an unknown tool, or arguments the schema rejects) —
        which is what separates a correction from an ordinary domain result.
        A schema-invalid call is never dispatched: the registry would only
        turn it into the same error, and the model needs the error either way.
        """
        error = _validate_call(call, self._schemas())
        if error is not None:
            return {"ok": False, "error": error}, True
        return self.dispatcher.dispatch(call.name, call.arguments), False

    def _schemas(self) -> dict[str, dict[str, Any]]:
        return {
            d["function"]["name"]: d["function"]["parameters"]
            for d in self.tool_definitions
        }

    def _thinking(self, run: _RunState) -> bool:
        """Thinking is off until an analysis tool has answered; from then on
        the run is reasoning about a position, not parsing a move."""
        if any(r["name"] in _ANALYSIS_TOOLS for r in run.tool_results):
            return True
        return self.enable_thinking

    def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool | None = None,
    ) -> ChatResult:
        # The provider owns the wire (model, top_p, top_k, the payload shape);
        # the brain owns the policy knobs — whether the thinking channel is on,
        # and the planner phase's temperature. Tools are always offered: the loop
        # ends when the model declines to use them, not because we took them
        # away (the phase that may not act is the narrator, and it is a
        # different call).
        return self.provider.chat(
            messages,
            tools=self.tool_definitions,
            enable_thinking=(self.enable_thinking if thinking is None else thinking),
            temperature=self.planner_temperature,
        )

    def _messages(
        self,
        board_state: dict[str, Any],
        command: str,
        transcript: Sequence[dict[str, str]] = (),
    ) -> list[dict[str, Any]]:
        # Small prompt: the planner's contract, prior conversation (a bounded
        # Transcript window, final answers only), then board truth + command.
        # It is only the *opening* of the run — the loop grows this list turn by
        # turn rather than rebuilding it, so the KV cache holds.
        user = f"Board state:\n{json.dumps(board_state)}\n\nCommand: {command}"
        return [
            {"role": "system", "content": self._resolve_planner_prompt()},
            *transcript,
            {"role": "user", "content": user},
        ]


def _resolve(prompt: str | Callable[[], str]) -> str:
    """A prompt that may be a fixed string or a per-call provider."""
    return prompt() if callable(prompt) else prompt


def _fast_path_brief(board_state: dict[str, Any], changes: list[dict[str, Any]]) -> str:
    """The narrator's brief for a move the loop never saw (the fast path). The
    board here *is* fresh — the caller read it after the move landed."""
    return (
        "You just acted on the player's behalf. Here is what happened "
        f"(each entry is a tool call and its result):\n{json.dumps(changes)}"
        f"\n\nNew board state:\n{json.dumps(board_state)}\n\n"
        "React with a short, in-character comment for the player, based "
        "only on these results and the new board. Do not call any tools."
    )


def _closing_brief(command: str, changes: list[dict[str, Any]], note: str) -> str:
    """The narrator's brief for a turn the planner just finished.

    No board state: the one the loop opened with is stale the moment a tool
    mutates anything, and the brain has no session to re-read (that is the
    point of the seam). The tool results are the record of what changed, and
    the planner's note says what it believes it did or what needs answering.
    """
    return (
        f"The player said:\n{command}\n\n"
        "Here is what was done about it (each entry is a tool call and its "
        f"result):\n{json.dumps(changes)}\n\n"
        f"Note from the layer that did it:\n{note}\n\n"
        "Reply to the player in character, based only on those results and "
        "that note. Do not call any tools."
    )


def _usage_ints(usage: Usage | None) -> tuple[int, int]:
    """`(prompt_tokens, completion_tokens)` from a completion's usage, or
    `(0, 0)` when llama-server omitted it — a missing count is not a failure,
    it just adds nothing to the turn's total."""
    if usage is None:
        return 0, 0
    return usage.prompt_tokens, usage.completion_tokens


def _tool_message(call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """One tool result, as the message the model was trained to read back."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload),
    }


def _validate_call(
    call: ProviderToolCall, schemas: dict[str, dict[str, Any]]
) -> str | None:
    """The schema-level complaint about a call, or None if it is well-formed.

    The provider has already guaranteed the arguments are a JSON object —
    malformed JSON never reaches here, it raised `ToolCallArgumentsError`
    upstream.
    """
    schema = schemas.get(call.name)
    if schema is None:
        return f"unknown tool: {call.name}"
    try:
        jsonschema.validate(call.arguments, schema)
    except jsonschema.ValidationError as exc:
        return f"invalid args for {call.name}: {exc.message}"
    return None


def _wire_correction(exc: ToolCallArgumentsError) -> str:
    return (
        f"Your tool call failed before execution: {exc}. "
        "Call the tool again with corrected JSON arguments."
    )


def create_llama_brain(
    *,
    base_url: str,
    model: str,
    dispatcher: ToolDispatcher,
    tool_definitions: list[dict[str, Any]],
    system_prompt_provider: Callable[[], str] | None = None,
    planner_prompt_provider: Callable[[], str] | None = None,
    enable_thinking: bool = False,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    max_corrections: int = _DEFAULT_MAX_CORRECTIONS,
    planner_temperature: float | None = None,
    provider: ChatProvider | None = None,
) -> LlamaBrain:
    """Build a LlamaBrain against a real llama-server (e.g. localhost:8200/v1).

    `dispatcher` and `tool_definitions` should come from the same registry —
    the app assembly passes one `ToolRegistry` for both, so what the agent is
    offered is exactly what can be run.

    Two prompts, because a turn is two phases: `system_prompt_provider` is the
    narrator's (the personality) and `planner_prompt_provider` is the loop's
    (the tool contract). Each defaults to being resolved once into a fixed
    string; pass a zero-arg callable — which the brain calls per command — so
    live settings changes (verbosity, hints) take effect immediately (the
    app-assembly wires both to read `ctx.settings`). Either way the brain stays
    prompt-agnostic: it just carries a string or a callable.

    `planner_temperature` samples the planner phase apart from the narrator;
    None leaves both on the provider's default.

    `provider` is injected in tests / alternate backends; otherwise the factory
    builds a real `LlamaCppProvider` against `base_url` + `model` (no API key —
    llama-server needs none).
    """
    if provider is None:
        provider = LlamaCppProvider(base_url, model)
    system_prompt: str | Callable[[], str] = (
        system_prompt_provider
        if system_prompt_provider is not None
        else system_prompt_for()
    )
    planner_prompt: str | Callable[[], str] = (
        planner_prompt_provider
        if planner_prompt_provider is not None
        else planner_prompt_for()
    )
    return LlamaBrain(
        provider=provider,
        dispatcher=dispatcher,
        tool_definitions=tool_definitions,
        system_prompt=system_prompt,
        planner_prompt=planner_prompt,
        enable_thinking=enable_thinking,
        max_iterations=max_iterations,
        max_corrections=max_corrections,
        planner_temperature=planner_temperature,
    )
