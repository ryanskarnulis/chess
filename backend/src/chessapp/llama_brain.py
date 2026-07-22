"""llama-server brain: the `Brain` implementation over a `ChatProvider`.

The brain owns orchestration — prompt assembly, the bounded tool loop, the
thinking-toggle policy — and delegates the wire to a `ChatProvider`
(`provider.py`), which speaks the OpenAI chat API to llama-server over plain
httpx, and execution to a `ToolDispatcher` (the registry). This module is the
only place that knows the model is Gemma-4 behind llama.cpp; everything else
sees the `Brain` protocol.

The loop is the fleet's standard shape (`../agent-standard/STANDARD.md` §3,
reference: `project-command-center/backend/app/ai/loop.py`): call the model
with tools, append its turn, dispatch each call, append each result as a
`role: "tool"` message, repeat. A turn with no tool calls terminates the loop
and its text is the commentary. Termination is structural — at most
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
  It stays OFF for fast move parsing and flips ON for the rest of the run once
  an analysis tool's result lands in context — the turn that comments on an
  evaluation is analysis work, the turn that parses "knight f3" is not.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jsonschema

from chessapp.brain import AgentResponse, Narration, ToolDispatcher, _RunState
from chessapp.personality import system_prompt_for
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
    enable_thinking: bool = False
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    max_corrections: int = _DEFAULT_MAX_CORRECTIONS

    def _resolve_system_prompt(self) -> str:
        """The system prompt for this request. A callable is re-resolved every
        call, so a live settings change (verbosity/hints mutating what the
        provider reads) takes effect on the next command; a plain string is a
        fixed prompt."""
        prompt = self.system_prompt
        return prompt() if callable(prompt) else prompt

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
                result = self._complete(messages, thinking=self._thinking(run))
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
                # A text turn ends the run: it is the commentary, and it was
                # produced with no tools on offer, so it cannot also act.
                return run.response(result.content or "", "completed")

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
        # The fast path's stand-in for the loop's closing turn: it reads the new
        # board and what changed, never the raw utterance, and is offered no
        # tools — so it can only comment, exactly like the turn it replaces.
        prompt = (
            "You just acted on the player's behalf. Here is what happened "
            f"(each entry is a tool call and its result):\n{json.dumps(changes)}"
            f"\n\nNew board state:\n{json.dumps(board_state)}\n\n"
            "React with a short, in-character comment for the player, based "
            "only on these results and the new board. Do not call any tools."
        )
        messages = [
            {"role": "system", "content": self._resolve_system_prompt()},
            *transcript,
            {"role": "user", "content": prompt},
        ]
        result = self.provider.chat(
            messages, tools=None, enable_thinking=self.enable_thinking
        )
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
        # The provider owns the wire (model, sampling, top_k, the payload
        # shape); the brain owns only the policy knob — whether the thinking
        # channel is on for this call. Tools are always offered: the loop ends
        # when the model declines to use them, not because we took them away.
        return self.provider.chat(
            messages,
            tools=self.tool_definitions,
            enable_thinking=(self.enable_thinking if thinking is None else thinking),
        )

    def _messages(
        self,
        board_state: dict[str, Any],
        command: str,
        transcript: Sequence[dict[str, str]] = (),
    ) -> list[dict[str, Any]]:
        # Small prompt: personality/instructions, prior conversation (a
        # bounded Transcript window, final answers only), then board truth +
        # command. It is only the *opening* of the run — the loop grows this
        # list turn by turn rather than rebuilding it, so the KV cache holds.
        user = f"Board state:\n{json.dumps(board_state)}\n\nCommand: {command}"
        return [
            {"role": "system", "content": self._resolve_system_prompt()},
            *transcript,
            {"role": "user", "content": user},
        ]


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
    enable_thinking: bool = False,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    max_corrections: int = _DEFAULT_MAX_CORRECTIONS,
    provider: ChatProvider | None = None,
) -> LlamaBrain:
    """Build a LlamaBrain against a real llama-server (e.g. localhost:8200/v1).

    `dispatcher` and `tool_definitions` should come from the same registry —
    the app assembly passes one `ToolRegistry` for both, so what the agent is
    offered is exactly what can be run.

    The system prompt, two ways: by default it is resolved once to a fixed
    string; pass a `system_prompt_provider` — a zero-arg callable the brain
    calls per command — so live settings changes (verbosity/hints) take effect
    immediately (the app-assembly wires it to read `ctx.settings`). Either way
    the brain stays prompt-agnostic: it just carries a string or a callable.

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
    return LlamaBrain(
        provider=provider,
        dispatcher=dispatcher,
        tool_definitions=tool_definitions,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        max_iterations=max_iterations,
        max_corrections=max_corrections,
    )
