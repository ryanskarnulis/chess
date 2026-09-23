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
  personality — offered **no tools**, given the utterance and a handoff the
  harness builds from the turn's results (`handoff.py`, #289): what was done,
  refused and looked up, the fresh facts it may state, and the planner's note
  labelled as the planner's reading rather than a record. Its text is the
  reply. `narrate()`, the fast
  path's commentary turn, is the same call with a different brief; both run
  through `_speak`. Because the phase that talks holds no tools, the closing
  pass is tool-free by construction rather than by the model declining.

**A budget stop speaks from what ran** (#288). The planning phase is bounded
five ways — model turns (`max_iterations`), malformed calls
(`max_corrections`), dispatched tool calls (`max_tool_calls`), expensive
analysis calls (`max_analysis_calls`, the Stockfish-backed reads) and wall
clock (`planning_deadline_s`, checked between planner round trips; the
per-call `max_tokens` caps bound each trip) — and whichever trips first ends
the phase, named on the response as `budget`. A call past a per-turn cap is
never dispatched: it is answered with a refusal, so the wire keeps one answer
per call and the narrator's record shows it as not done. When some tool did
real work first, the narrator closes the turn from it under the loop's own
note — the player hears what was done and that the rest was not, in Glitch's
words, instead of a canned line. When nothing did (every call malformed, or
none made), there is nothing verified to speak from: the turn ends silent and
the pipeline answers with its stuck reply. A provider death anywhere in the
turn — mid-loop or in the narrator itself — ends it silent too, under
`stop_reason="provider_error"`, with everything that verifiably ran still in
the response (audit item 20: the pipeline, not an exception path, decides what
happens to a turn whose move already landed).

The loop is the fleet's standard shape (`../agent-standard/STANDARD.md` §3,
reference: `project-command-center/backend/app/ai/loop.py`): call the model
with tools, append its turn, dispatch each call, append each result as a
`role: "tool"` message, repeat. Termination is structural — at most
`max_iterations` model turns — so the model can read a tool result while it
still holds tools (that is what makes `get_best_moves` → `make_move` possible)
without ever being able to spin.

**A turn that learns nothing new is the planner's last** (`no_progress`, a
fifth stop reason beside the standard's three and chess's `provider_error`).
Measured, not theoretical: asked "what should I play?" with hints off — an ask
whose right answer, under the since-retired hints mode, was "no, hints are
off" — the planner re-ran reads it had already run and spent the whole budget
doing it (2 of 20 samples, `docs/agent-evals.md`), and a budget stop reaches no
narrator, so the player got the pipeline's canned stuck line. The stall is read
off the *results*, not the calls: a repeated call that comes back with an
answer this turn has already seen brought nothing new, so the loop ends the
planning phase itself rather than granting iterations that can only repeat.
The first cut keyed on the call alone — "an identical call cannot bring
anything new back" — and that premise is false for a mutation: `undo` takes
the same empty arguments every time and pops a different exchange every time,
so "undo my knight move and undo the bishop move and then play my knight move"
ended after the second undo with the move never played (live, 2026-07-30 and
2026-08-08). A repeat that comes back different — another `undone`, another
`fen` — did new work and the loop goes on; a repeat that comes back the same,
read or mutation, is the spin the rule exists for. (A read whose numbers
jitter, an engine score re-searched, is then bounded by the iteration budget
alone, as it was before the rule.) It is a *termination* rule and nothing
more: the repeated call is still dispatched, because whether a repeat may run
is the tool layer's judgment (the phase machine already refuses a second
player move), and the results are real — so unlike a budget stop this one
reaches the narrator and the player gets an answer.

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
- Every call carries a per-phase `max_tokens` ceiling (`_PLANNER_MAX_TOKENS` /
  `_NARRATOR_MAX_TOKENS`), because a degenerate thought loop with no cap
  generates until the read timeout (300 s) instead of for seconds. A cut-off
  call's *words* never travel: the planner's fragment is dropped and the phase
  ends under `no_progress` with the loop's own note, a cut-off narration
  becomes the empty reply the pipeline already knows how to stand in for, and a
  cut-off confirmation reading is `unrelated` — the answer that changes
  nothing. Its *tool calls* are a different matter and do run: the provider
  parses each call's arguments whole before the result exists, so a call that
  arrived is a complete call, and dropping work the model finished asking for
  would strand a batch half-done. The loop dispatches them, goes on to the
  next iteration, and treats the `content` fragment beside them as no handoff
  note at all (audit 2026-09-05, decided).
- `narrate` — and only `narrate` — also carries a wall-clock ceiling
  (`_NARRATE_TIMEOUT`). A token cap bounds generation, not queueing or a
  stalled server, and the observe beat is the one phase whose caller has
  already decided how long it will wait (`api._REACTION_BUDGET_S`): hanging up
  is what stops an abandoned reaction holding a llama-server slot the next
  turn needs (#283).
"""

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import jsonschema

from chessapp.brain import (
    CANCEL,
    CONFIRM,
    RETRY_DIFFERENT_ARGS,
    RETRY_NEVER,
    UNRELATED,
    AgentResponse,
    Answer,
    Narration,
    ToolDispatcher,
    _RunState,
)
from chessapp.handoff import build as build_handoff
from chessapp.handoff import narrator_result_view
from chessapp.handoff import render as render_handoff
from chessapp.personality import PLANNER_PROMPT, system_prompt_for
from chessapp.progress import BRAIN_NARRATING, BRAIN_PLANNING, BRAIN_REWRITING
from chessapp.provider import (
    ChatProvider,
    ChatResult,
    LlamaCppProvider,
    ProviderError,
    ToolCallArgumentsError,
    Usage,
)
from chessapp.provider import ToolCall as ProviderToolCall

logger = logging.getLogger(__name__)

# How many model turns one command gets, and how many of those may be spent
# correcting a malformed tool call. Chess's tool schemas are small and closed
# (the eval baseline records zero schema corrections on the passing scenarios)
# and every turn is a local-12B round trip, so both budgets are deliberately
# tighter than PCC's 10/3.
_DEFAULT_MAX_ITERATIONS = 4
_DEFAULT_MAX_CORRECTIONS = 2

# The per-turn budgets the iteration cap does not give (#288): four iterations
# bound the round trips, not what each one asks for. Sized from the deployed
# trace (2026-09-04 → 09-18, 236 turns: at most 2 tool calls and 2 analysis
# calls in any turn, planning p99 8.7 s) and the eval suite's multi-step asks
# (3–4 calls), generous side up — a budget that cuts a legitimate turn is the
# worse failure. The deadline is checked between planner round trips and never
# interrupts one; with `_PLANNER_MAX_TOKENS` bounding a runaway call to ~30 s,
# the planning phase ends within about 90 s whatever the model does.
_DEFAULT_MAX_TOOL_CALLS = 8
_DEFAULT_MAX_ANALYSIS_CALLS = 3
_DEFAULT_PLANNING_DEADLINE_S = 60.0

# The input budget: how large a prompt the brain will send (#288). A safety net,
# not a tuning knob — llama-server runs a 131k context and the heaviest eval
# prompt (the 84-ply game with twenty turns of history) is ~3.6k tokens, so
# this sits nearly ten times above anything real and far below the window. Read
# off an estimate, never a tokenizer round trip: `_CHARS_PER_TOKEN` is set low
# (English and JSON run ~4 on Gemma's vocabulary) so the estimate errs large.
_DEFAULT_INPUT_BUDGET_TOKENS = 32_000
_CHARS_PER_TOKEN = 3

# Hard ceilings on what one model call may generate (`max_tokens`; thinking
# tokens count toward it on this server). Without one, llama-server runs
# n_predict -1 and a degenerate thought loop generates until the provider's
# 300 s read timeout fires — observed live twice on 2026-07-27, 20k+ tokens on
# ordinary planner calls, the player watching "thinking" for five minutes and
# the GPU still grinding after the disconnect (cancellation does not reliably
# propagate through llama-swap). The numbers are sized from measured output,
# generous side up, because truncating a legitimate turn is the worse failure:
# the planner never thinks and its real output is tool calls or a one-line
# note (tens of tokens), so 2048 is ~20× headroom and bounds a runaway to
# ~30 s; the narrator's one thinking turn legitimately reaches ~2.6k tokens
# (docs/agent-evals.md; a live 2,408 in the 2026-07-27 trace), so 4096 keeps
# every observed real narration intact and bounds a runaway to ~60 s.
_PLANNER_MAX_TOKENS = 2048
_NARRATOR_MAX_TOKENS = 4096
# The planner's sampling temperature. The narrator keeps the model profile's
# 1.0 (`provider._TEMPERATURE`, agent-standard/model-profile.md): its job is
# words, and words want the spread. The planner's job is a parse — which tool,
# if any, and with what arguments — and a parse wants the mode. Measured, not
# derived (2026-09-17, #286, `docs/knight-ask-campaign.md`): "move my kings
# knight" on a fresh board must be asked about, never played, and at 1.0 the
# planner played one of the two knights 6/40; at 0.6, 3/40; at 0.3, 0/40 —
# arms round-robin per sample on one server, the interleaving the correlated
# server makes necessary. Cooling it did not make a wrong parse consistent
# anywhere the rule also governs: the STT knight, the rook ask, both
# refusals, castling and the undo-then-replace first call all read 20/20 at
# both temperatures, and the harness confirm and the full gate are recorded
# in `docs/agent-evals.md`. `CHESSAPP_PLANNER_TEMPERATURE` still overrides
# it, so the next measurement needs no code change either.
_PLANNER_TEMPERATURE = 0.3
# The answer-reading phase writes one word. Sized for the word plus whatever
# punctuation or preamble a 12B insists on wrapping it in, and nothing more:
# this call sits in front of a destructive op and must not be a place a
# thought loop can live.
_ANSWER_MAX_TOKENS = 16

# How long one `narrate` round trip may take before the socket is closed on it.
# The *pipeline* is what gives up first: `api._REACTION_BUDGET_S` stops waiting
# for the reaction at 10 s and plays the reply Stockfish already computed, so
# ordinary slowness is always that budget's call and never this one. This is the
# backstop underneath it — without it the abandoned generation keeps a
# llama-server slot until the 300 s read timeout and the next turn's planner
# queues behind words nobody will ever hear. Sized just above the budget so the
# two cannot race, and scoped to `narrate` alone: the planner and the loop's own
# closing narrator legitimately run 30 s and more with thinking on
# (`docs/agent-evals.md`), and they are not calls the app stops waiting for.
_NARRATE_TIMEOUT = 15.0

# The whole prompt for that phase. No persona and no board — the question is
# about what the player meant, and every extra line is one more thing for a
# 12B to answer instead. The three words are the entire output vocabulary, and
# the pipeline reads anything else as `unrelated`, which changes nothing.
_ANSWER_PROMPT = """\
You read one short reply and say what it means. Nothing else.

The player has been asked a yes-or-no question and has answered in their own
words. Reply with exactly one of these words and no other text:

- confirm — they mean yes, go ahead
- cancel — they mean no, don't
- unrelated — they are asking for something else entirely, or you cannot tell

Judge the reply against the question they were asked. Impatience is still a
yes ("just do it", "get on with it"); reluctance is still a no ("actually,
forget it"); a different request is unrelated ("undo my last move instead").
"""

# The word the model settled on, if it settled on one. A model reading its own
# three-word menu is the one place a literal match is the right reader —
# nothing here is player language.
_VERDICTS = {CONFIRM: CONFIRM, CANCEL: CANCEL, UNRELATED: UNRELATED}

# Once one of these has answered, the rest of the run is analysis work and
# thinking goes ON (BRIEF: thinking OFF for fast move parsing, ON for analysis).
_ANALYSIS_TOOLS = frozenset(
    {"evaluate_position", "get_best_moves", "analyze_last_move"}
)

# What `max_analysis_calls` counts: every tool that runs a Stockfish search.
# `review_game` is one search per ply of the game, so it is the dearest call
# on the menu — but it does not flip thinking, which is `_ANALYSIS_TOOLS`'s
# separate question.
_EXPENSIVE_TOOLS = _ANALYSIS_TOOLS | {"review_game"}

# What a call past a per-turn cap is answered with instead of being run. Worded
# for the narrator, who reads it as the call's refusal reason: a fact about the
# work ("not run"), never about the machinery.
_NOT_RUN = "not run this turn"

# The planner's clarification (`tools.ASK_PLAYER`): the one call that ends the
# planning phase by itself, because what it asks for only the player can answer.
_ASK_PLAYER = "ask_player"

# The handoff note for a turn the *loop* ended (`no_progress`) rather than the
# planner. The planner never reached the turn that writes one, and handing the
# narrator a brief with no note at all was measured to cost real seconds: with
# nothing saying the work was finished, it reasoned about what to do next
# instead of what to say (`docs/agent-evals.md`). So the loop supplies the one
# fact it owns. Deliberately about the *work* and not the machinery: how the
# planning phase ended is nobody's business but the trace's, and a note that
# mentioned repeated calls would invite commentary about the loop.
_NO_PROGRESS_NOTE = (
    "Nothing further was done — the results above are everything this turn has."
)

# The handoff note for a turn a budget ended (#288), in the same spirit: about
# the work, never the machinery. Unlike a stall, a budget can end a turn with
# part of the ask still undone, so the note says that whatever the record does
# not show as done was not done — the one fact the narrator needs to tell the
# player the turn stopped part-way without being handed the words for it.
_BUDGET_NOTE = (
    "This turn ended before everything the player asked for was done. "
    "Whatever the record above does not show as done was not done."
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
    # What the loop is *offered* — and therefore all it may call: a call
    # outside this list is a schema-level unknown even when the dispatcher
    # behind it could run it. A callable is re-resolved per command, the same
    # live seam as the two prompts, because live state changes what exists
    # for the model (no claimable draw withholds `claim_draw`), not just
    # what it is told.
    tool_definitions: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]]
    system_prompt: str | Callable[[], str]
    # The loop's own prompt. Defaults to the shipped planner contract so a
    # caller that only cares about the persona still gets the split.
    planner_prompt: str | Callable[[], str] = PLANNER_PROMPT
    enable_thinking: bool = False
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    max_corrections: int = _DEFAULT_MAX_CORRECTIONS
    # The per-turn tool-work and wall-clock budgets (#288; see the module
    # constants for the sizing). `planning_deadline_s=None` disables the clock.
    max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS
    max_analysis_calls: int = _DEFAULT_MAX_ANALYSIS_CALLS
    planning_deadline_s: float | None = _DEFAULT_PLANNING_DEADLINE_S
    # The largest prompt one call may send, in estimated tokens (#288; see
    # `_DEFAULT_INPUT_BUDGET_TOKENS`). `None` disables the guard.
    input_budget_tokens: int | None = _DEFAULT_INPUT_BUDGET_TOKENS
    # Per-phase sampling: the planner runs cooler than the narrator, which
    # keeps the provider's default. The shipped number is `_PLANNER_TEMPERATURE`,
    # applied by `create_llama_brain`; here None still means "whatever the
    # provider samples at", so a direct construction changes nothing it did not
    # ask for.
    planner_temperature: float | None = None
    # Per-phase generation ceilings (see the module constants for the sizing).
    # A call the ceiling cuts off is a failed turn, never a truncated one that
    # travels: the loop and `_speak` both check `finish_reason == "length"`.
    planner_max_tokens: int = _PLANNER_MAX_TOKENS
    narrator_max_tokens: int = _NARRATOR_MAX_TOKENS
    # The observe beat's own read ceiling (see `_NARRATE_TIMEOUT`). Only
    # `narrate` carries one, because it is the only phase whose caller has
    # already decided it will not wait; `None` disables it.
    narrate_timeout: float | None = _NARRATE_TIMEOUT
    # Wall clock for the per-call latencies the trace records. Injected so the
    # timing is testable, and read *here* rather than in the provider because a
    # round trip that raises has a latency too — and only the caller of a raising
    # call is still around to record it.
    clock: Callable[[], float] = field(default=time.monotonic)
    # Told which phase is about to run — `planning` before every planner round
    # trip, `narrating` before the one that writes the words (audit item 19).
    # The brain's *own* vocabulary: it knows nothing about turns, coordinators
    # or sockets, and whoever wires this up is free to read more into it than
    # the brain does. The app reads `narrating` as the observation beat opening,
    # because a brain that holds no coordinator (by design) cannot say so
    # itself. Nothing is reported for a phase that does not run: a budget stop
    # and a dead provider reach no narrator, and must not claim to.
    on_phase: Callable[[str], None] | None = None
    # The board as of *now*, asked once per planner iteration (#282). The
    # opening state block is the only `legal_moves` this loop holds, and it
    # ages the moment one of the turn's own tools mutates; this is how the loop
    # is told. `None` means nobody can say — no seam wired, or a caller
    # declining because the board is mid-exchange and its side to move is not
    # the player's. The brain reads nothing into the dict: it compares each
    # answer with the last one it showed and appends it when they differ, and
    # that is the whole of the policy here. What the view holds, and when it is
    # withheld, is app assembly's judgment (`api.planner_board_refresh`) — this
    # class holds no session, by design.
    #
    # Deliberately not a `_resolve`-style field: that is read once per command
    # (the offer must not change under a run), and this is read once per
    # iteration, which is the point.
    board_refresh: Callable[[], dict[str, Any] | None] | None = None
    # The game as the narrator may state it, read once as the planner hands
    # off (#289): a small, side-free view (`api.narrator_facts`) plus
    # `reply_owed`, whether the player's move is still waiting on the engine's
    # answer. The planner's opening board is stale by the time the narrator
    # speaks, and the brain holds no session to re-read — the same reason
    # `board_refresh` exists, one phase later. `None` (unwired, or a closure
    # that raised) closes the turn from the results alone, as it did before.
    narrator_facts: Callable[[], dict[str, Any] | None] | None = None

    def _resolve_system_prompt(self) -> str:
        """The narrator's system prompt for this request. A callable is
        re-resolved every call, so a live settings change (verbosity mutating
        what the provider reads) takes effect on the next command; a plain
        string is a fixed prompt."""
        return _resolve(self.system_prompt)

    def _resolve_planner_prompt(self) -> str:
        """The loop's system prompt for this request, resolved per call
        through the same seam — the contract is static today, but the wire
        must not assume it stays that way."""
        return _resolve(self.planner_prompt)

    def get_agent_response(
        self,
        board_state: dict[str, Any],
        command: str,
        transcript: Sequence[dict[str, str]] = (),
    ) -> AgentResponse:
        # One resolution per command: the offer and the schemas it is validated
        # against must be the same list for the whole run, even if the run's
        # own work (a move that makes a draw claimable) changes what the
        # *next* command gets.
        tools = _resolve(self.tool_definitions)
        schemas = _schemas_of(tools)
        run = _RunState()
        # Admission (#288): the oldest conversation goes first when the opening
        # prompt would not fit. Fitted once, here, because the loop only ever
        # appends — trimming mid-run would rewrite the prefix the KV cache
        # holds. The same trimmed conversation is what the narrator is handed.
        transcript, run.input_trimmed = self._admit(
            lambda kept: self._messages(board_state, command, kept),
            transcript,
            tools,
        )
        messages = self._messages(board_state, command, transcript)
        corrections = 0
        # Every exchange this turn has already had — a call and what it brought
        # back — so a turn that learns nothing new can be recognized as the
        # planner's last (see `_exchange_key`).
        seen: set[tuple[str, str, str]] = set()
        # Which board the planner has been shown. Seeded from the seam before
        # the first turn so the first refresh fires on a real change and not
        # merely on the seam existing, and held as the version rather than the
        # view because the version is what says "a different board": the view
        # also carries facts a tool can move without touching the position (a
        # setting, a save), and those the tool's own result already reports.
        shown = _board_version_of(self._current_board())
        # The per-turn work budgets (#288): what has been dispatched, and when
        # the planning phase must stop asking for more.
        dispatched = 0
        expensive = 0
        # When the phase opened, read off the first round trip's own start
        # rather than a clock read of its own, so the per-call latencies the
        # trace records stay one reading each.
        opened: float | None = None

        for _ in range(self.max_iterations):
            started = self.clock()
            if opened is None:
                opened = started
            elif (
                self.planning_deadline_s is not None
                and started - opened >= self.planning_deadline_s
            ):
                # Checked between round trips, never during one: the call in
                # flight is bounded by its `max_tokens`, and what it asked for
                # has already run. This only declines to start another.
                return self._budget_stop(
                    run, command, transcript, "budget", "wall_time", dispatched
                )
            if self._over_input_budget(messages, tools):
                # The run's own results have grown the prompt past the budget
                # (or the opening could not be fitted at all). No trim can
                # help without rewriting what the planner already read, so the
                # phase ends here — spoken from what ran, like any budget.
                return self._budget_stop(
                    run, command, transcript, "budget", "input", dispatched
                )
            self._report(BRAIN_PLANNING)
            try:
                # Planner turns never think: picking (or declining) a tool is a
                # parse, even when an analysis result is in context — the phase
                # that *reasons* about that result is the narrator, and it
                # inherits the thinking flip in `_close`. One thinking turn per
                # analysis question, not two.
                result = self._complete(messages, tools)
            except ToolCallArgumentsError as exc:
                # The model was still called and the loop pays for it, so the
                # round trip counts (with no tokens — nothing came back to read).
                run.count_call(latency_ms=self._elapsed_ms(started), metered=False)
                # Nothing to attach a tool result to (see module docstring):
                # correct with a user-role message and drop the unusable turn.
                corrections += 1
                if corrections > self.max_corrections:
                    return self._budget_stop(
                        run,
                        command,
                        transcript,
                        "correction_limit",
                        "corrections",
                        dispatched,
                    )
                _append_user(
                    messages, {"role": "user", "content": _wire_correction(exc)}
                )
                continue
            except ProviderError as exc:
                # The provider died mid-turn (audit item 20). Whatever tools
                # already ran are real board history — hand them back under
                # their own stop so the pipeline can close the turn and tell
                # the truth, instead of an exception escaping after the board
                # changed. The round trip is counted like any raised call.
                # `exc.failure` rides along: the stop says the turn died, the
                # kind says whether asking again is worth anything.
                run.count_call(latency_ms=self._elapsed_ms(started), metered=False)
                return run.response("", "provider_error", str(exc.failure))
            run.count_call(
                *_usage_ints(result.usage),
                self._elapsed_ms(started),
                metered=result.usage is not None,
            )

            if not result.tool_calls:
                if result.finish_reason == "length":
                    # The cap cut the model off mid-generation (the
                    # runaway-thought fix: without it this was five minutes of
                    # "thinking" ending in a dead socket). Whatever content
                    # survived is a fragment — or nothing, when the whole
                    # budget went to reasoning — never a handoff note, so it
                    # must not travel. The turn brought nothing usable back,
                    # which is the no-progress stop's exact contract: end the
                    # phase, let the narrator close from what the turn
                    # verified, under the loop's own note.
                    return self._close(
                        run, command, _NO_PROGRESS_NOTE, transcript, "no_progress"
                    )
                # The planner is done. Its text is a handoff note, never the
                # reply — the narrator turns the turn's verified results into
                # what the player actually reads.
                return self._close(run, command, result.content or "", transcript)

            # Tool calls, so `finish_reason` is deliberately not consulted:
            # the provider parsed every call's arguments before this result
            # existed, so each call here is a whole call, and the cap having
            # cut the generation short says nothing about them. They run and
            # the loop goes on — a batch abandoned half-way because the model
            # was still talking would drop work it had finished asking for.
            # What the truncation does cost is the turn's prose: a fragment
            # riding along in `content` is never a handoff note, and only a
            # tool-free turn's text is (see the branch above).
            messages.append(result.to_message())
            schema_error = False
            progressed = False
            tripped = ""
            for call in result.tool_calls:
                over = self._over_budget(call.name, dispatched, expensive)
                if over:
                    # Past a per-turn cap: answered, never run (#288). The
                    # refusal keeps the wire's one answer per call and puts the
                    # call on the narrator's record as not done; it is neither
                    # progress nor a stall, so the rule below never sees it.
                    tripped = tripped or over
                    payload = self.dispatcher.refusal(_NOT_RUN, RETRY_NEVER)
                    run.record(call.name, call.arguments, payload)
                    messages.append(_tool_message(call.id, payload))
                    continue
                payload, bad_schema = self._dispatch(call, schemas)
                schema_error = schema_error or bad_schema
                if not bad_schema:
                    dispatched += 1
                    expensive += call.name in _EXPENSIVE_TOOLS
                # Judged after the dispatch, on what came back: a repeated call
                # is a stall only when it is answered as it already was this
                # turn. A second `undo` pops a different exchange and says so.
                exchange = _exchange_key(call.name, call.arguments, payload)
                progressed = progressed or exchange not in seen
                seen.add(exchange)
                run.record(call.name, call.arguments, payload)
                messages.append(_tool_message(call.id, payload))
            # What the next decision is actually about (#282). Once per
            # iteration and never per call: a refresh between two tool messages
            # would break the contiguous answer-per-call shape the wire expects,
            # and the decision this repairs is the next *turn's*, not the next
            # call's. Appended rather than merged into a result, which is what
            # keeps it the planner's alone — the narrator's brief is built from
            # `run.tool_results` and the stall rule keys on the payload, so a
            # fact stapled to a result would reach both (#193, `no_progress`).
            # Before the two branches below deliberately: the message is simply
            # never sent on a turn that returns, and the alternative is a second
            # copy of the condition.
            if any(
                r["name"] == _ASK_PLAYER and r["result"].get("ok") is True
                for r in run.tool_results
            ):
                # The planner asked the player to choose (#289): the question is
                # the turn's answer, and no further iteration can bring the
                # player's choice back. Terminal by construction, so a planner
                # that asked cannot go on to play one of the candidates anyway.
                return self._close(run, command, "", transcript)
            if tripped:
                # A cap refused part of this batch, so the next iteration could
                # only be refused more of the same: the phase ends here.
                return self._budget_stop(
                    run, command, transcript, "budget", tripped, dispatched
                )
            current = self._current_board()
            version = _board_version_of(current)
            if current is not None and version != shown:
                _append_user(messages, _board_refresh_message(current))
                run.show_board(version)
                shown = version
            if schema_error:
                corrections += 1
                if corrections > self.max_corrections:
                    return self._budget_stop(
                        run,
                        command,
                        transcript,
                        "correction_limit",
                        "corrections",
                        dispatched,
                    )
                # A malformed call never dispatched, so repeating it is not the
                # stall below — it is what the correction budget exists for, and
                # that budget (smaller than the iteration one) already ends the
                # turn early. Leave this turn to it.
            elif not progressed:
                # Every call this turn repeated one the turn had already made
                # and came back with the same answer, so no further iteration
                # can bring anything new back — the planner has stopped making
                # progress and this turn is its last.
                # Not a budget stop: results *did* come back, so the narrator
                # closes the turn from them and the player gets an answer
                # instead of the pipeline's canned stuck line. The note is the
                # loop's own, because the planner never reached the turn that
                # writes one — see `_NO_PROGRESS_NOTE`.
                return self._close(
                    run, command, _NO_PROGRESS_NOTE, transcript, "no_progress"
                )

        return self._budget_stop(
            run, command, transcript, "max_iterations", "iterations", dispatched
        )

    def _over_budget(self, name: str, dispatched: int, expensive: int) -> str:
        """Which per-turn cap a call named `name` would exceed, or "" when it
        may run. The total comes first: a call over both is over the total."""
        if dispatched >= self.max_tool_calls:
            return "tool_calls"
        if name in _EXPENSIVE_TOOLS and expensive >= self.max_analysis_calls:
            return "analysis_calls"
        return ""

    def _budget_stop(
        self,
        run: _RunState,
        command: str,
        transcript: Sequence[dict[str, str]],
        stop_reason: str,
        budget: str,
        dispatched: int,
    ) -> AgentResponse:
        """End the planning phase on a budget (#288) — spoken when there is
        something to speak from, silent when there is not.

        `dispatched` counts the calls that actually ran. Zero means every call
        was malformed or refused, or none was made: nothing verified happened,
        and the pipeline's stuck reply is the honest answer. Otherwise the
        narrator closes from the record under `_BUDGET_NOTE`, so the player
        hears what was done and that the rest was not.
        """
        run.budget = budget
        if not dispatched:
            return run.response("", stop_reason)
        return self._close(run, command, _BUDGET_NOTE, transcript, stop_reason)

    def _over_input_budget(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
    ) -> bool:
        return (
            self.input_budget_tokens is not None
            and _estimate_tokens(messages, tools) > self.input_budget_tokens
        )

    def _admit(
        self,
        build: Callable[[Sequence[dict[str, str]]], list[dict[str, Any]]],
        transcript: Sequence[dict[str, str]],
        tools: Sequence[dict[str, Any]] = (),
    ) -> tuple[list[dict[str, str]], int]:
        """Fit a prompt to the input budget by dropping the conversation's
        oldest exchanges; return what is kept and how many exchanges went.

        Only the conversation is ever trimmed — the system prompt, the state
        block (`legal_moves` among it) and the brief are what the call is
        *for*. It goes a user/assistant pair at a time, oldest first (the
        digest before any verbatim turn), so the alternation the chat template
        expects holds, and the latest exchange is never dropped: it is what
        "do the second one" refers to, and where an unanswered `ask_player`
        clarification lives. When that is still too large the caller decides
        what an over-budget prompt means for its phase.
        """
        kept = list(transcript)
        trimmed = 0
        while len(kept) > 2 and self._over_input_budget(build(kept), tools):
            del kept[:2]
            trimmed += 1
        if trimmed:
            logger.warning("input_budget_trimmed exchanges=%d", trimmed)
        return kept, trimmed

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
        # Timed here rather than in `_speak` for the same reason the loop times
        # its own calls: latency belongs wherever the call is *accounted for*,
        # and this route's accounting is the `Narration` itself.
        self._report(BRAIN_NARRATING)
        started = self.clock()
        narration = self._speak(
            _fast_path_brief(board_state, changes),
            transcript,
            thinking=self.enable_thinking,
            timeout=self.narrate_timeout,
        )
        return replace(narration, latency_ms=self._elapsed_ms(started))

    def rewrite(
        self,
        commentary: str,
        corrections: Sequence[str],
        transcript: Sequence[dict[str, str]] = (),
    ) -> Narration:
        # The honesty guard's second try: the narrator phase once more, on the
        # same persona prompt and the same conversation, with a brief that
        # holds its own reply and the facts the board actually backs. No
        # tools, like every narrator call; thinking off, because this is a
        # rephrase and not a position to reason about. Timed here for the
        # reason `narrate` is: the `Narration` is this call's accounting.
        self._report(BRAIN_REWRITING)
        started = self.clock()
        narration = self._speak(
            _rewrite_brief(commentary, corrections),
            transcript,
            thinking=False,
        )
        return replace(narration, latency_ms=self._elapsed_ms(started))

    def read_answer(self, question: str, text: str) -> Answer:
        """One tool-free round trip that classifies a reply, and nothing else.

        Fails to `UNRELATED` on every unhappy path — a dead provider, a
        truncated reply, a word that is not on the menu. That is the direction
        the destructive gate fails in everywhere else, and it is the only safe
        one here: the alternative is a game ended by a provider hiccup.
        """
        started = self.clock()
        messages = [
            {"role": "system", "content": _ANSWER_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\nTheir reply: {text}",
            },
        ]
        try:
            result = self.provider.chat(
                messages,
                tools=None,
                enable_thinking=False,
                max_tokens=_ANSWER_MAX_TOKENS,
            )
        except ProviderError:
            logger.warning("answer_reading_failed", exc_info=True)
            # Still `unrelated`, which changes nothing — but the round trip was
            # made and the turn waited on it, so it is counted like any raised
            # call: one call, a real latency, tokens unknown (#290).
            return Answer(
                model_calls=1,
                unmetered_calls=1,
                latency_ms=self._elapsed_ms(started),
            )
        prompt_tokens, completion_tokens = _usage_ints(result.usage)
        unmetered = 0 if result.usage is not None else 1
        if result.finish_reason == "length":
            # The cap cut this call off, so whatever came back is the start of
            # something rather than a verdict — and the one place in the app
            # where a word read too generously ends a game. The same rule the
            # planner and the narrator keep for a truncated call, applied where
            # it matters most: the round trip happened and the turn pays for
            # it (unlike a call that never returned at all), and the word does
            # not travel. `unrelated` by omission, which changes nothing.
            logger.warning("answer_reading_truncated")
            return Answer(
                model_calls=1,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=self._elapsed_ms(started),
                unmetered_calls=unmetered,
            )
        word = (result.content or "").strip().strip(".!,'\"").lower()
        return Answer(
            verdict=_VERDICTS.get(word, UNRELATED),
            model_calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=self._elapsed_ms(started),
            unmetered_calls=unmetered,
        )

    def _close(
        self,
        run: _RunState,
        command: str,
        note: str,
        transcript: Sequence[dict[str, str]],
        stop_reason: str = "completed",
    ) -> AgentResponse:
        """The narrator phase: speak as Glitch from what the turn actually did.

        What it did is the harness's reading, not the planner's (#289): the
        results are sorted into done, refused and looked up (`handoff.build`),
        and the note is passed on labelled as the planner's reading of the ask.
        The facts the narrator may state come through the `narrator_facts`
        seam — the brain has no session, by design. The round trip is counted
        on the turn, so the split's extra call shows up in the trace and the
        eval baseline.

        `stop_reason` is how the planning phase ended: `completed` when the
        planner declared itself done, `no_progress` when it repeated itself to
        no effect and the loop ended the phase for it. Both reach the narrator —
        the distinction is what the trace and the eval report read.
        """
        facts = dict(self._narrator_facts() or {})
        handoff = build_handoff(
            run.tool_results,
            stop_reason,
            note=note,
            reply_owed=bool(facts.pop("reply_owed", False)),
            facts=facts,
        )
        self._report(BRAIN_NARRATING)
        started = self.clock()
        try:
            narration = self._speak(
                render_handoff(handoff, command, run.tool_results),
                transcript,
                thinking=self._thinking(run),
            )
        except ProviderError as exc:
            # The plan finished; the persona call died. Same contract as a
            # mid-loop failure: the verified results come back, the words don't,
            # and the kind of death comes back with them. This is the call a
            # context overrun reaches first — the narrator carries the whole
            # conversation, so it is the longest prompt of the turn.
            run.count_call(latency_ms=self._elapsed_ms(started), metered=False)
            return run.response("", "provider_error", str(exc.failure), handoff)
        run.count_call(
            narration.prompt_tokens,
            narration.completion_tokens,
            self._elapsed_ms(started),
            metered=not narration.unmetered_calls,
        )
        return run.response(narration.text, stop_reason, handoff=handoff)

    def _speak(
        self,
        brief: str,
        transcript: Sequence[dict[str, str]],
        *,
        thinking: bool,
        timeout: float | None = None,
    ) -> Narration:
        """One narrator round trip: the persona prompt, the conversation, a
        brief describing what happened — and no tools, so this phase cannot
        act on anything it reads.

        `timeout` is the observe beat's alone (`_NARRATE_TIMEOUT`): the rewrite
        and the loop's closer are calls the pipeline waits for, so they send
        none and keep the client's."""
        system = self._resolve_system_prompt()

        def build(kept: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
            return [
                {"role": "system", "content": system},
                *kept,
                {"role": "user", "content": brief},
            ]

        kept, _ = self._admit(build, transcript)
        messages = build(kept)
        if self._over_input_budget(messages):
            # Even the conversation's latest exchange does not leave room for
            # this brief: nothing is sent, and the empty reply is the one every
            # caller already knows how to stand in for — a cut-off narration's.
            logger.warning("narration_over_input_budget")
            return Narration(text="")
        result = self.provider.chat(
            messages,
            tools=None,
            enable_thinking=thinking,
            max_tokens=self.narrator_max_tokens,
            timeout=timeout,
        )
        prompt_tokens, completion_tokens = _usage_ints(result.usage)
        # A narration the cap cut off is not commentary: the words stop
        # mid-claim, or never started (a thought loop spends the whole budget
        # in reasoning and leaves content empty). Half a sentence shown to the
        # player is worse than none — the pipeline already composes its
        # deterministic lines around an empty reply — so a "length" finish
        # keeps the cost and drops the words.
        truncated = result.finish_reason == "length"
        return Narration(
            text="" if truncated else (result.content or ""),
            model_calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            unmetered_calls=0 if result.usage is not None else 1,
        )

    def _dispatch(
        self, call: ProviderToolCall, schemas: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, Any], bool]:
        """Run one tool call; return its result and whether it failed at the
        *schema* level (an unknown tool, or arguments the schema rejects) —
        which is what separates a correction from an ordinary domain result.
        A schema-invalid call is never dispatched: the registry would only
        turn it into the same error, and the model needs the error either way.
        `schemas` are the run's offered tools' — so a tool withheld from the
        offer (no claimable draw) is an unknown here too, never a dispatch.

        The refusal is built *through the dispatcher* so that a "no" the loop
        wrote reads exactly like a "no" a handler raised: `retry` saying
        whether another call can help, and `board_version` saying which board
        it was about. These two used to be the turn's only failures carrying
        neither, which made the planner's "every failure tells you how to
        recover" contract untrue for the two failures most in need of it.
        """
        complaint = _validate_call(call, schemas)
        if complaint is not None:
            error, retry = complaint
            return self.dispatcher.refusal(error, retry), True
        return self.dispatcher.dispatch(call.name, call.arguments), False

    def _current_board(self) -> dict[str, Any] | None:
        """The refresh seam's answer, and never let asking cost the turn.

        The same rule `_report` and the tracer keep, and sharper here: this runs
        on a turn whose move may already have landed, so a closure that raises
        must degrade to the behavior that shipped before the seam existed, not
        to a dead turn.
        """
        if self.board_refresh is None:
            return None
        try:
            return self.board_refresh()
        except Exception:
            logger.warning("board_refresh_failed", exc_info=True)
            return None

    def _narrator_facts(self) -> dict[str, Any] | None:
        """The facts seam's answer, degrading to none — `_current_board`'s rule:
        this runs after the turn's work has landed, and a closure that raises
        must cost the narrator its facts, never the turn its words."""
        if self.narrator_facts is None:
            return None
        try:
            return self.narrator_facts()
        except Exception:
            logger.warning("narrator_facts_failed", exc_info=True)
            return None

    def _report(self, phase: str) -> None:
        """Say which phase is starting, and never let that cost the turn — the
        same rule the tracer and the tool observer keep."""
        if self.on_phase is None:
            return
        try:
            self.on_phase(phase)
        except Exception:
            logger.warning("brain_phase_report_failed", exc_info=True)

    def _elapsed_ms(self, started: float) -> int:
        """Whole milliseconds since `started`, never negative.

        Milliseconds because that is the resolution a local 12B's round trips
        are read at (hundreds to thousands), and an int because a trace record
        is read by eye.
        """
        return max(0, round((self.clock() - started) * 1000))

    def _thinking(self, run: _RunState) -> bool:
        """Thinking is off until an analysis tool has answered; from then on
        the run is reasoning about a position, not parsing a move."""
        if any(r["name"] in _ANALYSIS_TOOLS for r in run.tool_results):
            return True
        return self.enable_thinking

    def _complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        thinking: bool | None = None,
    ) -> ChatResult:
        # The provider owns the wire (model, top_p, top_k, the payload shape);
        # the brain owns the policy knobs — whether the thinking channel is on,
        # the planner phase's temperature and its generation ceiling. Tools are
        # always offered: the loop
        # ends when the model declines to use them, not because we took them
        # away (the phase that may not act is the narrator, and it is a
        # different call).
        return self.provider.chat(
            messages,
            tools=tools,
            enable_thinking=(self.enable_thinking if thinking is None else thinking),
            max_tokens=self.planner_max_tokens,
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


def _estimate_tokens(
    messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] = ()
) -> int:
    """A prompt's size in tokens, estimated from its characters — every
    message's content and tool calls, plus the tool schemas the call offers,
    which ride in the same prompt. Deliberately pessimistic (see
    `_CHARS_PER_TOKEN`): this guards a ceiling, and an estimate that errs large
    trims early rather than letting a prompt through that does not fit."""
    chars = sum(len(message.get("content") or "") for message in messages)
    chars += sum(
        len(json.dumps(message["tool_calls"], default=str))
        for message in messages
        if message.get("tool_calls")
    )
    chars += len(json.dumps(list(tools))) if tools else 0
    return chars // _CHARS_PER_TOKEN


def _resolve[T](value: T | Callable[[], T]) -> T:
    """A prompt or tool list that may be a fixed value or a per-call provider."""
    return value() if callable(value) else value


def _schemas_of(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {d["function"]["name"]: d["function"]["parameters"] for d in tools}


def _exchange_key(
    name: str, args: dict[str, Any], result: dict[str, Any]
) -> tuple[str, str, str]:
    """Identity of one exchange — the tool, its arguments and what it answered —
    for telling a stall from new work. The result is part of the identity
    because the call alone is not: `undo` with the same empty arguments pops a
    different exchange every time, and it is the answer that says so. Sorted
    keys and `default=str` because the key must be the *exchange*, not how the
    model happened to serialize the call, and a value that will not serialize
    must still produce a key rather than raise inside the loop.
    """
    return (
        name,
        json.dumps(args, sort_keys=True, default=str),
        json.dumps(result, sort_keys=True, default=str),
    )


# Whose move the narrator is about to react to. The brief used to open "You
# just acted on the player's behalf", and a 12B reads that as "I moved": live,
# Glitch narrated the player's capture as his own. Who moved is deterministic
# state, so the brief says it outright instead of leaving it to be inferred
# from move-history parity.
#
# It says that and nothing more, deliberately. The first cut also named each
# side's color, and that was an identity the narrator *used*: it reacts
# mid-turn, from a board where it is the engine's move, so "you are playing
# black" turned the reaction beat into a move-selection beat — every reaction
# in the 2026-07-28 game announced a reply ("i'll go with d6", then Bb4 was
# played) that Stockfish was still computing. The attribution the
# misattribution fix needed was purely negative — whose move the results are
# NOT — and the narrator must be offered no side to play for. Cutting the
# prose alone did not finish the job: the state block still said the same
# thing in data (`turn` beside `player_color`, the engine's `legal_moves` as
# the menu) and the next game announced replies all the same ("My turn.
# ...Be6."), so the app now withholds those fields from the narrator's view
# too (`api._narrator_state_dict`, #193).
_ATTRIBUTION = (
    "The player just made their own move, and you carried it out for them: "
    "the moves in these results are the player's moves, not yours."
)


def _fast_path_brief(board_state: dict[str, Any], changes: list[dict[str, Any]]) -> str:
    """The narrator's brief for a move the loop never saw (the fast path). The
    board here *is* fresh — the caller read it after the move landed.

    The changes go through the same projection the closing brief's results do
    (`handoff.narrator_result_view`): a confirmed `new_game` or a resign beat
    narrates from results that carry `fen`/`turn`, and the state view beside
    them withholding those keys was no use while the results handed them over.
    """
    shown = [narrator_result_view(change) for change in changes]
    return (
        f"{_ATTRIBUTION}\n\nHere is what happened "
        f"(each entry is a tool call and its result):\n{json.dumps(shown)}"
        f"\n\nNew board state:\n{json.dumps(board_state)}\n\n"
        "React with a short, in-character comment for the player, based "
        "only on these results and the new board. Do not call any tools."
    )


def _rewrite_brief(commentary: str, corrections: Sequence[str]) -> str:
    """The narrator's brief for saying a reply again with the facts right.

    The reply comes back whole, so the rewrite can keep everything that was
    fine — the tone, the trash talk, the answer to what the player asked —
    and one line per unbacked claim says what is actually so (built by
    `honesty.corrections`, addressed to Glitch). It asks for a second draft
    and states facts; it does not scold, because a reprimand fed to a 12B
    produces an apology, and the player never heard the first draft.
    """
    facts = "\n".join(f"- {line}" for line in corrections)
    return (
        f"You were about to reply to the player with:\n{commentary}\n\n"
        f"Some of that is not what the board says. The facts:\n{facts}\n\n"
        "Say it again, in character, keeping everything that was right and "
        "making it fit those facts. Do not mention this correction or apologize "
        "for it — the player has not seen the first version. Do not call any "
        "tools."
    )


def _usage_ints(usage: Usage | None) -> tuple[int, int]:
    """`(prompt_tokens, completion_tokens)` from a completion's usage, or
    `(0, 0)` when llama-server omitted it — a missing count is not a failure,
    it adds nothing to the turn's totals, and the caller counts the call as
    unmetered so the record says the totals are a lower bound (#290)."""
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


# The label the board-refresh block carries. Deliberately a label and not a
# rule: it dates the block against the opening `Board state:` by naming what
# happened in between, and nothing more. Every measured arm that added a *fact*
# to what this model reads made the decision worse (docs/agent-evals.md), so
# supersession is left to recency and adjacency — the block sits immediately
# after the tool results that caused it, which narrate the same change in their
# own words.
_REFRESH_LABEL = "Board state after those tool calls:"


def _board_version_of(state: dict[str, Any] | None) -> int | None:
    """Which board a refresh view describes, or None when there is no view.

    The one key the brain reads out of the seam's answer. Everything else in
    there is for the model, but *whether to send it at all* is a question about
    the position, and this is the app's counter for that (`board_version`).
    """
    return None if state is None else state.get("board_version")


def _board_refresh_message(state: dict[str, Any]) -> dict[str, Any]:
    """One board refresh, as the message the planner reads it in.

    A `user` message and not a `tool` one: no tool produced this, a batch of N
    calls has no honest id for an N+1st result, and reusing the last call's id
    would break the one-answer-per-call shape both the wire and this loop's own
    tests keep — besides stapling board facts onto whatever happened to come
    last in the batch. The precedent in this loop is `_wire_correction`, for
    the same reason: the app has something to say and no call to say it under.
    """
    return {"role": "user", "content": f"{_REFRESH_LABEL}\n{json.dumps(state)}"}


def _append_user(messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
    """Append one loop-authored `user` message, merging into the last one when
    it is already the player's role.

    Chat templates are within their rights to reject two consecutive user
    turns, and the loop now has two things it can say in that role — a board
    refresh and a schema correction — which a single iteration can produce back
    to back. Merging keeps the rendered conversation alternating whatever order
    they arrive in.
    """
    if messages and messages[-1]["role"] == "user":
        messages[-1] = {
            "role": "user",
            "content": f"{messages[-1]['content']}\n\n{message['content']}",
        }
        return
    messages.append(message)


def _validate_call(
    call: ProviderToolCall, schemas: dict[str, dict[str, Any]]
) -> tuple[str, str] | None:
    """The schema-level complaint about a call and its `retry`, or None if the
    call is well-formed.

    The two words are the dispatcher's (`brain.RETRY_*`), and they are decided
    here rather than left to the caller because this is where the difference is
    known: a name outside the offer is not reachable by rewriting arguments —
    the offer is the offer, and a tool withheld for this command is not there
    to be argued with — while arguments the schema rejected are precisely what
    a corrected call would change.

    The provider has already guaranteed the arguments are a JSON object —
    malformed JSON never reaches here, it raised `ToolCallArgumentsError`
    upstream.
    """
    schema = schemas.get(call.name)
    if schema is None:
        return f"unknown tool: {call.name}", RETRY_NEVER
    try:
        jsonschema.validate(call.arguments, schema)
    except jsonschema.ValidationError as exc:
        return f"invalid args for {call.name}: {exc.message}", RETRY_DIFFERENT_ARGS
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
    tool_definitions: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]],
    system_prompt_provider: Callable[[], str] | None = None,
    planner_prompt_provider: Callable[[], str] | None = None,
    enable_thinking: bool = False,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    max_corrections: int = _DEFAULT_MAX_CORRECTIONS,
    planner_temperature: float | None = _PLANNER_TEMPERATURE,
    provider: ChatProvider | None = None,
    on_phase: Callable[[str], None] | None = None,
    board_refresh: Callable[[], dict[str, Any] | None] | None = None,
    narrator_facts: Callable[[], dict[str, Any] | None] | None = None,
) -> LlamaBrain:
    """Build a LlamaBrain against a real llama-server (e.g. localhost:8200/v1).

    `dispatcher` and `tool_definitions` should come from the same registry —
    the app assembly passes one `ToolRegistry` for both, so what the agent is
    offered is exactly what can be run. Like the prompts, `tool_definitions`
    may be a zero-arg callable resolved per command, so live state can change
    the offer itself (a draw becoming claimable adds `claim_draw`).

    Two prompts, because a turn is two phases: `system_prompt_provider` is the
    narrator's (the personality) and `planner_prompt_provider` is the loop's
    (the tool contract). Each defaults to being resolved once into a fixed
    string; pass a zero-arg callable — which the brain calls per command — so
    live settings changes (verbosity) take effect immediately (the
    app-assembly wires both to read `ctx.settings`). Either way the brain stays
    prompt-agnostic: it just carries a string or a callable.

    `planner_temperature` samples the planner phase apart from the narrator,
    `_PLANNER_TEMPERATURE` (0.3) unless a caller says otherwise; None leaves
    both on the provider's default.

    `provider` is injected in tests / alternate backends; otherwise the factory
    builds a real `LlamaCppProvider` against `base_url` + `model` (no API key —
    llama-server needs none).

    `on_phase` is told `planning` / `narrating` as each phase starts — the live
    progress seam (`progress.py`). Passed at construction rather than assigned
    later so the brain is complete when it is handed over, and optional because
    nothing about a turn depends on anyone listening.

    `board_refresh` is how the loop learns that its own tools moved the board
    (#282) — a zero-arg callable read once per planner iteration, answering
    `None` when nobody can say. A caller with no state injection omits it and
    gets exactly the loop that shipped before it existed.

    `narrator_facts` is the same kind of seam for the narrator (#289): read
    once as the planner hands off, it answers the side-free facts the narrator
    may state and whether the engine's reply is still owed.
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
        else PLANNER_PROMPT
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
        on_phase=on_phase,
        board_refresh=board_refresh,
        narrator_facts=narrator_facts,
    )
