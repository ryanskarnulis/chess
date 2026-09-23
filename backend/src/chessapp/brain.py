"""The swappable-brain seam.

Everything model-specific lives behind `Brain.get_agent_response`; nothing
outside a Brain implementation may know which model or backend answers.

The brain runs the agent loop — it calls the model, feeds each tool result
back, and keeps going until the model stops asking for tools — but it never
*executes* anything itself: every call it decides on goes out through a
`ToolDispatcher` (the validated `ToolRegistry`), which is what makes it
impossible for a brain to corrupt game state. The loop is bounded, and the
stop reason says how it ended (`completed | max_iterations |
correction_limit`, the fleet's vocabulary — `../agent-standard/STANDARD.md`
§3 — plus three chess additions: `provider_error` when the provider died
mid-turn (the response still carries every tool result that verifiably ran,
so the pipeline can close the turn and tell the truth instead of catching an
exception after the board changed, and `provider_failure` names *which*
death it was so a caller can tell one worth retrying from one that is not),
`no_progress` when the loop ended a planning phase that had started
repeating itself, and `budget` when a per-turn tool-work or wall-time budget
ended it (#288; `budget` on the response names which). What the player gets
is decided by what ran, not by the stop: a dead provider produces no
commentary, and a stop on any budget produces none when no tool did any work;
otherwise the narrator closes the turn like `completed` does — results came
back, so there is something verified to speak from).

How the words get written is the implementation's business, and
`LlamaBrain`'s answer is a second, tool-free model phase
(`docs/planner-narrator.md`). This seam only promises that `text` is what the
player may be shown and that it was produced from verified results.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from chessapp.handoff import Handoff

# Which phase of a turn made a model round trip (#317). The trace used to hold
# one unlabelled latency per call and left a reader to infer the phase from
# the route and the call order, which a budget stop or a provider death
# silently breaks. The brain names the phases it runs itself (`planner`,
# `closer`); the others are single-call seams whose *caller* knows which beat
# it is (`reaction`, `rewrite`, `answer`), so the pipeline stamps them.
PHASE_PLANNER = "planner"
PHASE_CLOSER = "closer"
PHASE_REACTION = "reaction"
PHASE_REWRITE = "rewrite"
PHASE_ANSWER = "answer"
# A call from a brain that does not tag its own (a test double): known to have
# happened, not known to have been which.
PHASE_UNKNOWN = "unknown"

# How one round trip ended. `truncated` came back cut off by its `max_tokens`
# (the tokens are real, the words were dropped); `bad_args` came back with
# tool arguments that were not a JSON object; `failed` raised; `late` was still
# running when its caller stopped waiting — not a failure of the provider, and
# its `ms` is the wait, a censored reading rather than the call's duration.
CALL_OK = "ok"
CALL_TRUNCATED = "truncated"
CALL_BAD_ARGS = "bad_args"
CALL_FAILED = "failed"
CALL_LATE = "late"


@dataclass(frozen=True)
class ServerStamp:
    """What the server said about the one call it served (#317), in plain
    values so the seam stays model-agnostic (`provider.ServerMeta` is the
    wire's reading of it). `fingerprint` is the server build that answered,
    `server_ms` its own prompt plus generation time — the caller's wall clock
    minus this is queueing, transport and any cold load — and `cached_tokens`
    how much of the prompt the KV cache served. `None` when the server did not
    say."""

    fingerprint: str | None = None
    server_ms: int | None = None
    cached_tokens: int | None = None


@dataclass(frozen=True)
class ModelCall:
    """One model round trip, as the trace records it (#317).

    Every call a turn made is one of these, whichever phase made it and however
    it ended — which is what lets a reader tell a slow planner from a slow
    narrator without guessing from the call order, and a failed call from a
    cheap one. The turn's totals (`model_calls`, the token sums, `model_ms`)
    are these added up, never a second count kept beside them.

    `prompt_tokens`/`completion_tokens` are `None` when unknown — the call
    raised, or the server sent no usage — never a zero that reads as measured.
    `failure` is the `provider.ProviderFailure` kind of a `failed` call, and
    `budget_ms` the wait the caller held this call to, when it held it to one:
    beside a `late` status it says what the censored `ms` was censored at.
    """

    phase: str
    status: str = CALL_OK
    ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    failure: str = ""
    budget_ms: int | None = None
    # The server's own account of the call, when it gave one (#317).
    server: ServerStamp | None = None

    @property
    def metered(self) -> bool:
        return self.prompt_tokens is not None and self.completion_tokens is not None

    def as_trace(self, seq: int) -> dict[str, Any]:
        """This call as one entry of the record's `calls` list. `seq` is its
        position in the turn: with the record's `correlation_id` it names the
        attempt, so a retried or duplicated call is two entries, never one."""
        return {
            "seq": seq,
            "phase": self.phase,
            "status": self.status,
            "ms": self.ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "failure": self.failure,
            "budget_ms": self.budget_ms,
            "fingerprint": self.server.fingerprint if self.server else None,
            "server_ms": self.server.server_ms if self.server else None,
            "cached_tokens": self.server.cached_tokens if self.server else None,
        }


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the brain wanted: a name and the args it passed."""

    name: str
    args: dict[str, Any]


# The two answers to "is calling again worth a round trip?" — the `retry` key on
# every refusal a dispatcher produces. Two rather than a scale because the agent
# only ever does one of two things with the answer: fix the arguments and call
# again, or stop and tell the player. `never` is the default for an
# unclassified failure: a loop that does not know why it failed must not spend
# its budget finding out.
#
# They live beside `ToolDispatcher` rather than in `tools.py` because they are
# part of that protocol's contract, not the tool layer's private vocabulary: a
# brain that builds a refusal (the loop answers a call it never dispatched)
# needs the same two words, and the seam is what both sides share. `tools.py`
# re-exports them, so every `tools.RETRY_*` reference still reads from the one
# definition.
RETRY_DIFFERENT_ARGS = "different_args"
RETRY_NEVER = "never"


class ToolDispatcher(Protocol):
    """Whatever executes a named tool call and answers with a result dict.

    `ToolRegistry` (`tools.py`) satisfies this structurally — the brain is
    handed one at assembly and never imports the tool layer. It never raises
    on an agent-caused fault: a bad call comes back as error *data* the model
    can read and correct from.
    """

    def dispatch(self, name: str, args: Any) -> dict[str, Any]: ...

    def refusal(self, error: str, retry: str, **details: Any) -> dict[str, Any]:
        """One "no" in the shape every refusal from this dispatcher takes.

        Part of the protocol because the loop has its own refusals to make: a
        call whose name is not in the offer, or whose arguments the schema
        rejects, is never dispatched, and a "no" the loop wrote by hand used to
        be the one failure in the turn carrying neither `retry` nor
        `board_version` — while the planner's contract promises every failure
        says how to recover from it. The dispatcher owns that shape, so the
        loop asks for it rather than imitating it.
        """
        ...


@dataclass(frozen=True)
class AgentResponse:
    """One finished agent run.

    `text` is the user-facing commentary — spoken by the narrator phase from
    the turn's verified results (empty when the loop stopped on a budget
    before any tool did work, in which case no narrator ran and the pipeline
    substitutes its stuck reply). Every call the loop made and ran is in
    `tool_calls`, with `tool_results` holding each one's `{"name", "result"}`
    in the *same order*: the two are parallel by construction, and the
    delegate wire zips them strictly.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    stop_reason: str = "completed"
    # Only meaningful beside `stop_reason == "provider_error"`, and empty
    # otherwise: *which* kind of provider failure ended the turn. A dead socket
    # and a refused request stop a turn identically but want opposite handling —
    # one is worth asking again, the other is worth asking about — and the
    # implementation used to swallow the distinction with the exception. A
    # plain string for the same reason the token counts are plain ints: the
    # seam stays model-agnostic, so the provider's enum never crosses it (the
    # values are `provider.ProviderFailure`'s).
    provider_failure: str = ""
    # The run's cost at the provider boundary: how many times the model was
    # called and the tokens summed across those calls. Plain ints — the seam
    # stays model-agnostic, so the provider's `Usage` type never crosses it.
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # How many of those calls reported no token usage — they raised, or the
    # server sent none (#290). The token totals are what *was* measured, so a
    # non-zero count here says they are a lower bound, never a measured zero.
    unmetered_calls: int = 0
    # One wall-clock reading per model call, in call order — parallel to
    # `model_calls` by construction, including the calls that raised (a round
    # trip that died still spent the time). Empty from a brain that doesn't
    # measure: an unmeasured turn is not a fast one, so it records no readings
    # rather than zeros.
    model_latencies_ms: tuple[int, ...] = ()
    # The same calls, one `ModelCall` each, in call order and tagged with the
    # phase that made them and how they ended (#317). The fields above are
    # these summed; a brain that does not tag its calls leaves this empty and
    # the pipeline records its readings under `PHASE_UNKNOWN`.
    calls: tuple[ModelCall, ...] = ()
    # How the planning phase spent its wall clock against its deadline (#317):
    # `{"elapsed_ms", "deadline_ms", "overrun_ms"}`, or `None` when no planner
    # call ran. The deadline is checked *between* round trips, so the last one
    # may start inside it and end past it; `overrun_ms` is by how much, which is
    # the one number the check itself can never show.
    planning: dict[str, int | None] | None = None
    # The board versions the planner was shown *during* the run, in order —
    # one per mid-command refresh of its state block (#282). Empty on a turn
    # that mutated nothing, and equally on one whose mutation left the board
    # mid-exchange, where the refresh is withheld by design; `mutations` beside
    # it in the trace is what tells those two apart. Plain ints, like the token
    # counts, so the seam stays model-agnostic.
    state_refreshes: tuple[int, ...] = ()
    # The board versions at which the planner's *offer* changed with that
    # refresh (#315) — `ask_player`'s candidates re-narrowed, or a tool that
    # appeared or went. A subset of `state_refreshes`: a refresh whose menu
    # left the offer as it was swaps nothing, and a swap is what costs the
    # planner a re-read of its prompt.
    offer_refreshes: tuple[int, ...] = ()
    # What the narrator was told the turn did (#289): the results sorted into
    # done / refused / looked up, the kind derived from them, and whether the
    # engine's reply was still owed as it spoke. `None` on a turn no narrator
    # closed — a budget stop with nothing done, or a provider that died before
    # the plan finished.
    handoff: Handoff | None = None
    # Which per-turn budget ended the planning phase (#288), or "" when none
    # did: `iterations`, `corrections`, `tool_calls`, `analysis_calls`,
    # `wall_time` or `input`. The stop reason says the phase ended early; this says on
    # what, which is the number a trace reader tunes. `input` is the prompt
    # budget: the run's own results outgrew it.
    budget: str = ""
    # How many of the conversation's oldest exchanges were dropped to fit the
    # input budget (#288); 0 on every turn that fit, which is every real one.
    input_trimmed: int = 0
    # True when the closing narration was still being written when the brain
    # stopped waiting for it (#316): the plan ran and its record stands, the
    # words are gone. `text` is empty, and it is not a provider failure — the
    # model may well have answered, just not before the turn went on.
    narration_late: bool = False


@dataclass(frozen=True)
class Narration:
    """One narrator turn: commentary on work already done, with no tools on
    offer. `text` is what the player sees; the cost fields mirror
    `AgentResponse`'s so a narrated turn reaches the trace with the same
    accounting a looped one does. It is always exactly one model call."""

    text: str
    model_calls: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    # 1 when the call returned no usage, so its token counts are unknown.
    unmetered_calls: int = 0
    # How the one call ended (#317): `CALL_OK`, or `CALL_TRUNCATED` when the
    # cap cut it off and `text` is empty for that reason. Which phase it was
    # is the caller's to say — the same narrator serves three beats.
    status: str = CALL_OK
    # What the server said about the call (#317), when it said anything.
    server: ServerStamp | None = None

    @property
    def model_latencies_ms(self) -> tuple[int, ...]:
        """The one call's latency, under the name `AgentResponse` uses — so
        whatever reads a turn's cost reads both shapes the same way. Empty for
        a narration that never reached the model (its brief did not fit)."""
        return (self.latency_ms,) if self.model_calls else ()


# How a reply to a pending confirmation question can read. Deliberately three
# and not two: "actually, undo my last move instead" is neither a yes nor a no,
# and reading it as either would answer a question the player has stopped
# asking.
CONFIRM = "confirm"
CANCEL = "cancel"
UNRELATED = "unrelated"


@dataclass(frozen=True)
class Answer:
    """How the model read a reply to a pending destructive-op question.

    The narrow seam behind walkthrough #6: "just do it" after a resign
    question did nothing, because the only reader was a set of literal
    affirmations. Understanding what the player meant is the model's job —
    *acting* on it stays the pipeline's, which runs the armed op through the
    same confirm path a bare yes takes, so the model still cannot reach a
    destructive tool from here. All it returns is one of three words.

    `verdict` defaults to `UNRELATED`, the answer that changes nothing: an
    implementation that cannot reach the model, or one that has nothing to say,
    must never land on `CONFIRM` by omission. The cost fields mirror
    `Narration`'s so the trace accounts for this round trip like any other.
    """

    verdict: str = UNRELATED
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    unmetered_calls: int = 0
    # How the one call ended (#317) — `CALL_FAILED` with the provider's
    # `failure` kind for a reader that died, `CALL_TRUNCATED` for one the cap
    # cut off. Either way the verdict is `unrelated`.
    status: str = CALL_OK
    failure: str = ""
    server: ServerStamp | None = None

    @property
    def model_latencies_ms(self) -> tuple[int, ...]:
        return (self.latency_ms,) if self.model_calls else ()


@dataclass
class _RunState:
    """The loop's accumulator: what has been called, what came back, what it cost."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    unmetered_calls: int = 0
    latencies_ms: list[int] = field(default_factory=list)
    calls: list[ModelCall] = field(default_factory=list)
    # The planning phase against its deadline (see `AgentResponse.planning`).
    planning: dict[str, int | None] | None = None
    boards_shown: list[int] = field(default_factory=list)
    offers_refreshed: list[int] = field(default_factory=list)
    # Which turn budget ended the planning phase (#288), or "" when none did.
    budget: str = ""
    # How many of the conversation's oldest exchanges the input budget dropped
    # from this run's prompts (#288).
    input_trimmed: int = 0

    def record(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        self.tool_calls.append(ToolCall(name=name, args=args))
        self.tool_results.append({"name": name, "result": result})

    def show_board(self, version: int) -> None:
        """Note that the planner was handed the board as of `version` — one
        reading per mid-command state refresh, in the order they were sent."""
        self.boards_shown.append(version)

    def refresh_offer(self, version: int) -> None:
        """Note that the planner's offer was re-resolved for `version`."""
        self.offers_refreshed.append(version)

    def count_call(self, call: ModelCall) -> None:
        """Tally one model round trip: its phase, how it ended, its tokens and
        its wall clock. Called for every trip — including one that raised
        before returning a result (it still cost a call, and usually the most
        time), which arrives with no tokens: a real latency, and a count that
        says the tokens are unknown rather than zero. The totals are summed off
        the calls here, so the two can never disagree."""
        self.calls.append(call)
        self.model_calls += 1
        self.unmetered_calls += 0 if call.metered else 1
        self.prompt_tokens += call.prompt_tokens or 0
        self.completion_tokens += call.completion_tokens or 0
        self.latencies_ms.append(call.ms)

    def response(
        self,
        text: str,
        stop_reason: str,
        provider_failure: str = "",
        handoff: Handoff | None = None,
        *,
        narration_late: bool = False,
    ) -> AgentResponse:
        return AgentResponse(
            text=text,
            tool_calls=tuple(self.tool_calls),
            tool_results=tuple(self.tool_results),
            stop_reason=stop_reason,
            provider_failure=provider_failure,
            model_calls=self.model_calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            unmetered_calls=self.unmetered_calls,
            model_latencies_ms=tuple(self.latencies_ms),
            calls=tuple(self.calls),
            planning=self.planning,
            state_refreshes=tuple(self.boards_shown),
            offer_refreshes=tuple(self.offers_refreshed),
            handoff=handoff,
            budget=self.budget,
            input_trimmed=self.input_trimmed,
            narration_late=narration_late,
        )


class Brain(Protocol):
    def get_agent_response(
        self,
        board_state: dict[str, Any],
        command: str,
        transcript: Sequence[dict[str, str]] = (),
    ) -> AgentResponse:
        """Run the agent loop for one utterance: turn it into tool calls, run
        them through the dispatcher, feed the results back, and stop on the
        first turn that asks for no tools. The commentary is then written from
        the turn's results by a phase that holds no tools, so it cannot act on
        the utterance a second time. `board_state` is the agent-facing view
        (fen, turn, player_color,
        in_check, SAN history, captured, legal_moves, game_over/outcome — not
        the UI state document), captured before the loop runs. A later change
        the loop's own tools make reaches it two ways: in the tool results
        themselves, and — for the legal-move menu, which no result reports —
        through whatever board-refresh seam the implementation was wired with
        (#282). A brain given no such seam works from the opening view alone.
        `transcript` is the prior conversation as chat messages (final answers
        only) so the agent can follow references to earlier turns. How far back
        it reaches and in what form is the app's memory policy, not the brain's:
        what actually arrives is `Transcript.memory()` — the last few turns
        verbatim behind a digest of the older asks (`docs/turn-memory.md`) — and
        a brain neither knows nor needs to know which of them were condensed."""
        ...

    def narrate(
        self,
        board_state: dict[str, Any],
        changes: list[dict[str, Any]],
        transcript: Sequence[dict[str, str]] = (),
    ) -> Narration:
        """Commentary on a move the loop did not make: the deterministic fast
        path (`parse_move` → `make_move`) skips the planner entirely, so there
        is no turn for the narrator to close. This is the narrator phase on its
        own — the *new* board plus `changes` (each a `{"name", "result"}` tool
        result), no tools offered, no access to the raw utterance. What the
        board view contains is the caller's policy, and the app deliberately
        hands a view with no side to play for — no turn, no legal moves, no
        FEN (`api._narrator_state_dict`): the beat runs while the engine's
        reply is still being computed, and a narrator that can see whose move
        it is announces one. It exists
        because the fast path is deliberately outside the loop; at verbosity=low
        even this is skipped for a canned confirmation, making a plain move
        zero-LLM."""
        ...

    def rewrite(
        self,
        commentary: str,
        corrections: Sequence[str],
        transcript: Sequence[dict[str, str]] = (),
    ) -> Narration:
        """Say `commentary` again with the facts in `corrections` right.

        The honesty guard's second try. The pipeline checks every operational
        claim in the narrator's text against the board and the tool results
        (`honesty.unverified`); when one is not backed, this is what happens
        next — the narrator is handed its own reply and one plain sentence
        per unbacked claim saying what is actually so, and writes the reply
        again. Same persona, same conversation, no tools, so the second draft
        is Glitch's words and not the app's, and the phase still cannot act.

        The pipeline checks the rewrite too and falls back to the
        deterministic facts if it still asserts something the board does not
        back; an implementation is never asked twice. A `ProviderError` here
        costs the words and nothing else — the turn is already settled."""
        ...

    def read_answer(self, question: str, text: str) -> Answer:
        """Read a reply to a pending destructive-op question as one of
        `CONFIRM` / `CANCEL` / `UNRELATED`.

        Called only after the deterministic reader (`fastparse.parse_confirmation`)
        has already declined the utterance, and only while an op is armed. No
        tools, so this phase cannot act on what it reads — the pipeline runs the
        armed op through the same confirm path a bare yes takes, or drops it.

        `question` is what the player is being asked, in the app's own words, so
        the reading is about *this* question and not about the utterance in the
        abstract: "no, the other one" answers a choice and cancels a
        resignation.

        An implementation that cannot answer returns `UNRELATED`. Failing
        towards "nothing happened" is the whole shape of the destructive gate,
        and a dead provider must never be able to end a game."""
        ...
