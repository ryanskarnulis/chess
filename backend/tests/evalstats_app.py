"""Chess's half of the eval gate's arithmetic: what the fleet's verdict does
not know about chess.

`evalstats.py` beside this file is agent-standard's canonical verdict, vendored
verbatim (`../agent-standard/STANDARD.md` §6) with its spec `test_evalstats.py`
— never edited here, re-copied to fix drift. It was cut from this repo's own
module, and what it left behind is chess's and lives here, on top of it:

- **Vacuous passes are not passes.** A budget stop reaches no narrator, so a
  purely negative commentary check passes for never having been tested (three
  of four `hints_off_no_advice` passes in one recorded run). `VacuousRun` names
  that, and `classify` reads it as `Outcome.INCONCLUSIVE`, which canonical's
  `Outcome` does not have: a counted non-pass, reported as what it was.
- **Five failures naming five moves are one mode.** Canonical's
  `failure_signature` collapses numbers, so "Bxe6" would read "Bxe<n>"; chess
  collapses whole SAN first, and `RateResult` derives its failure modes, and
  so `deterministic_suspect`, from that signature. It also carries the
  `inconclusive` count.
- **Which phase spent a turn's milliseconds and tokens** (`split_latencies`,
  `split_tokens`). The readings were always on the trace per call; the
  attribution off a route, a stop reason and the calls' phase tags is a
  decision, and decisions belong somewhere a test can run them.

The import name is `evalstats`, never `tests.evalstats`: chess's tests and
scripts import their neighbours as top-level modules (`tests/` has no
`__init__.py`), and `tests.evalstats` exists only so the vendored spec can
import the module the way every fleet repo spells it. The two names load two
module objects with two `Outcome` classes, so this module, the harness and the
scripts all say `evalstats`, and every name defined here over a canonical one
(`Outcome`, `classify`, `RateResult`, `failure_signature`, `scenario_record`)
must be imported from here. `test_evalstats_app.py` holds this module's spec,
the drift guards tying canonical's literals to chess's loop and provider, and
a check that no chess file imports one of those names from the wrong module.

Like canonical, nothing here knows about pytest, HTTP or the app: the
harness is the only thing that may touch the model, and these two modules are
the only things that may decide what its samples mean.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import evalstats
from evalstats import Z_ONE_SIDED_95

# The loop's own stop reason for a provider that died mid-turn, as `llama_brain`
# produces it — and the only one of canonical's `INFRA_STOPS` chess's loop has
# (its wall-time budget ends as `budget`, scored as behavior, never as
# `timed_out`). Named here rather than imported: this module must stay
# importable with nothing installed, and `test_evalstats_app` reads the loop's
# stop vocabulary off its code and pins both facts.
STOP_PROVIDER_ERROR = "provider_error"

# The loop's *budget* stops, as opposed to the two stops that still reach the
# narrator (`completed`, `no_progress`). A turn that ended on one of these
# produced no commentary at all, which is what makes it both an untested sample
# under a commentary-only check (`Outcome.INCONCLUSIVE`) and a turn with no
# narrator round trip to attribute time to (`Attribution.NO_NARRATOR`). One set,
# because those two readings must never disagree about what a stop means.
BUDGET_STOPS = frozenset({"max_iterations", "correction_limit"})

# The trace's route vocabulary, split by the only thing this module needs from
# it: how many model *phases* the route can have spent time in. The brain route
# runs the planner loop and then the narrator; every other route reaches the
# model only through `Brain.narrate` — one narrator call, no planner at all.
# Literals rather than imports (this module must stay importable with nothing
# installed), and `test_evalstats_app.test_the_route_vocabularies_agree` is what
# pays for the copy.
ROUTE_BRAIN = "brain"
# The one narrate route with a *second* kind of model call in front of it: the
# confirmation reader (`llama_brain.read_answer`). Named so the attribution can
# single it out; still a narrate route, because when it spends one round trip
# that round trip may equally have been the reader or the narrator.
ROUTE_CONFIRMATION = "confirmation"
NARRATE_ROUTES = frozenset(
    {"fast_path", "resign", "board", ROUTE_CONFIRMATION, "control"}
)


class Outcome(StrEnum):
    """What one sample was, before it is allowed to count as anything.

    Canonical's `evalstats.Outcome`, value for value, plus INCONCLUSIVE. A
    StrEnum cannot be extended, so this is its own class: a member compares
    *equal* to canonical's of the same name (both are the string), but never
    `is` it, which is why chess code takes `Outcome` and `classify` from here
    and nowhere else. `test_evalstats_app` pins the values.

    Precedence, highest first: HARNESS, then INFRA / PROVIDER_REJECTED, then
    INCONCLUSIVE, then FAIL, then PASS.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    # The provider died (or the route rate-limited us). Retried, never scored.
    INFRA = "INFRA"
    # The provider failed in a way a retry cannot fix — an HTTP 400 (a context
    # overrun on a long transcript), a body this app can no longer parse.
    # Never retried, never scored: the harness fails the item on the *first*
    # one, instead of spending the infra budget re-asking a question already
    # answered.
    PROVIDER_REJECTED = "PROVIDER_REJECTED"
    # The turn never reached the thing the check was about (a budget stop under
    # a commentary-dependent check). Neither a pass nor a miss — an untested
    # sample, counted against the rate and reported as what it was.
    INCONCLUSIVE = "INCONCLUSIVE"
    # A bug in this repo (a KeyError in a check). Never retried, never scored,
    # fails the item immediately — a retry cannot fix a typo.
    HARNESS = "HARNESS"


class VacuousRun(AssertionError):
    """The turn never reached what the check was about, so the check tested
    nothing.

    An `AssertionError` subclass on purpose: `classify` reads it as
    INCONCLUSIVE, but a caller that skips this module's `classify` — canonical's
    included — gets a *failed* sample rather than a silent pass. Failing safe is
    the whole point — the bug this names is a check that passed without running.
    """


def classify(
    *,
    status_code: int,
    stop_reason: str | None,
    error: BaseException | None,
    provider_failure: str | None = None,
) -> Outcome:
    """What kind of sample this was: canonical `evalstats.classify`, plus the
    one reading chess adds.

    Canonical decides everything it knows — a harness bug, then a provider
    death split on `provider_failure` (the kind the brain named,
    `AgentResponse.provider_failure`, whose two deterministic values are
    canonical's `DETERMINISTIC_FAILURES`), then a failed assertion or a non-200
    that isn't infrastructure as a real miss. Then one change: a FAIL whose
    error is a `VacuousRun` on a turn that answered (200) is INCONCLUSIVE.
    `VacuousRun` being an `AssertionError` is what puts it under infrastructure
    and over a plain miss without restating canonical's precedence here. A
    non-200 stays the miss it is, whatever the check then said about it.
    """
    outcome = evalstats.classify(
        status_code=status_code,
        stop_reason=stop_reason,
        error=error,
        provider_failure=provider_failure,
    )
    if (
        outcome is evalstats.Outcome.FAIL
        and status_code == 200
        and isinstance(error, VacuousRun)
    ):
        return Outcome.INCONCLUSIVE
    return Outcome(outcome.value)


# SAN and bare squares first, then canonical collapses any remaining number:
# "Bxe6" must collapse as one token, not to "Bx<square>" and certainly not to
# "Bxe<n>".
_SAN = re.compile(
    r"\b(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b"
)


def failure_signature(message: str) -> str:
    """One failure message, normalised so identical *modes* collapse.

    Five failures naming five different moves are one finding; five different
    messages are five. Canonical's signature over the message with its moves
    already collapsed, so every rule canonical applies after that (numbers,
    whitespace) is canonical's. Normalised rather than hashed, because the
    signature goes in the report and the printed summary and has to stay
    recognisable.
    """
    return evalstats.failure_signature(_SAN.sub("<san>", message))


@dataclass(frozen=True)
class RateResult(evalstats.RateResult):
    """What sampling one scenario measured, and what it means — canonical's
    record plus the `inconclusive` count, with its failure modes read by
    chess's signature.

    Every derived field is still built by `from_blocks` from the counts, never
    passed in: a record whose verdict disagrees with its counts is worse than
    no record.
    """

    # Samples that never reached what their check was about. Counted in `runs`
    # as non-passes (and in `failures`), and printed beside `infra` so a rate
    # with untested samples in it reads as one.
    inconclusive: int = 0

    @classmethod
    def from_blocks(
        cls,
        *,
        blocks: Sequence[tuple[int, int]],
        failures: Sequence[str],
        floor: float,
        infra: int = 0,
        inconclusive: int = 0,
        z: float = Z_ONE_SIDED_95,
    ) -> RateResult:
        # Canonical derives `failure_modes` and `deterministic_suspect` from its
        # own number-only signature. Handed the failures already signed the
        # chess way, it derives both from chess's signature instead — its own
        # leaves a chess signature unchanged, which `test_evalstats_app` pins —
        # and the messages themselves go back on the record afterwards. So both
        # fields are canonical's rule over chess's modes, and neither can
        # disagree with the other or with the counts.
        signed = super().from_blocks(
            blocks=blocks,
            failures=[failure_signature(failure) for failure in failures],
            floor=floor,
            infra=infra,
            z=z,
        )
        return replace(signed, failures=tuple(failures), inconclusive=inconclusive)

    def summary(self) -> str:
        """The one-line `[eval]` verdict a human reads off the terminal:
        canonical's, with the inconclusive count straight after the infra one,
        where the chess line has always carried it."""
        head, flag, tail = super().summary().partition(" DETERMINISTIC_SUSPECT")
        return f"{head} inconclusive={self.inconclusive}{flag}{tail}"


def scenario_record(
    scenario: str, result: RateResult, floor: float, **meta: Any
) -> dict[str, Any]:
    """One scenario's verdict as a JSON-serialisable record: canonical's, plus
    the inconclusive count.

    Written as it finishes, one line per scenario, so a baseline stops being
    hand-transcribed from terminal scrollback and a Ctrl-C mid-suite still
    leaves every finished scenario on disk. `**meta` carries whatever the call
    site knows and this module must not — per-sample latencies, the seam,
    whether the item is release-blocking.
    """
    return evalstats.scenario_record(
        scenario, result, floor, inconclusive=result.inconclusive, **meta
    )


# --- phase attribution --------------------------------------------------------


class Attribution(StrEnum):
    """How much of a turn's per-call latencies could be pinned to a phase."""

    # Planner and narrator are both known: the readings split at the phase
    # boundary the route and stop reason put there.
    SPLIT = "SPLIT"
    # A budget stop: no narrator ran, so every reading is the planner's and
    # there is no narrator time — not an unmeasured one, none. Also a tagged
    # turn whose only call was the confirmation reader (#317): no narrator
    # ran there either, and the reader is neither phase.
    NO_NARRATOR = "NO_NARRATOR"
    # No readings came back: either the model was never called (a canned
    # confirmation at verbosity=low) or nothing was traced at all (a request that
    # died before the pipeline's trace point). The sample's own outcome tells
    # those two apart — a thrown-away INFRA sample from a measured zero-LLM turn.
    NONE = "NONE"
    # The boundary is not knowable — the provider died on a call that could have
    # been either phase, the route is one this module has never heard of, or a
    # confirmation turn spent more than one round trip and the record cannot say
    # which of them was the reader and which the narrator. Reported rather than
    # guessed: a median over guesses is worse than a median over fewer samples.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TurnLatencies:
    """One turn's model round trips, and which phase spent each of them.

    `call_ms` is the turn's readings in call order — `model_latencies_ms` off the
    trace, one per round trip including the ones that raised. `total_ms` is
    derived here rather than carried, the same rule `turn_record` follows for
    `model_ms`: a record whose total disagrees with its parts is worse than no
    record.

    `planner_ms` and `narrator_ms` are `None` when the phase boundary is not
    knowable (see `Attribution`), never 0 — an unmeasured phase is not a fast
    one. The one real zero is a narrate route's planner, which genuinely never
    ran.
    """

    call_ms: tuple[int, ...]
    attribution: Attribution
    planner_ms: int | None
    narrator_ms: int | None

    @property
    def total_ms(self) -> int:
        """The turn's whole model time — what the trace calls `model_ms`. A fact
        on every turn, including the ones whose split is unknown."""
        return sum(self.call_ms)

    def summary(self) -> str:
        """The latency clause of the `[eval]` line a human reads. An unknown
        prints as `?`, not as a number, because the point is that it is not one."""
        return (
            f"call_ms=[{_readings(self.call_ms)}] "
            f"planner_ms={_known(self.planner_ms)} "
            f"narrator_ms={_known(self.narrator_ms)} phases={self.attribution}"
        )

    def as_record(self) -> dict[str, Any]:
        """The per-sample latency fields for the JSONL report.

        This is the half a campaign reads: `docs/agent-evals.md` has claimed
        per-sample `model_ms` since the statistics slice while the report carried
        none, so the medians in the repeat-stop finding were read out of terminal
        scrollback. Now the split ships with them, per sample, per scenario."""
        return {
            "model_ms": self.total_ms,
            "call_ms": list(self.call_ms),
            "planner_ms": self.planner_ms,
            "narrator_ms": self.narrator_ms,
            "phases": str(self.attribution),
        }


_WHOLE_TURN = slice(None)

# The phase tags a schema-2 trace puts on each call (`brain.PHASE_*`, #317),
# restated here as strings because this module reads records, not the app. The
# narrator's phase is `narrator` (#369); records from before it say `closer`
# for the loop's narration and `reaction` for a narrate route's, and records
# from before #368 add the honesty guard's `rewrite` — the same persona call. The
# reader (`answer`) is neither phase, which is exactly the call the positional
# rule could never place, and nor is the mover (`mover`, #472).
# `test_evalstats_app` pins these against `brain.PHASE_*`.
_PLANNER_PHASE = "planner"
_NARRATOR_PHASES = frozenset({"narrator", "closer", "reaction", "rewrite"})
_KNOWN_PHASES = _NARRATOR_PHASES | {_PLANNER_PHASE, "answer", "mover"}

# Which calls belong to a phase: a slice when derived from call positions, the
# indices themselves when read off the tags (a reader call can sit in front of
# a narrator one, so a tagged phase need not be contiguous).
type _Phase = slice | tuple[int, ...]


def _attribute_tagged(
    phases: Sequence[str] | None, calls: int
) -> tuple[Attribution, _Phase | None, _Phase | None] | None:
    """The phases read off the calls' own tags, or `None` when the record has
    none to read (schema 1), has a different number of them than readings, or
    carries a call a brain could not tag — then the positional rule decides,
    as it always did. A tag is the brain saying which phase ran, so nothing
    here depends on the route or the stop reason, and a provider death is no
    longer an unknown."""
    if phases is None or len(phases) != calls or calls == 0:
        return None
    if any(phase not in _KNOWN_PHASES for phase in phases):
        return None
    planner = tuple(i for i, phase in enumerate(phases) if phase == _PLANNER_PHASE)
    narrator = tuple(i for i, phase in enumerate(phases) if phase in _NARRATOR_PHASES)
    if not narrator:
        return Attribution.NO_NARRATOR, planner, None
    return Attribution.SPLIT, planner, narrator


def _attribute_phases(
    calls: int, *, route: str | None, stop_reason: str | None
) -> tuple[Attribution, slice | None, slice | None]:
    """Which of a turn's per-call readings belong to which phase.

    The rule itself, stated once over call *positions* and nothing else, so that
    every kind of per-call reading — milliseconds off the trace, tokens off the
    call meter — is attributed identically. Two clauses on one `[eval]` line
    disagreeing about which call was the narrator would make the line unreadable
    in exactly the case it is there to explain.

    `None` for a phase means the boundary is not knowable, and an *empty* slice
    means the phase genuinely never ran (a narrate route's planner). The
    difference is the whole point: an unmeasured phase is not a free one.

    See `split_latencies` for why each branch is what it is.
    """
    if calls == 0:
        # No round trips: nothing to attribute, and zeros would read as a turn
        # that planned and narrated instantly.
        return Attribution.NONE, None, None
    if route == ROUTE_CONFIRMATION and calls > 1:
        # The one narrate route that can spend more than one round trip, and
        # the record cannot say where the boundary falls. Since the free-text
        # confirmation reader landed, a confirmation turn is any of three
        # shapes: reader only (verbosity=low, or a cancel), reader *and*
        # narrator (a free-text confirm at normal verbosity), or narrator only
        # (a literal "yes" at normal). Two calls fit the second and could in
        # principle fit a re-narration; nothing on the trace distinguishes
        # them, and calling all of it narrator time — which this branch used to
        # — reports the reader's round trip as narration and inflates every
        # narrator median that includes a confirmation.
        #
        # UNKNOWN rather than a guess, and rather than inventing a reader
        # phase: `Attribution` says how much was knowable, and a third phase
        # would need `AgentResponse` to carry which calls were the reader's,
        # which is a production change this module does not get to make.
        return Attribution.UNKNOWN, None, None
    if route in NARRATE_ROUTES:
        # One narrator call and no planner phase — including a call that died,
        # which spent its time in the narrator all the same. A lone
        # confirmation call comes through here too: it is one round trip either
        # way, so "all of it, and no planner" is true whichever phase made it.
        return Attribution.SPLIT, slice(0, 0), _WHOLE_TURN
    if route != ROUTE_BRAIN or stop_reason == STOP_PROVIDER_ERROR:
        return Attribution.UNKNOWN, None, None
    if stop_reason in BUDGET_STOPS:
        return Attribution.NO_NARRATOR, _WHOLE_TURN, None
    if calls < 2:
        # A brain turn that completed ran a planner call and a narrator call:
        # fewer is a record that contradicts itself, and a guess would be
        # reported as a measurement.
        return Attribution.UNKNOWN, None, None
    return Attribution.SPLIT, slice(0, calls - 1), slice(calls - 1, calls)


def _total(readings: Sequence[int | None], phase: _Phase | None) -> int | None:
    """One phase's readings summed, or `None` for either honest unknown: the
    phase boundary was not knowable, or one of the phase's own calls reported no
    reading at all (a round trip that raised has no usage). Summing the rest
    would report a *smaller* number as though it were the whole."""
    if phase is None:
        return None
    values = (
        readings[phase]
        if isinstance(phase, slice)
        else tuple(readings[i] for i in phase)
    )
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _readings(values: Sequence[int | None]) -> str:
    """A per-call array for the `[eval]` line, unknowns as `?`."""
    return ",".join("?" if value is None else str(value) for value in values)


def _known(value: int | None) -> str:
    return "?" if value is None else str(value)


def split_latencies(
    call_ms: Sequence[int],
    *,
    route: str | None,
    stop_reason: str | None,
    phases: Sequence[str] | None = None,
) -> TurnLatencies:
    """Attribute a turn's per-call readings to the phase that spent them.

    `model_ms` is a per-turn sum, and the question it cannot answer is the one
    Sprint 5 is asking: a `no_progress` turn was measured narrating 2–3× slower
    than a `completed` one at the *same* model-call count (40.5 s and ≈26.7 s
    medians against 9.9–11.5 s). Whether that is the narrator's own
    round trip or a hard sample that made both the repeat and the long narration
    happen needs planner time told from narrator time, which the readings can
    give and the total cannot.

    The attribution is derived, never guessed, from two things the trace already
    records:

    - **the route** decides how many phases existed. `brain` runs the planner
      loop then one narrator call; every other route reaches the model only
      through `Brain.narrate`, so its planner time is a real 0. The exception is
      a `confirmation` turn that spent more than one round trip: the reader runs
      in front of the narrator there and the record does not say where one ends,
      so that one is UNKNOWN rather than all-narrator (audit 2026-09-05,
      "Statistical interpretation").
    - **the stop reason** decides whether the last call was the narrator. It is
      on `completed` and `no_progress` (both reach it), it does not exist on a
      budget stop, and on `provider_error` the dead call could have been either
      phase — the other case where the honest answer is UNKNOWN.

    Both are fallbacks now (#317): a schema-2 record tags each call with the
    phase that made it, and `phases` — the record's `calls[].phase`, in order —
    is read before either rule is consulted. The rules still decide a schema-1
    record, and one from a brain that does not tag its calls.
    """
    readings = tuple(call_ms)
    attribution, planner, narrator = _attribute_tagged(
        phases, len(readings)
    ) or _attribute_phases(len(readings), route=route, stop_reason=stop_reason)
    return TurnLatencies(
        readings,
        attribution,
        _total(readings, planner),
        _total(readings, narrator),
    )


@dataclass(frozen=True)
class TurnTokens:
    """One turn's model round trips priced in tokens, split by the same rule.

    The half `TurnLatencies` cannot supply. It settled *where* a repeat-stop
    turn's extra 30 s goes — the narrator's own round trip, and not a harder
    sample (p = 0.00088, `docs/agent-evals.md`) — and left the mechanism open,
    because "the narrator emitted three times the tokens" and "generation ran at
    a third of the rate" are the same milliseconds. Tokens say which.

    `call_in` and `call_out` are the per-call prompt and completion counts in
    call order, `None` where the round trip reported no usage. Both are read off
    the harness's call meter (`fakes.CountingProvider`), not the trace: the
    trace carries the turn's *totals*, which have the blind spot `model_ms` had.

    `narrator_in` is here beside the completion counts because prompt size is
    its own candidate mechanism — a `no_progress` turn dispatched the duplicate
    call, so its narrator reads one more tool result than a `completed` turn's
    does, and a longer prompt costs prefill before a single token is written.
    """

    call_in: tuple[int | None, ...]
    call_out: tuple[int | None, ...]
    attribution: Attribution
    planner_out: int | None
    narrator_in: int | None
    narrator_out: int | None

    @property
    def prompt_tokens(self) -> int | None:
        """The turn's prompt total — the same sum the trace record reports, from
        the other seam, so the two can be compared."""
        return _total(self.call_in, _WHOLE_TURN) if self.call_in else None

    @property
    def completion_tokens(self) -> int | None:
        """`None`, not 0, for a turn that never called the model: a zero-LLM
        fast path has no token cost to report, and 0 reads as one measured at
        nothing."""
        return _total(self.call_out, _WHOLE_TURN) if self.call_out else None

    def summary(self) -> str:
        """The token clause of the `[eval]` line, unknowns as `?` — same rule as
        the latency clause, and printed straight after it so the reader can line
        a call's milliseconds up against what it wrote."""
        return (
            f"call_in=[{_readings(self.call_in)}] "
            f"call_out=[{_readings(self.call_out)}] "
            f"planner_out={_known(self.planner_out)} "
            f"narrator_in={_known(self.narrator_in)} "
            f"narrator_out={_known(self.narrator_out)}"
        )

    def as_record(self) -> dict[str, Any]:
        """The per-sample token fields for the JSONL report, merged beside
        `TurnLatencies.as_record()`. The pairing is the point: 20 samples of
        `narrator_ms` next to `narrator_out` answer in one pass what reading
        either column alone cannot."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "call_in": list(self.call_in),
            "call_out": list(self.call_out),
            "planner_out": self.planner_out,
            "narrator_in": self.narrator_in,
            "narrator_out": self.narrator_out,
        }


def split_tokens(
    usage: Sequence[tuple[int | None, int | None]],
    *,
    route: str | None,
    stop_reason: str | None,
    phases: Sequence[str] | None = None,
) -> TurnTokens:
    """Attribute a turn's per-call token counts to the phase that spent them.

    `usage` is `(prompt_tokens, completion_tokens)` per round trip, in call
    order — `fakes.ModelCall`, which records one per `chat()` including the ones
    that raised (those report `(None, None)`: the result never came back).

    The attribution is `_attribute_phases`, unchanged and unduplicated, so a
    sample's tokens and its milliseconds always name the same call as the
    narrator's.
    """
    call_in = tuple(prompt for prompt, _ in usage)
    call_out = tuple(completion for _, completion in usage)
    attribution, planner, narrator = _attribute_tagged(
        phases, len(usage)
    ) or _attribute_phases(len(usage), route=route, stop_reason=stop_reason)
    return TurnTokens(
        call_in=call_in,
        call_out=call_out,
        attribution=attribution,
        planner_out=_total(call_out, planner),
        narrator_in=_total(call_in, narrator),
        narrator_out=_total(call_out, narrator),
    )


def generation_rate(completion_tokens: int | None, ms: int | None) -> float | None:
    """Tokens per second — the discriminator the two halves exist to compute.

    A narration that ran 3× longer at the *same* rate wrote 3× the tokens, and
    the standing candidate fix (bounding the narrator's thinking budget) is
    aimed at the right thing. The same tokens at a third of the rate is the
    server, and a token cap would not touch it. Setting that cap against an
    unexplained latency is a guess with a number on it — hence this.

    `None` whenever either side is unknown, and on a 0 ms reading: a rate off a
    clock that did not run is a division, not a measurement.
    """
    if completion_tokens is None or not ms:
        return None
    return completion_tokens * 1000 / ms
