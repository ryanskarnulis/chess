"""The eval gate's arithmetic: how many samples, and what the count means.

Sprint 5, slice 3 (audit item 23). The live suite in `test_agent_evals.py`
cannot be unit-tested — every assertion there needs a 12B on the GPU — so its
*verdict* never was. This module is that verdict, extracted: pure functions over
counts, stop reasons and failure strings, with no I/O, no model and no import of
anything that touches one. `test_evalstats.py` is its spec and runs on every
commit, which is the only way the gate itself gets regression-tested.

It exists because `assert result.rate >= 0.8` over 5 samples is a coin flip at
the floor, and the record proves it: `long_capture[poisoned]` measured 3/5 then
4/5 on one build (red once, green once), `main` gave 5/5 then 3/5, and
`play_as_black` gave 5/5, 2/5, 0/5, 5/5 across four same-build runs
(`docs/agent-evals.md`). A gate that flaps gets re-run until green, which is how
a baseline becomes a hand-assembled composite.

Three separate defects hid inside that one number, and each gets a function
here:

- **Infrastructure was scored as behavior.** llama-server crash-restarts every
  3–8 minutes of sustained generation, and since audit 20 the brain *catches*
  `ProviderError` and answers 200 with `stop_reason="provider_error"` — so a
  crash stopped reading as a 502 and started reading as a silent behavioral
  miss. `classify` separates the two so the harness can retry instead of score
  — and splits the death itself, because the brain now names the kind: a dead
  socket is worth asking again, a request the server *rejected* (an HTTP 400 on
  an overrun context) answers the same way every time and is
  `Outcome.PROVIDER_REJECTED` on the first one.
- **Vacuous passes counted as passes.** A budget stop reaches no narrator, so a
  purely negative commentary check passes for never having been tested (three of
  four `hints_off_no_advice` passes in one recorded run). `Outcome.INCONCLUSIVE`
  plus `VacuousRun` names that.
- **More repetitions against `rate >= floor` measures worse, not better.** At a
  true rate of exactly 0.80, fixed-5 goes green 74% of the time and fixed-20
  goes green **63%**: four times the GPU for a worse gate. `decide` replaces the
  comparison with a one-sided bound and block-sequential escalation, which at
  n=5 costs a healthy scenario exactly what it costs today.

Nothing here knows about pytest, HTTP or the app. That is deliberate: the
harness is the only thing that may touch the model, and this is the only thing
that may decide what its samples mean.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# The **one-sided** 95% normal quantile. The FAIL test asks a single-tailed
# question — is the interval's *upper* end below the floor? — so 1.645 is what
# makes it a 95% test; 1.96 is the two-tailed point and would overstate the
# evidence by a whole tail. Every operating characteristic recorded in
# docs/agent-evals.md is this z's, so it is pinned by value in the tests.
Z_ONE_SIDED_95 = 1.645

# A block stops being "the same run" when its rate differs from another block's
# by this much. Tuned against the record rather than chosen: 0.6 is the spread
# `play_as_black` showed between its extremes, and it stays quiet on
# `long_capture`'s 5/5-vs-3/5 (ordinary spread at a true rate near the floor). A
# threshold that flagged the second would flag everything.
UNSTABLE_SPREAD = 0.6

# The loop's own stop reasons, as `llama_brain` produces them. Named here rather
# than imported: this module must stay importable with nothing installed, and a
# string constant is not worth a dependency on the package under test.
STOP_PROVIDER_ERROR = "provider_error"

# The provider-failure kinds a retry cannot fix, mirroring the non-transient
# half of `chessapp.provider.ProviderFailure` (carried to here on
# `AgentResponse.provider_failure` and recorded in the trace). Literals for the
# same reason as above; `test_evalstats` pins the two vocabularies against each
# other so the duplication cannot drift.
#
# Anything *not* listed — including no kind at all, which is what a non-200 and
# every pre-existing caller carry — is retried. Unknown means retry on purpose:
# spending a few samples on a deterministic failure is cheaper than aborting a
# whole suite on a server that was only restarting.
DETERMINISTIC_FAILURES = frozenset(
    {"rejected", "malformed_response", "bad_tool_arguments"}
)

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
# model only through `Brain.narrate` — one narrator call, no planner at all. As
# with `DETERMINISTIC_FAILURES` these are literals rather than imports (this
# module must stay importable with nothing installed), and
# `test_evalstats.test_the_route_vocabularies_agree` is what pays for the copy.
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
    """What one sample was, before it is allowed to count as anything."""

    PASS = "PASS"
    FAIL = "FAIL"
    # The provider died. Retried, never scored: a crash cadence is not a rate.
    INFRA = "INFRA"
    # The provider failed in a way a retry cannot fix — an HTTP 400 (a context
    # overrun on a long transcript), a body this app can no longer parse. Like
    # HARNESS it is never retried and never scored, and unlike HARNESS it
    # arrives with no exception to re-raise: the request was answered, just not
    # with a completion. The harness reports it and fails the item on the
    # *first* one, instead of spending the infra budget re-asking a question
    # already answered.
    PROVIDER_REJECTED = "PROVIDER_REJECTED"
    # The turn never reached the thing the check was about (a budget stop under
    # a commentary-dependent check). Neither a pass nor a miss — an untested
    # sample, which today's harness silently scores as a pass.
    INCONCLUSIVE = "INCONCLUSIVE"
    # A bug in this repo (a KeyError in a check). Never retried, never scored,
    # fails the item immediately — a retry cannot fix a typo.
    HARNESS = "HARNESS"


class Decision(StrEnum):
    """What a scenario's accumulated count says about the floor."""

    # The point estimate reaches the floor. Green.
    ABOVE_FLOOR = "ABOVE_FLOOR"
    # The one-sided upper bound is *below* the floor: evidence of regression,
    # not a bad sample. Red.
    BELOW_FLOOR = "BELOW_FLOOR"
    # The interval straddles the floor and the point estimate is under it. Take
    # another block; at the cap this is green, flagged.
    UNDECIDED = "UNDECIDED"


class Stability(StrEnum):
    """Whether the blocks looked like samples of one thing."""

    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"


class Attribution(StrEnum):
    """How much of a turn's per-call latencies could be pinned to a phase."""

    # Planner and narrator are both known: the readings split at the phase
    # boundary the route and stop reason put there.
    SPLIT = "SPLIT"
    # A budget stop: no narrator ran, so every reading is the planner's and
    # there is no narrator time — not an unmeasured one, none.
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


class VacuousRun(AssertionError):
    """The turn never reached what the check was about, so the check tested
    nothing.

    An `AssertionError` subclass on purpose: `classify` reads it as
    INCONCLUSIVE, but a caller that skips `classify` gets a *failed* sample
    rather than a silent pass. Failing safe is the whole point — the bug this
    names is a check that passed without running.
    """


def wilson_interval(
    passed: int, runs: int, z: float = Z_ONE_SIDED_95
) -> tuple[float, float]:
    """The Wilson score interval for `passed`/`runs`, clamped to [0, 1].

    Wilson rather than the normal approximation because the counts here are tiny
    and the rates are near 1, which is exactly where `p̂ ± z·√(p̂q̂/n)` breaks
    (it puts the upper bound above 1 and the lower bound at 0 for 5/5).

    `runs == 0` is the whole unit interval: a scenario every sample of which was
    an infra death knows nothing, and the interval has to say so rather than
    divide by zero.
    """
    if runs <= 0:
        return (0.0, 1.0)
    point = passed / runs
    z2 = z * z
    denominator = 1 + z2 / runs
    centre = (point + z2 / (2 * runs)) / denominator
    half = (z / denominator) * math.sqrt(
        point * (1 - point) / runs + z2 / (4 * runs * runs)
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def decide(
    passed: int, runs: int, floor: float, *, z: float = Z_ONE_SIDED_95
) -> Decision:
    """Is this count below the floor, at it, or not yet resolved?

    The asymmetry is deliberate and is what makes the gate worth trusting:

    - **red** takes evidence — the upper bound has to be *below* the floor, so a
      bad sample is not a regression,
    - **green** takes only the point estimate, because demanding the lower bound
      clear the floor would call every 5-sample scenario a regression (5/5's
      lower bound is 0.65),
    - everything else is UNDECIDED, which the sampler answers with more samples.

    At n=5 and floor 0.8 this is: 0–2 red, 3 escalates, 4–5 green. A healthy
    suite therefore costs exactly what it costs today.
    """
    upper = wilson_interval(passed, runs, z)[1]
    if upper < floor:
        return Decision.BELOW_FLOOR
    if runs > 0 and passed / runs >= floor:
        return Decision.ABOVE_FLOOR
    return Decision.UNDECIDED


def classify(
    *,
    status_code: int,
    stop_reason: str | None,
    error: BaseException | None,
    provider_failure: str | None = None,
) -> Outcome:
    """What kind of sample this was. Precedence is the point of the function.

    A harness bug outranks everything (a retry cannot fix a typo, and the infra
    budget must not be spent on one). Infrastructure outranks a failed
    assertion, because when the provider died the check failed *because* it
    died — scoring that as behavior is how a crash cadence gets written down as
    a pass rate. Only then is a failed assertion a real miss.

    A provider death then splits on `provider_failure` — the kind the brain
    named (`AgentResponse.provider_failure`, read off the trace). Retrying is
    only worth anything against a server that might answer differently, so a
    kind in `DETERMINISTIC_FAILURES` is PROVIDER_REJECTED and everything else,
    the unnamed included, stays INFRA.
    """
    if error is not None and not isinstance(error, AssertionError):
        return Outcome.HARNESS
    if status_code != 200 or stop_reason == STOP_PROVIDER_ERROR:
        if provider_failure in DETERMINISTIC_FAILURES:
            return Outcome.PROVIDER_REJECTED
        return Outcome.INFRA
    if isinstance(error, VacuousRun):
        return Outcome.INCONCLUSIVE
    if error is not None:
        return Outcome.FAIL
    return Outcome.PASS


# SAN and bare squares first, then any remaining number: "Bxe6" must collapse as
# one token, not to "Bx<square>" and certainly not to "Bxe<n>".
_SAN = re.compile(
    r"\b(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b"
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WHITESPACE = re.compile(r"\s+")


def failure_signature(message: str) -> str:
    """One failure message, normalised so identical *modes* collapse.

    Five failures naming five different moves are one finding; five different
    messages are five. Today the harness prints both as "2/5" and the reader has
    to eyeball the ✗ lines. Normalised rather than hashed, because the signature
    goes in the report and the printed summary and has to stay recognisable.
    """
    text = _SAN.sub("<san>", message)
    text = _NUMBER.sub("<n>", text)
    return _WHITESPACE.sub(" ", text).strip()


def block_stability(blocks: Sequence[tuple[int, int]]) -> Stability:
    """Did the blocks look like samples of one thing?

    Twenty consecutive samples of one utterance is a cluster of size 1, so the
    Wilson width understates the truth — `play_as_black` measured 0/5 only
    mid-suite and 5/5 in isolation. This slice does not fix that (interleaving
    blocks across scenarios needs a session-scoped sampler); it makes the
    confound *measurable*.

    **Two honest limits on how much this flag can see.** It only detects drift
    *within* one scenario's sampling, and it can only speak at all where
    escalation bought a second block — a count that decides on its first block
    has one rate and no spread. So the shape that motivated it (5/5 in one whole
    run, 0/5 in another) is precisely the shape it cannot catch: a mid-suite 0/5
    goes red at block 1 and is reported STABLE, correctly, because within that
    run nothing varied. Comparing *across* runs is what finds that one, which is
    what the report's per-scenario lines are for and why the campaign measures a
    scenario both isolated and mid-suite.

    Blocks with no samples (every one retried for infra) have no rate and are
    skipped, rather than counted as 0/5 instabilities that never happened.
    """
    rates = [passed / runs for passed, runs in blocks if runs > 0]
    if len(rates) < 2:
        return Stability.STABLE
    return (
        Stability.UNSTABLE
        if max(rates) - min(rates) >= UNSTABLE_SPREAD
        else Stability.STABLE
    )


@dataclass(frozen=True)
class RateResult:
    """What sampling one scenario measured, and what it means.

    Every derived field — `interval`, `decision`, `stability`, `failure_modes`,
    `deterministic_suspect` — is built by `from_blocks` from the counts, never
    passed in. Same rule `turn_record` follows for `model_ms`: a record whose
    total disagrees with its parts is worse than no record.
    """

    passed: int
    runs: int
    blocks: tuple[tuple[int, int], ...]
    failures: tuple[str, ...]
    # Signature → count, on *every* scenario rather than only the red ones: a
    # scenario at 4/5 whose one failure is the same mode every time is a
    # different animal from one at 4/5 that fails differently each way.
    failure_modes: Mapping[str, int] = field(default_factory=dict)
    # Samples thrown away and re-taken because the provider died. Printed even
    # at 1 — a rate taken over a crashing server is worth less than the same
    # rate taken clean, and the old baseline's re-run procedure hid exactly this.
    infra: int = 0
    inconclusive: int = 0
    interval: tuple[float, float] = (0.0, 1.0)
    decision: Decision = Decision.UNDECIDED
    stability: Stability = Stability.STABLE
    # 0 passes and exactly one failure signature: not sampling noise but a
    # broken build, a changed tool offer, or a deterministic `provider_error`.
    deterministic_suspect: bool = False

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
        passed = sum(block_passed for block_passed, _ in blocks)
        runs = sum(block_runs for _, block_runs in blocks)
        modes = Counter(failure_signature(failure) for failure in failures)
        return cls(
            passed=passed,
            runs=runs,
            blocks=tuple(tuple(block) for block in blocks),  # type: ignore[misc]
            failures=tuple(failures),
            failure_modes=dict(modes),
            infra=infra,
            inconclusive=inconclusive,
            interval=wilson_interval(passed, runs, z),
            decision=decide(passed, runs, floor, z=z),
            stability=block_stability(blocks),
            deterministic_suspect=passed == 0 and len(modes) == 1,
        )

    @property
    def rate(self) -> float:
        """0.0 for an unsampled scenario — the harness fails that item on the
        infra abort, and a report line still has to serialise."""
        return self.passed / self.runs if self.runs else 0.0

    def summary(self) -> str:
        """The one-line `[eval]` verdict a human reads off the terminal."""
        low, high = self.interval
        blocks = "+".join(f"{p}/{r}" for p, r in self.blocks) or "none"
        line = (
            f"{self.passed}/{self.runs} ({self.rate:.0%}) "
            f"ci=[{low:.2f},{high:.2f}] {self.decision} {self.stability} "
            f"blocks={blocks} infra={self.infra} inconclusive={self.inconclusive}"
        )
        if self.deterministic_suspect:
            line += " DETERMINISTIC_SUSPECT"
        return line


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
    return Attribution.SPLIT, slice(0, calls - 1), slice(calls - 1, calls)


def _total(readings: Sequence[int | None], phase: slice | None) -> int | None:
    """One phase's readings summed, or `None` for either honest unknown: the
    phase boundary was not knowable, or one of the phase's own calls reported no
    reading at all (a round trip that raised has no usage). Summing the rest
    would report a *smaller* number as though it were the whole."""
    if phase is None:
        return None
    values = readings[phase]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _readings(values: Sequence[int | None]) -> str:
    """A per-call array for the `[eval]` line, unknowns as `?`."""
    return ",".join("?" if value is None else str(value) for value in values)


def _known(value: int | None) -> str:
    return "?" if value is None else str(value)


def split_latencies(
    call_ms: Sequence[int], *, route: str | None, stop_reason: str | None
) -> TurnLatencies:
    """Attribute a turn's per-call readings to the phase that spent them.

    `model_ms` is a per-turn sum, and the question it cannot answer is the one
    Sprint 5 is asking: a `no_progress` turn was measured narrating 2–3× slower
    than a `completed` one at the *same* model-call count (40.5 s and ≈26.7 s
    medians against 9.9–11.5 s — `TODO.md`). Whether that is the narrator's own
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
    """
    readings = tuple(call_ms)
    attribution, planner, narrator = _attribute_phases(
        len(readings), route=route, stop_reason=stop_reason
    )
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
    attribution, planner, narrator = _attribute_phases(
        len(usage), route=route, stop_reason=stop_reason
    )
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


def scenario_record(
    scenario: str, result: RateResult, floor: float, **meta: Any
) -> dict[str, Any]:
    """One scenario's verdict as a JSON-serialisable record.

    Written as it finishes, one line per scenario, so a baseline stops being
    hand-transcribed from terminal scrollback and a Ctrl-C mid-suite still
    leaves every finished scenario on disk (the tracer's own precedent).
    `**meta` carries whatever the call site knows and this module must not —
    per-sample latencies, the seam, whether the item is release-blocking.
    """
    return {
        "scenario": scenario,
        "floor": floor,
        "passed": result.passed,
        "runs": result.runs,
        "rate": result.rate,
        "blocks": [list(block) for block in result.blocks],
        "interval": list(result.interval),
        "decision": str(result.decision),
        "stability": str(result.stability),
        "infra": result.infra,
        "inconclusive": result.inconclusive,
        "deterministic_suspect": result.deterministic_suspect,
        "failure_modes": dict(result.failure_modes),
        "failures": list(result.failures),
        **meta,
    }
