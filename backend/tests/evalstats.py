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
  miss. `classify` separates the two so the harness can retry instead of score.
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


class Outcome(StrEnum):
    """What one sample was, before it is allowed to count as anything."""

    PASS = "PASS"
    FAIL = "FAIL"
    # The provider died. Retried, never scored: a crash cadence is not a rate.
    INFRA = "INFRA"
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
    *, status_code: int, stop_reason: str | None, error: BaseException | None
) -> Outcome:
    """What kind of sample this was. Precedence is the point of the function.

    A harness bug outranks everything (a retry cannot fix a typo, and the infra
    budget must not be spent on one). Infrastructure outranks a failed
    assertion, because when the provider died the check failed *because* it
    died — scoring that as behavior is how a crash cadence gets written down as
    a pass rate. Only then is a failed assertion a real miss.
    """
    if error is not None and not isinstance(error, AssertionError):
        return Outcome.HARNESS
    if status_code != 200 or stop_reason == STOP_PROVIDER_ERROR:
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
