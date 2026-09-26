"""Speech accuracy: how often what Glitch says is true, scored offline (#367).

The honesty guard reads a narration's operational claims — a capture, a check,
a move, a saved game, a setting, an engine number — and checks each against
what the turn can back (`honesty.claims`, `facts.assemble`). Live, that verdict
decides what the player hears. Here the same reading is a *measurement*: every
traced turn is re-judged from its own record and the answers are counted, per
family, as claims made and claims backed. Nothing in this module runs on the
player's turn, and nothing it returns changes what anyone hears.

A turn is re-judged from two trace fields (schema 3): `draft`, the model's own
words as the guard was handed them, and `evidence`, the `facts.TurnEvidence`
the live facts were assembled from. The facts come from the same `assemble`
the guard calls, so a scored turn is scored on exactly the facts it had.

**Scorer errors are not model errors.** A family the reading cannot score
reliably — one whose claims it cannot tell from a correct line — is reported
with its counts and left out of the accuracy (`UNSCORED`). An older record
(schema 1 or 2) has no evidence: its draft is recovered from `suppressed` or
from the commentary with the app's own reply line removed, its facts from what
the record does hold, and only the families those facts fully decide are
scored on it (`LEGACY_SCORED`). The rest are counted as unscored there too,
never guessed.

`docs/speech-accuracy.md` says what each family covers, why the unscored ones
are, and the recorded baseline.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from chessapp.facts import (
    TurnEvidence,
    analysis_numbers,
    assemble,
    destructive_succeeded,
    settings_changed_by,
)
from chessapp.game import GameSession
from chessapp.honesty import CLAIM_NAMES, Claim, VerifiedFacts, claims

# The families the reading cannot score reliably, and why. Their claims are
# counted and shown, never held against the model (docs/speech-accuracy.md).
UNSCORED: Mapping[str, str] = {}

# What an older record's facts can fully decide: the ending and draw from the
# recorded outcome and the final board, check from that board and the tool
# results, and the facts that only the turn's own tool results carry. Every
# other family needs the move history, the player's colour, the settings or a
# mid-turn board, none of which a pre-schema-3 record holds.
LEGACY_SCORED = frozenset(
    {
        "ending",
        "draw",
        "outcome",
        "check",
        "takeback",
        "restart",
        "save",
        "verbosity_change",
        "evaluation",
    }
)

# The routes a model speaks on. `control` and `resign` compose only the app's
# own lines, and a `confirmation` turn's words were never guarded, so an older
# record from one of them has no draft to recover.
_SPOKEN_ROUTES = frozenset({"brain", "fast_path", "board"})

# The app's own lines, composed after the model's words (`api._move_commentary`,
# `api._engine_lost_words`). A trailing paragraph that starts with one of these
# is the app speaking, not Glitch.
_APP_LINE_PREFIXES = ("Game over:", "My engine dropped out before it answered")


@dataclass(frozen=True)
class TurnScore:
    """One turn, re-judged: every claim its draft made, and the families this
    record cannot score (`UNSCORED`, plus the legacy gaps)."""

    claims: tuple[Claim, ...]
    unscored: frozenset[str]
    legacy: bool


def _is_turn(record: Mapping[str, Any]) -> bool:
    return record.get("kind", "turn") == "turn"


def _legacy_draft(record: Mapping[str, Any]) -> str | None:
    """The model's words in a record that predates `draft`, or None when the
    record holds none: the suppressed draft when the guard fired, else the
    commentary with the app's reply announcement removed."""
    if record.get("route") not in _SPOKEN_ROUTES or not record.get("model_calls"):
        return None
    if record.get("guarded"):
        return record.get("suppressed") or ""
    text = record.get("commentary") or ""
    head, _, tail = text.rpartition("\n\n")
    reply = (record.get("engine_reply") or {}).get("san")
    if head and (
        (reply and tail.startswith(f"{reply}.")) or tail.startswith(_APP_LINE_PREFIXES)
    ):
        return head
    return text


def _legacy_facts(record: Mapping[str, Any]) -> VerifiedFacts:
    """What an older record can back: only the facts `LEGACY_SCORED` reads,
    each read off the record exactly as `assemble` reads it off the evidence."""
    tools = [{"name": t["name"], "result": t["result"]} for t in record["tools"]]
    after = GameSession(fen=record["fen_after"])
    outcome = record.get("outcome")
    succeeded = {t["name"] for t in tools if t["result"].get("ok") is True}
    return VerifiedFacts(
        ended=outcome is not None
        or after.is_game_over()
        or destructive_succeeded(tools),
        drawn=outcome is not None and outcome.get("winner") is None,
        winner=(outcome or {}).get("winner"),
        termination=(outcome or {}).get("termination"),
        check=after.is_check()
        or any(
            t["result"].get("ok") is True and t["result"].get("check") is True
            for t in tools
        ),
        saved=bool(succeeded & {"save_game", "resume_game"}),
        settings_changed=frozenset(settings_changed_by(tools)),
        numbers=frozenset(analysis_numbers(tools)),
        undone="undo" in succeeded,
        restarted="new_game" in succeeded,
    )


def score_record(record: Mapping[str, Any]) -> TurnScore | None:
    """The turn re-judged, or None when the record holds no words of the
    model's to judge (another kind of record, a route the model does not speak
    on, a turn that never reached the guard)."""
    if not _is_turn(record):
        return None
    if record.get("evidence"):
        evidence = TurnEvidence.from_trace(record["evidence"], record["tools"])
        facts = assemble(evidence)
        return TurnScore(
            _meaningful(
                claims(record.get("draft") or "", facts),
                facts,
                reply_pending=evidence.pending_reply_fen is not None,
            ),
            frozenset(UNSCORED),
            legacy=False,
        )
    if record.get("schema", 1) >= 3:
        return None  # a current record with no evidence never reached the guard
    draft = _legacy_draft(record)
    if draft is None:
        return None
    unscored = frozenset(UNSCORED) | (frozenset(CLAIM_NAMES) - LEGACY_SCORED)
    # The ending's winner is the player's side only where the record says so.
    if "outcome" not in record:
        unscored |= {"outcome"}
    facts = _legacy_facts(record)
    return TurnScore(
        _meaningful(claims(draft, facts), facts, reply_pending=None),
        unscored,
        legacy=True,
    )


def _meaningful(
    found: tuple[Claim, ...], facts: VerifiedFacts, *, reply_pending: bool | None
) -> tuple[Claim, ...]:
    """The claims that assert something on this turn.

    Two classes pass by construction where their fact does not apply, which
    is right for a guard and wrong for a count: `outcome` defers to `ending`
    on a live board (the same sentence, one fact), and `unplayed_reply` reads
    every SAN and can only be false while a reply is owed. Counted there, each
    would add a backed claim for every "Checkmate!" or "Nf3" that the other
    class already judged. `reply_pending` is None when the record cannot say,
    and the class is then left for the unscored count.
    """
    return tuple(
        claim
        for claim in found
        if not (claim.claim == "outcome" and not facts.ended)
        and not (claim.claim == "unplayed_reply" and reply_pending is False)
    )


@dataclass
class Unbacked:
    """One unbacked claim, with enough of its turn to find it again."""

    family: str
    sentence: str
    said: str
    ts: str
    route: str
    utterance: str


@dataclass
class Tally:
    """Speech accuracy over a run of turns: per family, claims made and
    claims backed, the scored and the unscored kept apart."""

    turns: int = 0
    legacy_turns: int = 0
    made: Counter[str] = field(default_factory=Counter)
    backed: Counter[str] = field(default_factory=Counter)
    unscored_made: Counter[str] = field(default_factory=Counter)
    unscored_backed: Counter[str] = field(default_factory=Counter)
    unbacked: list[Unbacked] = field(default_factory=list)

    def observe(self, record: Mapping[str, Any]) -> None:
        """Score one trace record into the tally, if it holds words to judge."""
        score = score_record(record)
        if score is not None:
            self.add(record, score)

    def add(self, record: Mapping[str, Any], score: TurnScore) -> None:
        self.turns += 1
        self.legacy_turns += score.legacy
        for claim in score.claims:
            scored = claim.claim not in score.unscored
            made, backed = (
                (self.made, self.backed)
                if scored
                else (self.unscored_made, self.unscored_backed)
            )
            made[claim.claim] += 1
            backed[claim.claim] += claim.backed
            if scored and not claim.backed:
                self.unbacked.append(
                    Unbacked(
                        claim.claim,
                        claim.sentence,
                        claim.said,
                        str(record.get("ts", "")),
                        str(record.get("route", "")),
                        str(record.get("utterance", "")),
                    )
                )

    @property
    def claims_made(self) -> int:
        return sum(self.made.values())

    @property
    def claims_backed(self) -> int:
        return sum(self.backed.values())

    @property
    def accuracy(self) -> float | None:
        """Backed over made across the scored families; None with no claims."""
        made = self.claims_made
        return self.claims_backed / made if made else None

    def as_dict(self) -> dict[str, Any]:
        """The run's numbers, in the shape the eval reports and the frontier
        history carry."""
        return {
            "turns": self.turns,
            "legacy_turns": self.legacy_turns,
            "made": self.claims_made,
            "backed": self.claims_backed,
            "accuracy": None if self.accuracy is None else round(self.accuracy, 4),
            "families": {
                name: {"made": self.made[name], "backed": self.backed[name]}
                for name in CLAIM_NAMES
                if self.made[name]
            },
            "unscored": {
                name: {
                    "made": self.unscored_made[name],
                    "backed": self.unscored_backed[name],
                }
                for name in CLAIM_NAMES
                if self.unscored_made[name]
            },
        }


def tally(records: Iterable[Mapping[str, Any]]) -> Tally:
    """Speech accuracy over every scoreable turn in `records`."""
    result = Tally()
    for record in records:
        result.observe(record)
    return result


def merge(summaries: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Several `Tally.as_dict` summaries added into one, for a report that
    ran as several parts (a frontier scenario's samples, a run's scenarios)."""
    total = Tally()
    for summary in summaries:
        if not summary:
            continue
        total.turns += summary["turns"]
        total.legacy_turns += summary["legacy_turns"]
        for name, counts in summary["families"].items():
            total.made[name] += counts["made"]
            total.backed[name] += counts["backed"]
        for name, counts in summary["unscored"].items():
            total.unscored_made[name] += counts["made"]
            total.unscored_backed[name] += counts["backed"]
    return total.as_dict()
