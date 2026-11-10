"""The confirmation reader, live (#544): does it read a reply the way it was meant?

`LlamaBrain.read_answer` turns a free-text reply to an armed confirmation into
`confirm`, `cancel` or `unrelated`, and a `confirm` can end a game. The `answer`
phase used to inherit the profile's temperature 1.0; each profile now sets
`[phases.answer] temperature = 0`. This measures the difference on a small
standing set of paraphrases of yes, no and neither, against the model the app
ships on.

Opt-in like the other live evals, and never in CI (it needs the GPU):

    cd backend
    CHESSAPP_ANSWER_EVALS=1 .venv/bin/pytest tests/test_answer_reader_live.py -v -s

Two arms run interleaved on paired sampling seeds: `shipped` (the profile as it
is) and `inherited` (the same profile with the answer phase's temperature
removed, so it samples at the profile's own, as before #544). Knobs:
`CHESSAPP_ANSWER_RUNS` (seeds per paraphrase, default 10), `CHESSAPP_EVAL_SEED`
(first seed, default 7), and `LLAMACPP_BASE_URL` / `LLAMACPP_MODEL`.

Every paraphrase is one the literal reader (`fastparse.parse_confirmation`)
does not settle, so it reaches the model; the offline tests below keep that
true and keep the set balanced.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, replace
from types import MappingProxyType

import pytest

from chessapp.app import DEFAULT_MODEL
from chessapp.brain import CALL_OK, CANCEL, CONFIRM, UNRELATED
from chessapp.fastparse import parse_confirmation
from chessapp.game import GameSession
from chessapp.llama_brain import create_llama_brain
from chessapp.profiles import ANSWER, ModelProfile, load_profile
from chessapp.provider import LlamaCppProvider
from chessapp.tools import (
    CONFIRM_QUESTIONS,
    ToolContext,
    brain_tool_definitions,
    build_registry,
)


@dataclass(frozen=True)
class Reply:
    op: str  # the armed op the question is about (a `CONFIRM_QUESTIONS` key)
    text: str
    verdict: str  # what a careful reader should reach


# Seven each of yes, no and neither, spread over the five questions the gate asks.
REPLIES = (
    Reply("resign", "just do it", CONFIRM),
    Reply("new_game", "yeah, go for it, start fresh", CONFIRM),
    Reply("claim_draw", "sure, take the draw", CONFIRM),
    Reply("resume_game", "I'm sure, load it", CONFIRM),
    Reply("save_game", "overwrite it, that's fine", CONFIRM),
    Reply("resign", "get on with it already", CONFIRM),
    Reply("new_game", "absolutely, wipe it", CONFIRM),
    Reply("resign", "actually, forget it", CANCEL),
    Reply("new_game", "hold on, I want to keep playing", CANCEL),
    Reply("claim_draw", "not yet, I think I can win this", CANCEL),
    Reply("resume_game", "no, leave the current game alone", CANCEL),
    Reply("save_game", "don't overwrite it", CANCEL),
    Reply("resign", "I'd rather fight on", CANCEL),
    Reply("new_game", "wait, don't, I changed my mind", CANCEL),
    Reply("resign", "show me the position instead", UNRELATED),
    Reply("new_game", "undo my last move instead", UNRELATED),
    Reply("claim_draw", "what's the evaluation?", UNRELATED),
    Reply("resume_game", "which saves do I have?", UNRELATED),
    Reply("save_game", "what was the name again?", UNRELATED),
    Reply("resign", "make the engine easier", UNRELATED),
    Reply("new_game", "how many moves have we played?", UNRELATED),
)

# The two ways to be wrong are not alike: reading a no (or a different request)
# as a yes can end a game, and reading a yes as anything else costs a repeat.
RISKY = "false_confirm"

# Agreement the shipped arm must keep, and the false confirms it may make.
AGREEMENT_FLOOR = 0.9
FALSE_CONFIRMS_ALLOWED = 0


# --- offline: what the set has to be to mean anything --------------------------


def test_the_set_is_balanced_across_yes_no_and_neither():
    assert Counter(r.verdict for r in REPLIES) == {CONFIRM: 7, CANCEL: 7, UNRELATED: 7}
    assert len({(r.op, r.text) for r in REPLIES}) == len(REPLIES)


def test_every_reply_reaches_the_model():
    # A reply the literal reader settles never gets to the model, so measuring
    # the model on it would measure nothing.
    assert [r.text for r in REPLIES if parse_confirmation(r.text) is not None] == []
    assert all(r.op in CONFIRM_QUESTIONS for r in REPLIES)


# --- live ------------------------------------------------------------------------

LIVE = pytest.mark.skipif(
    os.environ.get("CHESSAPP_ANSWER_EVALS") != "1",
    reason="live confirmation-reader eval: set CHESSAPP_ANSWER_EVALS=1 (needs the GPU)",
)
BASE_URL = os.environ.get("LLAMACPP_BASE_URL", "http://127.0.0.1:8200/v1")
MODEL = os.environ.get("LLAMACPP_MODEL", DEFAULT_MODEL)
RUNS = int(os.environ.get("CHESSAPP_ANSWER_RUNS", "10"))
SEED = int(os.environ.get("CHESSAPP_EVAL_SEED") or 7)


def _inherited(profile: ModelProfile) -> ModelProfile:
    """The profile as it was before #544: the answer phase without its own
    temperature, so it samples at the profile's."""
    phases = dict(profile.phases)
    phases[ANSWER] = replace(phases[ANSWER], temperature=None)
    return replace(profile, phases=MappingProxyType(phases))


def _verdict(profile: ModelProfile, reply: Reply, seed: int) -> str:
    ctx = ToolContext(session=GameSession())
    registry = build_registry(ctx)
    brain = create_llama_brain(
        base_url=BASE_URL,
        model=MODEL,
        dispatcher=registry,
        tool_definitions=lambda: brain_tool_definitions(registry, ctx),
        provider=LlamaCppProvider(BASE_URL, MODEL, profile=profile, seed=seed),
    )
    answer = brain.read_answer(CONFIRM_QUESTIONS[reply.op], reply.text)
    # A call that never produced a word is no reading: fail loudly rather than
    # score it as a verdict (`unrelated` by omission would flatter an arm).
    assert answer.status == CALL_OK, f"{reply.text!r}: {answer.status} {answer.failure}"
    return answer.verdict


@dataclass
class Tally:
    agreed: int = 0
    total: int = 0
    false_confirms: int = 0
    wrong: Counter[tuple[str, str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.wrong = Counter()

    @property
    def agreement(self) -> float:
        return self.agreed / self.total

    def add(self, reply: Reply, got: str) -> None:
        self.total += 1
        if got == reply.verdict:
            self.agreed += 1
            return
        self.wrong[(reply.text, reply.verdict, got)] += 1
        if got == CONFIRM:
            self.false_confirms += 1


def _report(name: str, tally: Tally) -> None:
    print(
        f"[answer] arm={name} model={MODEL} agree={tally.agreed}/{tally.total} "
        f"({tally.agreement:.1%}) false_confirms={tally.false_confirms}"
    )
    for (text, want, got), count in sorted(tally.wrong.items()):
        print(f"[answer]   {name}: {text!r} wanted {want}, read {got} x{count}")


@LIVE
def test_the_shipped_reader_agrees_with_what_the_player_meant():
    shipped = load_profile(MODEL)
    arms = {"shipped": shipped, "inherited": _inherited(shipped)}
    tallies = {name: Tally() for name in arms}
    for run in range(RUNS):
        for reply in REPLIES:
            # Paired: both arms see the same seed on the same reply, and the
            # arms alternate within it so server drift hits them evenly.
            for name, profile in arms.items():
                tallies[name].add(reply, _verdict(profile, reply, SEED + run))
    print(f"[answer] {len(REPLIES)} replies x {RUNS} seeds, first seed {SEED}")
    for name, tally in tallies.items():
        _report(name, tally)

    now = tallies["shipped"]
    assert now.false_confirms <= FALSE_CONFIRMS_ALLOWED, "a no or an aside read as yes"
    assert now.agreement >= AGREEMENT_FLOOR
