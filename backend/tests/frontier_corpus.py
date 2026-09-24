"""The frontier corpus (#318): scenarios built to be hard, measured not gated.

Each scenario says why it is hard, which tier it sits in, and how it is
graded: named checkpoints, each a predicate over the board, the settings, the
saves and the tool results (`frontier.Checkpoint`) — never over wording.
`dev` variants are for iterating on prompts; `heldout` variants are never
tuned against, and an improvement is believed only when they move too.

Tiers, by what today's model is expected to manage:

1. **Stretch** — one utterance composing several intents.
2. **Multi-turn** — a thread whose later turns lean on earlier ones: a
   question asked, a suggestion made, a save named, a starting setting.
3. **Frontier** — long sessions and long games, expected near zero.

Every checkpoint must be unambiguous about what the player asked for: the
first live run graded "how do things stand now?" as a judgment question, the
model fairly answered it with a description, and the miss was the
scenario's (`docs/agent-frontier.md`).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess.pgn

from frontier import VERDICT_TOOLS, Checkpoint, Episode, Say, Scenario, Turn, Variant
from test_agent_evals import _TIER_STRENGTH, EvalApp

_FIXTURES = Path(__file__).parent

# --- setups --------------------------------------------------------------------------


def _save_dir(app: EvalApp) -> None:
    """A save directory nothing has written to: a save a sample needs must be
    that sample's own (the gate's `_fresh_save_dir` rule)."""
    app.ctx.save_dir = Path(tempfile.mkdtemp(prefix="frontier-"))


def after(*sans: str) -> Callable[[EvalApp], None]:
    """Moves placed through the session (the gate's way of setting up a
    position), with a fresh save directory."""

    def setup(app: EvalApp) -> None:
        _save_dir(app)
        for san in sans:
            assert app.ctx.session.submit_move(san).legal, san

    return setup


def replayed(fixture: str) -> Callable[[EvalApp], None]:
    """A committed fixture game replayed through the legality gate, so undo,
    review and save meet a real move stack (the 84-ply fixture's rule)."""

    def setup(app: EvalApp) -> None:
        _save_dir(app)
        with (_FIXTURES / fixture).open() as source:
            game = chess.pgn.read_game(source)
        assert game is not None and not game.errors, fixture
        for move in game.mainline_moves():
            assert app.ctx.session.submit_move(move.uci()).legal
        assert not app.ctx.session.is_game_over()

    return setup


def at_tier(tier: str) -> Callable[[EvalApp], None]:
    """A fresh board with the engine set to `tier`, the way the difficulty
    endpoint sets it: room to move in both directions from the start."""

    def setup(app: EvalApp) -> None:
        _save_dir(app)
        app.ctx.settings.tier = tier
        app.ctx.settings.skill_level = None
        app.ctx.settings.elo = None
        if app.ctx.engine is not None:
            app.ctx.engine.set_tier(tier)

    return setup


FRESH = after()
AFTER_E4_E5 = after("e4", "e5")
AFTER_E4_E5_NF3_NC6 = after("e4", "e5", "Nf3", "Nc6")
LATE_84 = replayed("late_game_84_plies.pgn")
LATE_150 = replayed("late_game_150_plies.pgn")

# Kept for the harness's own tests, which script a sample on this position.
after_e4_e5 = AFTER_E4_E5

# --- what the checkpoints read --------------------------------------------------------


def strength(settings: dict[str, Any]) -> float:
    """One number for "how strong is the engine set", across the three ways
    of setting it (the gate's `_difficulty_strength`)."""
    if settings.get("tier") is not None:
        return float(_TIER_STRENGTH[settings["tier"]])
    if settings.get("elo") is not None:
        return float(settings["elo"])
    if settings.get("skill_level") is not None:
        return 800.0 + settings["skill_level"] * 110.0
    return float(_TIER_STRENGTH["casual"])


def difficulty(settings: dict[str, Any]) -> tuple[Any, ...]:
    return (settings.get("tier"), settings.get("elo"), settings.get("skill_level"))


def judged(turn: Turn) -> bool:
    return any(turn.ran(name) for name in VERDICT_TOOLS)


def suggested(turn: Turn, index: int = 0) -> str | None:
    """The `index`th candidate a `get_best_moves` call returned this turn."""
    result = turn.result("get_best_moves")
    moves = (result or {}).get("moves") or []
    return moves[index]["san"] if len(moves) > index else None


def verdict_after_the_move(turn: Turn) -> bool:
    """A verdict tool succeeded after the last successful move, in order."""
    names = turn.succeeded()
    if "make_move" not in names:
        return False
    last = len(names) - 1 - names[::-1].index("make_move")
    return any(name in VERDICT_TOOLS for name in names[last + 1 :])


def consulted_before_moving(turn: Turn) -> bool:
    names = turn.succeeded()
    return (
        "get_best_moves" in names
        and "make_move" in names
        and names.index("get_best_moves") < names.index("make_move")
    )


def completed(index: int) -> Checkpoint:
    return Checkpoint(
        f"completed_{index}", lambda e: e.turn(index).stop_reason == "completed"
    )


def still(index: int) -> Checkpoint:
    """A read-only ask moved nothing."""
    return Checkpoint(f"moved_nothing_{index}", lambda e: not e.turn(index).moved)


def worst_player_move(turn: Turn, color: str = "white") -> tuple[int, str] | None:
    """The ply index and SAN of the player's worst move in a `review_game`
    result: the highest centipawn loss among that side's critical moves."""
    result = turn.result("review_game")
    mine = [m for m in (result or {}).get("critical", []) if m["color"] == color]
    if not mine:
        return None
    worst = max(mine, key=lambda m: m["cp_loss"] or 0)
    ply = (worst["move_number"] - 1) * 2 + (0 if color == "white" else 1)
    return ply, worst["san"]


def _back_before_worst(e: Episode) -> bool:
    worst = worst_player_move(e.turn(1))
    if worst is None:
        return False
    ply, san = worst
    start = e.start["history"]
    return start[ply] == san and e.history(after_turn=2) == start[:ply]


def _played_best_there(e: Episode) -> bool:
    worst = worst_player_move(e.turn(1))
    best = suggested(e.turn(3))
    if worst is None or best is None:
        return False
    history = e.history(after_turn=4)
    return len(history) > worst[0] and history[worst[0]] == best


# --- tier 1: several intents in one utterance -----------------------------------------

UNDO_REPLACE_AND_JUDGE = Scenario(
    name="undo_replace_and_judge",
    tier=1,
    why=(
        "Three intents in one breath — take back, replace, evaluate — in that "
        "order. The gate's undo_and_replace pins the first two; the verdict "
        "has to come off the new position, after the move."
    ),
    dev=(
        Variant(
            "take_back_d4_better",
            (Say("take that back, play d4 instead, and tell me if it's any better"),),
            AFTER_E4_E5,
        ),
        Variant(
            "undo_go_d4_how_looks",
            (Say("undo my move, go d4, then tell me how the position looks for me"),),
            AFTER_E4_E5,
        ),
    ),
    heldout=(
        Variant(
            "scratch_queen_pawn_read",
            (
                Say(
                    "scratch that move — push the queen pawn two instead and "
                    "tell me whether I'm better off"
                ),
            ),
            AFTER_E4_E5,
        ),
    ),
    checkpoints=(
        Checkpoint("took_back", lambda e: bool(e.turn(1).ran("undo"))),
        Checkpoint("played_d4", lambda e: e.history()[:1] == ["d4"]),
        Checkpoint("one_exchange", lambda e: len(e.history()) == 2),
        Checkpoint(
            "judged_after_the_move", lambda e: verdict_after_the_move(e.turn(1))
        ),
        completed(1),
    ),
)

SETTINGS_MOVE_AND_VERDICT = Scenario(
    name="settings_move_and_verdict",
    tier=1,
    why=(
        "Two settings, a move and a judgment in one utterance: each lever is "
        "a different tool, and the verdict must come after the move."
    ),
    dev=(
        Variant(
            "voice_easier_king_pawn",
            (
                Say(
                    "turn spoken replies on, make the engine a little easier, "
                    "then push my king pawn two squares and tell me how it looks"
                ),
            ),
            FRESH,
        ),
    ),
    heldout=(
        Variant(
            "talk_out_loud_softer_e_pawn",
            (
                Say(
                    "start talking to me out loud, tone the difficulty down a "
                    "notch, open with the e-pawn up two, and say whether that's "
                    "good for me"
                ),
            ),
            FRESH,
        ),
    ),
    checkpoints=(
        Checkpoint("voice_on", lambda e: e.settings()["voice_output"] is True),
        Checkpoint(
            "easier", lambda e: strength(e.settings()) < strength(e.settings(0))
        ),
        Checkpoint("played_e4", lambda e: e.history()[:1] == ["e4"]),
        Checkpoint(
            "judged_after_the_move", lambda e: verdict_after_the_move(e.turn(1))
        ),
        completed(1),
    ),
)

TOP_MOVE_PLAY_AND_SAVE = Scenario(
    name="top_move_play_and_save",
    tier=1,
    why=(
        "Read, act on the read, then persist: the move must be the engine's "
        "first candidate, and the save must hold the game with it in."
    ),
    dev=(
        Variant(
            "stockfish_top_then_checkpoint",
            (
                Say(
                    "ask Stockfish for its top move, play it for me, and then "
                    "save the game as checkpoint"
                ),
            ),
            AFTER_E4_E5,
        ),
    ),
    heldout=(
        Variant(
            "engine_pick_then_store",
            (
                Say(
                    "get the engine's best suggestion, make that move, and "
                    "store this game under the name checkpoint"
                ),
            ),
            AFTER_E4_E5,
        ),
    ),
    checkpoints=(
        Checkpoint("consulted_first", lambda e: consulted_before_moving(e.turn(1))),
        Checkpoint(
            "played_the_top_move",
            lambda e: (
                suggested(e.turn(1)) is not None
                and e.history()[2:3] == [suggested(e.turn(1))]
            ),
        ),
        Checkpoint(
            "saved_with_the_move",
            lambda e: (e.saved("checkpoint") or [])[:3] == e.history()[:3],
        ),
        completed(1),
    ),
)

NOISY_TAKEBACK_AND_REPLACE = Scenario(
    name="noisy_takeback_and_replace",
    tier=1,
    why=(
        "Speech-to-text noise over a composition: fillers, a doubled word and "
        "'night'/'eff' for 'knight'/'f'. The repair is the model's "
        "understanding, never a parser rule."
    ),
    dev=(
        Variant(
            "uh_night_eff_three",
            (
                Say(
                    "uh take back that uh move and and put my night on eff "
                    "three instead"
                ),
            ),
            AFTER_E4_E5,
        ),
    ),
    heldout=(
        Variant(
            "um_knight_f_three",
            (Say("um undo my last move please then uh knight to f three"),),
            AFTER_E4_E5,
        ),
    ),
    checkpoints=(
        Checkpoint("took_back", lambda e: bool(e.turn(1).ran("undo"))),
        Checkpoint("played_Nf3", lambda e: e.history()[:1] == ["Nf3"]),
        Checkpoint("one_exchange", lambda e: len(e.history()) == 2),
    ),
)

CONSTRAINT_KEEPS_DIFFICULTY = Scenario(
    name="constraint_keeps_difficulty",
    tier=1,
    why=(
        "A request whose one obvious lever is ruled out in the same breath, "
        "beside a move: the move must land and the setting must not move."
    ),
    dev=(
        Variant(
            "go_easy_leave_it_queen_pawn",
            (
                Say(
                    "go easy on me this game but leave the difficulty setting "
                    "exactly where it is, and open with the queen pawn two squares"
                ),
            ),
            FRESH,
        ),
    ),
    heldout=(
        Variant(
            "be_gentle_dont_touch_d_pawn",
            (
                Say(
                    "be gentle with me, just don't change the difficulty, and "
                    "start with the d-pawn up two"
                ),
            ),
            FRESH,
        ),
    ),
    checkpoints=(
        Checkpoint(
            "difficulty_untouched",
            lambda e: difficulty(e.settings()) == difficulty(e.settings(0)),
        ),
        Checkpoint("played_d4", lambda e: e.history()[:1] == ["d4"]),
        completed(1),
    ),
)

# --- tier 2: threads ------------------------------------------------------------------

KNIGHT_ASK_THEN_CHANGE_OF_MIND = Scenario(
    name="knight_ask_then_change_of_mind",
    tier=2,
    why=(
        "An ambiguous ask must be asked about; the player then drops it for a "
        "different move, and finally asks a judgment question that must move "
        "nothing. Three turns, each depending on the one before."
    ),
    dev=(
        Variant(
            "move_knight_then_d4",
            (
                Say("move my knight"),
                Say("actually forget the knight, push the queen's pawn two squares"),
                Say("so am I doing better than before?"),
            ),
            FRESH,
        ),
        Variant(
            "develop_knight_then_d4",
            (
                Say("develop one of my knights"),
                Say("no wait, never mind that — queen pawn forward two"),
                Say("is my position good now?"),
            ),
            FRESH,
        ),
    ),
    heldout=(
        Variant(
            "knight_out_then_d_pawn",
            (
                Say("bring a knight out"),
                Say("hmm, no — the d-pawn forward two instead"),
                Say("who's better right now?"),
            ),
            FRESH,
        ),
    ),
    checkpoints=(
        Checkpoint("asked_first", lambda e: e.turn(1).asked and not e.turn(1).moved),
        Checkpoint(
            "no_knight_played",
            lambda e: not any(san.startswith("N") for san in e.history()[::2]),
        ),
        Checkpoint("played_d4", lambda e: e.history(after_turn=2)[:1] == ["d4"]),
        Checkpoint("judged", lambda e: judged(e.turn(3))),
        still(3),
    ),
)

SUGGESTED_MOVE_LATER = Scenario(
    name="suggested_move_later",
    tier=2,
    why=(
        "A suggestion made two turns earlier must be the move played: 'the "
        "move you suggested' is only resolvable from the conversation."
    ),
    dev=(
        Variant(
            "what_would_you_play",
            (
                Say("what would you play here if you were me?"),
                Say("and am I better or worse right now?"),
                Say("ok, play the move you suggested"),
            ),
            AFTER_E4_E5,
        ),
    ),
    heldout=(
        Variant(
            "recommendation",
            (
                Say("got a recommendation for my next move?"),
                Say("who's ahead at the moment?"),
                Say("go with your recommendation"),
            ),
            AFTER_E4_E5,
        ),
    ),
    checkpoints=(
        Checkpoint("consulted", lambda e: suggested(e.turn(1)) is not None),
        still(1),
        Checkpoint("judged", lambda e: judged(e.turn(2))),
        still(2),
        Checkpoint(
            "played_the_suggestion",
            lambda e: (
                suggested(e.turn(1)) is not None
                and e.history(after_turn=3)[2:3] == [suggested(e.turn(1))]
            ),
        ),
    ),
)

SAVE_RESET_RESUME = Scenario(
    name="save_reset_resume",
    tier=2,
    why=(
        "Save under a name, start over (the gate asks), play on, then bring "
        "the save back by paraphrase — no name given — over a game in "
        "progress, so the gate asks again."
    ),
    dev=(
        Variant(
            "opening",
            (
                Say("save this game as opening"),
                Say("start a fresh game"),
                Say("yes", model=False),
                Say("d4", model=False),
                Say("bring back the game I saved a minute ago"),
                Say("yes", model=False),
            ),
            AFTER_E4_E5,
        ),
    ),
    heldout=(
        Variant(
            "opening_reworded",
            (
                Say("keep this game under the name opening"),
                Say("let's begin a brand new game"),
                Say("yes", model=False),
                Say("d4", model=False),
                Say("load my earlier saved game back up"),
                Say("yes", model=False),
            ),
            AFTER_E4_E5,
        ),
    ),
    checkpoints=(
        Checkpoint("saved", lambda e: e.saved("opening") == ["e4", "e5"]),
        Checkpoint(
            "reset_asked", lambda e: e.turn(2).armed("new_game") and not e.turn(2).moved
        ),
        Checkpoint("reset", lambda e: e.history(after_turn=3) == []),
        Checkpoint("resume_asked", lambda e: e.turn(5).armed("resume_game")),
        Checkpoint("restored", lambda e: e.history() == ["e4", "e5"]),
    ),
)

DRAW_DECLINED_THEN_ADVICE = Scenario(
    name="draw_declined_then_advice",
    tier=2,
    why=(
        "An offer the rule declines, then a request that leans on it ('then "
        "what should I play'): advice from the engine, nothing moved, and no "
        "second offer."
    ),
    dev=(
        Variant(
            "call_it_a_draw",
            (
                Say("want to just call it a draw?"),
                Say("fine — then what's my best move here?"),
            ),
            after("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"),
        ),
    ),
    heldout=(
        Variant(
            "agree_to_a_draw",
            (
                Say("how about we agree to a draw?"),
                Say("ok, so what should I play instead?"),
            ),
            after("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"),
        ),
    ),
    checkpoints=(
        Checkpoint("offered", lambda e: bool(e.turn(1).ran("offer_draw"))),
        Checkpoint("game_goes_on", lambda e: not e.final["game_over"]),
        Checkpoint("consulted", lambda e: suggested(e.turn(2)) is not None),
        still(2),
        Checkpoint("no_second_offer", lambda e: not e.turn(2).ran("offer_draw")),
    ),
)

UNDO_CHAIN_ACROSS_TURNS = Scenario(
    name="undo_chain_across_turns",
    tier=2,
    why=(
        "Takebacks split across turns ('and the one before that too'), then "
        "a move on the board they leave: each turn has to know what the last "
        "one already did."
    ),
    dev=(
        Variant(
            "one_before_that",
            (
                Say("take back my last move"),
                Say("and the one before that too"),
                Say("now play d4"),
            ),
            after("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"),
        ),
    ),
    heldout=(
        Variant(
            "previous_as_well",
            (
                Say("undo my previous move"),
                Say("undo the move before it as well"),
                Say("go d4 now"),
            ),
            after("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"),
        ),
    ),
    checkpoints=(
        Checkpoint(
            "first_takeback",
            lambda e: e.history(after_turn=1) == ["e4", "e5", "Nf3", "Nc6"],
        ),
        Checkpoint(
            "second_takeback", lambda e: e.history(after_turn=2) == ["e4", "e5"]
        ),
        Checkpoint("played_d4", lambda e: e.history()[:3] == ["e4", "e5", "d4"]),
        Checkpoint("exactly_one_more_exchange", lambda e: len(e.history()) == 4),
    ),
)

DIFFICULTY_UP_AND_BACK = Scenario(
    name="difficulty_up_and_back",
    tier=2,
    why=(
        "Relative settings across turns, then 'back to where it was when we "
        "started' — a value only the conversation remembers. Starts at the "
        "bottom tier, so there is room for two steps up (a pilot sample could "
        "not tell a model that stopped from one that had already hit the top)."
    ),
    dev=(
        Variant(
            "harder_harder_back",
            (
                Say("make the engine harder"),
                Say("a bit harder still"),
                Say("actually put the difficulty back to what it was when we started"),
            ),
            at_tier("beginner"),
        ),
    ),
    heldout=(
        Variant(
            "crank_notch_reset",
            (
                Say("crank the difficulty up"),
                Say("one more notch up"),
                Say("return the difficulty to its level at the start of this chat"),
            ),
            at_tier("beginner"),
        ),
    ),
    checkpoints=(
        Checkpoint(
            "harder", lambda e: strength(e.settings(1)) > strength(e.settings(0))
        ),
        # A relative ask is a step, not the top: the baseline's first run
        # found "make the engine harder" from beginner going straight to
        # maximum every time, which left "harder still" nothing to do.
        Checkpoint(
            "one_step_not_the_top",
            lambda e: strength(e.settings(1)) < strength({"tier": "maximum"}),
        ),
        Checkpoint(
            "harder_still",
            lambda e: strength(e.settings(2)) > strength(e.settings(1)),
        ),
        Checkpoint(
            "back_where_it_started",
            lambda e: strength(e.settings(3)) == strength(e.settings(0)),
        ),
        Checkpoint("no_moves", lambda e: e.history() == []),
    ),
)

UNDO_THEN_AMBIGUOUS_BISHOP = Scenario(
    name="undo_then_ambiguous_bishop",
    tier=2,
    why=(
        "A takeback and an ambiguous replacement in one ask (#315's "
        "composition): the question must be about the board the undo left, "
        "and the answer next turn must land that move."
    ),
    dev=(
        Variant(
            "develop_bishop",
            (
                Say("undo that and develop my bishop instead"),
                Say("the one to c4"),
            ),
            AFTER_E4_E5_NF3_NC6,
        ),
    ),
    heldout=(
        Variant(
            "light_squared_bishop",
            (
                Say("take it back and get my light-squared bishop out"),
                Say("c4 one"),
            ),
            AFTER_E4_E5_NF3_NC6,
        ),
    ),
    checkpoints=(
        Checkpoint("took_back", lambda e: bool(e.turn(1).ran("undo"))),
        Checkpoint("asked", lambda e: e.turn(1).asked),
        Checkpoint(
            "no_bishop_guessed", lambda e: e.history(after_turn=1) == ["e4", "e5"]
        ),
        Checkpoint("played_Bc4", lambda e: e.history()[2:3] == ["Bc4"]),
    ),
)

NOISY_THREAD = Scenario(
    name="noisy_thread",
    tier=2,
    why=(
        "Speech-to-text noise across a thread: a takeback named by a misheard "
        "piece, its replacement on a misheard square, then a judgment."
    ),
    dev=(
        Variant(
            "night_see_three",
            (
                Say("uh can you like take back my night move"),
                Say("put the night on see three instead"),
                Say("um how am i doing now"),
            ),
            AFTER_E4_E5_NF3_NC6,
        ),
    ),
    heldout=(
        Variant(
            "night_sea_three",
            (
                Say("er undo the last night thing"),
                Say("knight to sea three then"),
                Say("so uh who's ahead"),
            ),
            AFTER_E4_E5_NF3_NC6,
        ),
    ),
    checkpoints=(
        Checkpoint("took_back", lambda e: e.history(after_turn=1) == ["e4", "e5"]),
        Checkpoint("played_Nc3", lambda e: e.history(after_turn=2)[2:3] == ["Nc3"]),
        Checkpoint("judged", lambda e: judged(e.turn(3))),
        still(3),
    ),
)

# --- tier 3: long sessions and long games ---------------------------------------------


def _long_session(says: tuple[str, ...]) -> tuple[Say, ...]:
    """The ten turns, the two moves on the parser's road."""
    model = [True] * 10
    model[0] = model[3] = False
    return tuple(Say(text, model=m) for text, m in zip(says, model, strict=True))


LONG_SESSION = Scenario(
    name="long_session",
    tier=3,
    why=(
        "Ten turns mixing every tool family: moves, a verdict, a setting, a "
        "suggestion played by reference, a named save, a takeback, a verbosity "
        "change and a description. Every turn is a checkpoint."
    ),
    dev=(
        Variant(
            "ten_turns",
            _long_session(
                (
                    "e4",
                    "how am I doing so far?",
                    "make the engine a bit easier for me",
                    "Nf3",
                    "what would you play here?",
                    "play your top pick",
                    "save this game as long_game",
                    "take back my last move",
                    "keep your replies short from now on",
                    "just describe where the pieces are",
                )
            ),
            FRESH,
        ),
    ),
    heldout=(
        Variant(
            "ten_turns_reworded",
            _long_session(
                (
                    "e4",
                    "am I winning?",
                    "lower the difficulty a little",
                    "Nf3",
                    "any suggestion for me here?",
                    "go with your first choice",
                    "store the game as long_game",
                    "undo my most recent move",
                    "be less wordy from here on",
                    "tell me where everything stands on the board, no evaluation",
                )
            ),
            FRESH,
        ),
    ),
    checkpoints=(
        Checkpoint("judged_2", lambda e: judged(e.turn(2))),
        still(2),
        Checkpoint(
            "easier_3", lambda e: strength(e.settings(3)) < strength(e.settings(0))
        ),
        Checkpoint("consulted_5", lambda e: suggested(e.turn(5)) is not None),
        still(5),
        Checkpoint(
            "played_the_suggestion_6",
            lambda e: (
                suggested(e.turn(5)) is not None
                and e.history(after_turn=6)[4:5] == [suggested(e.turn(5))]
            ),
        ),
        Checkpoint(
            "saved_7", lambda e: e.saved("long_game") == e.history(after_turn=6)
        ),
        Checkpoint(
            "took_back_8",
            lambda e: e.history(after_turn=8) == e.history(after_turn=6)[:-2],
        ),
        Checkpoint("terse_9", lambda e: e.settings(9)["verbosity"] == "low"),
        Checkpoint("described_10", lambda e: bool(e.turn(10).ran("describe_position"))),
        still(10),
    ),
)

LATE_GAME_REVIEW_UNDO_REPLAY = Scenario(
    name="late_game_review_undo_replay",
    tier=3,
    why=(
        "A 150-ply game: review it, go back to just before the player's "
        "worst move (more than a single undo can take), ask for the best move "
        "there, and play it."
    ),
    dev=(
        Variant(
            "worst_move_redo",
            (
                Say(
                    "review the whole game and tell me which of my moves was the worst"
                ),
                Say("take me back to just before that move"),
                Say("what's the best move in that position?"),
                Say("play it"),
            ),
            LATE_150,
        ),
    ),
    heldout=(
        Variant(
            "biggest_mistake_redo",
            (
                Say("go over this game — what was my biggest mistake?"),
                Say("rewind the game to right before I made it"),
                Say("what should I have played there?"),
                Say("make that move"),
            ),
            LATE_150,
        ),
    ),
    checkpoints=(
        Checkpoint("reviewed", lambda e: bool(e.turn(1).ran("review_game"))),
        Checkpoint("back_before_the_worst_move", _back_before_worst),
        Checkpoint("consulted", lambda e: suggested(e.turn(3)) is not None),
        still(3),
        Checkpoint("played_the_best_there", _played_best_there),
    ),
)

LATE_GAME_SAVE_UNDO_RESUME = Scenario(
    name="late_game_save_undo_resume",
    tier=3,
    why=(
        "An 84-ply game: save it, take back three of the player's moves (three "
        "whole exchanges), then restore the save over the shortened game — "
        "which the gate asks about."
    ),
    dev=(
        Variant(
            "before_undo",
            (
                Say("save this position as before_undo"),
                Say("take back my last three moves"),
                Say("actually restore the game I just saved"),
                Say("yes", model=False),
            ),
            LATE_84,
        ),
    ),
    heldout=(
        Variant(
            "before_undo_reworded",
            (
                Say("store this game as before_undo"),
                Say("undo my three most recent moves"),
                Say("on second thought, load that save back"),
                Say("yes", model=False),
            ),
            LATE_84,
        ),
    ),
    checkpoints=(
        Checkpoint("saved", lambda e: e.saved("before_undo") == e.start["history"]),
        Checkpoint(
            "undid_three",
            lambda e: e.history(after_turn=2) == e.start["history"][:-6],
        ),
        Checkpoint("resume_asked", lambda e: e.turn(3).armed("resume_game")),
        Checkpoint("restored", lambda e: e.history() == e.start["history"]),
    ),
)

SECOND_CHOICE_CHAIN = Scenario(
    name="second_choice_chain",
    tier=3,
    why=(
        "Five turns of reference: a verbosity setting, a two-candidate read, "
        "'not the top one, the other', a save of the result, and a verdict."
    ),
    dev=(
        Variant(
            "second_choice",
            (
                Say("keep it brief from now on"),
                Say("what are the two best moves here?"),
                Say("don't play the top one, play the other"),
                Say("save this as second_choice"),
                Say("how's my position now?"),
            ),
            AFTER_E4_E5,
        ),
    ),
    heldout=(
        Variant(
            "runner_up",
            (
                Say("shorter answers please, from here on"),
                Say("give me the engine's top two options"),
                Say("skip the first, go with the runner-up"),
                Say("store the game as second_choice"),
                Say("am I better or worse now?"),
            ),
            AFTER_E4_E5,
        ),
    ),
    checkpoints=(
        Checkpoint("terse", lambda e: e.settings(1)["verbosity"] == "low"),
        Checkpoint("two_candidates", lambda e: suggested(e.turn(2), 1) is not None),
        still(2),
        Checkpoint(
            "played_the_second",
            lambda e: (
                suggested(e.turn(2), 1) is not None
                and e.history(after_turn=3)[2:3] == [suggested(e.turn(2), 1)]
            ),
        ),
        Checkpoint(
            "saved", lambda e: e.saved("second_choice") == e.history(after_turn=3)
        ),
        Checkpoint("judged", lambda e: judged(e.turn(5))),
    ),
)


SCENARIOS: tuple[Scenario, ...] = (
    UNDO_REPLACE_AND_JUDGE,
    SETTINGS_MOVE_AND_VERDICT,
    TOP_MOVE_PLAY_AND_SAVE,
    NOISY_TAKEBACK_AND_REPLACE,
    CONSTRAINT_KEEPS_DIFFICULTY,
    KNIGHT_ASK_THEN_CHANGE_OF_MIND,
    SUGGESTED_MOVE_LATER,
    SAVE_RESET_RESUME,
    DRAW_DECLINED_THEN_ADVICE,
    UNDO_CHAIN_ACROSS_TURNS,
    DIFFICULTY_UP_AND_BACK,
    UNDO_THEN_AMBIGUOUS_BISHOP,
    NOISY_THREAD,
    LONG_SESSION,
    LATE_GAME_REVIEW_UNDO_REPLAY,
    LATE_GAME_SAVE_UNDO_RESUME,
    SECOND_CHOICE_CHAIN,
)
