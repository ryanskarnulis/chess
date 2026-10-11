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

Each split holds `WORDINGS_PER_SPLIT` wordings of its task (#339), and a
default run samples each once: one phrasing measures that phrasing, not the
task (corpus v1's dev and held-out wordings of one task swung by up to ten
samples). The wordings v2 added were written as one pool per scenario and
dealt to the splits by `random.Random(f"339:{name}")`, round-robin across a
scenario's setups where it has several, never by hand. A wording rewritten
later stays in its split.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess.pgn

from chessapp.game import GameSession
from chessapp.results import RESULTS_FILENAME, ResultsLog
from chessapp.speech_accuracy import score_record
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

# Black to move with one legal move, Kh7, and Qg7# after it: the player is
# Black and Glitch has a forced mate (#320).
_BLACK_FACING_MATE = "7k/8/5K2/8/8/8/8/6Q1 b - - 0 1"


def black_facing_mate(app: EvalApp) -> None:
    _save_dir(app)
    # Through `replace_session`, not a bare assignment: the swap has to bump
    # the board version, or `/api/state` keeps serving the opening position it
    # published and the response-is-the-board invariant sees two games.
    app.ctx.replace_session(
        GameSession(fen=_BLACK_FACING_MATE, player_color="black"), app.ctx.transcript
    )


# Kept for the harness's own tests, which script a sample on this position.
after_e4_e5 = AFTER_E4_E5

# --- wordings ------------------------------------------------------------------------

# Five per split: a default run (`CHESSAPP_FRONTIER_RUNS`) samples each once.
WORDINGS_PER_SPLIT = 5


def wordings(
    setup: Callable[[EvalApp], None] | None, **said: str | tuple[str | Say, ...]
) -> tuple[Variant, ...]:
    """A split's wordings of one task on one setup, by name. Each value is
    what the player says: one panel utterance, or a sequence of them, in
    which a `Say` marks a step that is not one (a parser move, a literal
    answer, a delegate thread's words)."""
    return tuple(
        Variant(
            name,
            tuple(
                step if isinstance(step, Say) else Say(step)
                for step in ((steps,) if isinstance(steps, str) else steps)
            ),
            setup,
        )
        for name, steps in said.items()
    )


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
    dev=wordings(
        AFTER_E4_E5,
        take_back_d4_better=(
            "take that back, play d4 instead, and tell me if it's any better"
        ),
        undo_go_d4_how_looks=(
            "undo my move, go d4, then tell me how the position looks for me"
        ),
        retract_d2_d4_assess=(
            "please retract my previous move, play d2 to d4 instead, and "
            "assess whether that improves my position"
        ),
        redo_with_d4_verdict=(
            "let me redo that: take back e4, play d4, and give me your "
            "verdict on the new position"
        ),
        swap_e4_for_d4="swap my e4 for d4 and then tell me if I'm doing better",
    ),
    heldout=wordings(
        AFTER_E4_E5,
        scratch_queen_pawn_read=(
            "scratch that move — push the queen pawn two instead and tell "
            "me whether I'm better off"
        ),
        rewind_d4_evaluate=(
            "rewind my last move, play d4 in its place, and then evaluate "
            "the position for me"
        ),
        not_e4_undo_d4_who_better=(
            "actually no, not e4 — undo it, play d4, and tell me who's "
            "better after that"
        ),
        queens_pawn_instead_stronger=(
            "take back my opening move and open with the queen's pawn two "
            "squares instead — is that stronger for me?"
        ),
        undo_then_d4_am_i_ahead="undo, then d4, then tell me if I'm ahead",
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
    dev=wordings(
        FRESH,
        voice_easier_king_pawn=(
            "turn spoken replies on, make the engine a little easier, then "
            "push my king pawn two squares and tell me how it looks"
        ),
        read_aloud_kings_pawn_judge=(
            "read your replies aloud, make the bot a touch easier, push the "
            "king's pawn up two, and judge the position after"
        ),
        level_down_speech_e4=(
            "lower the level by one, turn speech on, then play e4 and tell "
            "me if it's good for me"
        ),
        polite_voice_easier_e_pawn=(
            "could you please turn on voice, make the game slightly easier "
            "for me, play the e-pawn forward two squares, and let me know "
            "whether I'm doing well"
        ),
        audio_weaker_e2_e4_evaluate=(
            "enable audio replies, drop the engine strength slightly, move "
            "e2 to e4, then evaluate it"
        ),
    ),
    heldout=wordings(
        FRESH,
        talk_out_loud_softer_e_pawn=(
            "start talking to me out loud, tone the difficulty down a "
            "notch, open with the e-pawn up two, and say whether that's "
            "good for me"
        ),
        terse_speak_easier_e4_verdict=(
            "speak your answers, easier engine, e4, then your verdict"
        ),
        talk_out_loud_go_easier=(
            "I want you to talk out loud from now on and go a little easier "
            "on me. Open with e4 and tell me if that's a strong move"
        ),
        e4_first_then_settings=(
            "play e4, then turn on spoken replies and make the engine a bit "
            "weaker, and tell me if I'm better or worse"
        ),
        voice_lower_e4_good_start=(
            "switch voice output on, lower the difficulty a bit, play e4, "
            "and tell me if it's a good start"
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
    dev=wordings(
        AFTER_E4_E5,
        stockfish_top_then_checkpoint=(
            "ask Stockfish for its top move, play it for me, and then save "
            "the game as checkpoint"
        ),
        optimal_move_save_position=(
            "show me the optimal move, play it, then save this position as checkpoint"
        ),
        what_would_stockfish_play=(
            "what would Stockfish play here? play that for me, then save "
            "this game as checkpoint"
        ),
        engine_top_choice_save_under=(
            "have the engine pick its top choice and play it, then save "
            "under checkpoint"
        ),
        strongest_move_call_it=(
            "play the strongest move for me, then save the game and call it checkpoint"
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5,
        engine_pick_then_store=(
            "get the engine's best suggestion, make that move, and store "
            "this game under the name checkpoint"
        ),
        look_up_best_make_it=(
            "look up the engine's best move, make it, and save the game "
            "with the name checkpoint"
        ),
        let_stockfish_choose=(
            "let Stockfish choose my move — the top one — play it, and "
            "store the game as checkpoint"
        ),
        computer_number_one=(
            "get the computer's number one move for me, play it, and save "
            "everything as checkpoint"
        ),
        best_move_play_save=(
            "find the best move here, play it, and save the game as checkpoint"
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
    dev=wordings(
        AFTER_E4_E5,
        uh_night_eff_three=(
            "uh take back that uh move and and put my night on eff three instead"
        ),
        mm_undo_first_knight_eff="mm undo my move first and then knight to eff three",
        take_it_back_knight_eff_tree=(
            "take it back take it back and um knight eff tree instead"
        ),
        okay_so_takeback_and_and=(
            "okay so um take back what I just played and and play knight f "
            "three instead"
        ),
        like_go_back_knight_f_three=(
            "can you like go back one move and uh move my knight to like f three"
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5,
        um_knight_f_three="um undo my last move please then uh knight to f three",
        scratch_put_nite_f3="uh scratch that move and put the nite on f3",
        the_the_night_eff_three=(
            "uh take back the the last move and put the night on eff three"
        ),
        er_undo_night_f_three="er undo that one and uh night to f three",
        hmm_reverse_night_f_three="hmm reverse that move and uh go night f three",
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
    dev=wordings(
        FRESH,
        go_easy_leave_it_queen_pawn=(
            "go easy on me this game but leave the difficulty setting "
            "exactly where it is, and open with the queen pawn two squares"
        ),
        take_it_easy_keep_level=(
            "take it easy on me, but keep the difficulty level as it is, and play d4"
        ),
        relaxed_game_untouched=(
            "let's have a relaxed game — difficulty stays untouched — and "
            "I'll open with d4"
        ),
        gently_same_difficulty=(
            "play gently against me, keep the same difficulty though, and "
            "push my queen's pawn up two"
        ),
        be_nice_no_settings=(
            "be nice to me today, without changing any settings, and start "
            "with the queen's pawn two squares"
        ),
    ),
    heldout=wordings(
        FRESH,
        be_gentle_dont_touch_d_pawn=(
            "be gentle with me, just don't change the difficulty, and start "
            "with the d-pawn up two"
        ),
        dont_crush_me=(
            "don't crush me too hard this game — but don't touch the "
            "difficulty setting — and open with d4"
        ),
        easy_game_dont_lower=(
            "I'd like an easy game, but don't lower the difficulty — just "
            "play d2 to d4 for me"
        ),
        go_soft_level_stays=(
            "go soft on me but the level stays where it is. d4 to start"
        ),
        mercy_strength_alone=(
            "show me some mercy, but leave the engine strength alone, and "
            "open with the d-pawn two squares"
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
    dev=wordings(
        FRESH,
        move_knight_then_d4=(
            "move my knight",
            "actually forget the knight, push the queen's pawn two squares",
            "so am I doing better than before?",
        ),
        develop_knight_then_d4=(
            "develop one of my knights",
            "no wait, never mind that — queen pawn forward two",
            "is my position good now?",
        ),
        horse_out_scratch_that=(
            "let's get a horse out",
            "nah, scratch that, d-pawn two squares forward",
            "what's the evaluation now?",
        ),
        knight_move_then_d2_d4=(
            "play a knight move",
            "never mind the knight, do d2 to d4",
            "am I winning?",
        ),
        knight_out_then_skip=(
            "get one of my knights out",
            "on second thought, skip the knight and play d4",
            "how's my position looking now, am I better?",
        ),
    ),
    heldout=wordings(
        FRESH,
        knight_out_then_d_pawn=(
            "bring a knight out",
            "hmm, no — the d-pawn forward two instead",
            "who's better right now?",
        ),
        develop_a_knight_forget_it=(
            "I want to develop a knight",
            "actually no, forget knights for now — push d4",
            "is that good for me?",
        ),
        one_of_my_knights_changed_mind=(
            "move one of my knights",
            "hold on, I changed my mind: d4 instead",
            "who's doing better at this point?",
        ),
        jump_knight_change_plans=(
            "jump a knight out",
            "wait, change of plans — queen's pawn two squares instead",
            "so who's ahead now?",
        ),
        could_you_knight_oh_wait=(
            "could you move a knight for me",
            "oh wait, never mind — play the queen's pawn up two",
            "and how do I stand now, better or worse?",
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
    dev=wordings(
        AFTER_E4_E5,
        what_would_you_play=(
            "what would you play here if you were me?",
            "and am I better or worse right now?",
            "ok, play the move you suggested",
        ),
        stockfish_top_move=(
            "what's Stockfish's top move here?",
            "and what's the evaluation at the moment?",
            "play the top move you gave me",
        ),
        best_move_then_play_it=(
            "what's the best move for me here?",
            "and how am I doing, better or worse?",
            "ok, play the move you suggested before",
        ),
        suggest_a_move=(
            "suggest a move for me",
            "is my position any good?",
            "play your suggestion",
        ),
        top_suggestion=(
            "what's your top suggestion for me here?",
            "am I winning or losing at the moment?",
            "play your top suggestion",
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5,
        recommendation=(
            "got a recommendation for my next move?",
            "who's ahead at the moment?",
            "go with your recommendation",
        ),
        one_move_youd_play=(
            "what's the one move you'd play next?",
            "who's winning right now?",
            "let's go with what you suggested",
        ),
        which_move_recommend=(
            "which move would you recommend here?",
            "am I ahead right now?",
            "fine, play the move you recommended",
        ),
        recommend_me_a_move=(
            "recommend me a move",
            "is the game balanced right now?",
            "I'll follow your recommendation, play it",
        ),
        if_you_were_white=(
            "if you were playing white here, what would you pick?",
            "is the position in my favour right now?",
            "go ahead and play your pick",
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
    dev=wordings(
        AFTER_E4_E5,
        opening=(
            "save this game as opening",
            "start a fresh game",
            Say("yes", model=False),
            Say("d4", model=False),
            "bring back the game I saved a minute ago",
            Say("yes", model=False),
        ),
        under_the_name=(
            "please save under the name opening",
            "begin a new game",
            Say("yes", model=False),
            Say("d4", model=False),
            "reload the save I made before",
            Say("yes", model=False),
        ),
        could_you_save=(
            "could you save this game as opening",
            "start over with a new game",
            Say("yes", model=False),
            Say("d4", model=False),
            "resume the game I saved",
            Say("yes", model=False),
        ),
        call_it_opening=(
            "save the game and call it opening",
            "new game please",
            Say("yes", model=False),
            Say("d4", model=False),
            "restore the game I saved earlier",
            Say("yes", model=False),
        ),
        store_as_opening=(
            "store this as opening",
            "reset the board, I want to start over",
            Say("yes", model=False),
            Say("d4", model=False),
            "go back to my saved game",
            Say("yes", model=False),
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5,
        opening_reworded=(
            "keep this game under the name opening",
            "let's begin a brand new game",
            Say("yes", model=False),
            Say("d4", model=False),
            "load my earlier saved game back up",
            Say("yes", model=False),
        ),
        save_this_position=(
            "save this position under opening",
            "fresh board please",
            Say("yes", model=False),
            Say("d4", model=False),
            "load up the game I stored",
            Say("yes", model=False),
        ),
        save_my_progress=(
            "save my progress as opening",
            "let's start from scratch",
            Say("yes", model=False),
            Say("d4", model=False),
            "bring my saved game back",
            Say("yes", model=False),
        ),
        name_it_opening=(
            "save it — name it opening",
            "wipe the board, new game",
            Say("yes", model=False),
            Say("d4", model=False),
            "get my earlier save back",
            Say("yes", model=False),
        ),
        make_a_save=(
            "make a save called opening",
            "clear the board and start again",
            Say("yes", model=False),
            Say("d4", model=False),
            "open the saved game from before",
            Say("yes", model=False),
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
    dev=wordings(
        after("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"),
        call_it_a_draw=(
            "want to just call it a draw?",
            "fine — then what's my best move here?",
        ),
        accept_a_draw=(
            "would you accept a draw?",
            "no? then tell me a strong move to play",
        ),
        i_offer_a_draw=("I offer a draw", "alright, then what's the best move for me?"),
        take_a_draw_here=(
            "would you take a draw here?",
            "ok then, what's my strongest option?",
        ),
        end_it_as_a_draw=(
            "let's end this one as a draw",
            "fine, then what's the best continuation for me?",
        ),
    ),
    heldout=wordings(
        after("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"),
        agree_to_a_draw=(
            "how about we agree to a draw?",
            "ok, so what should I play instead?",
        ),
        agree_a_draw=(
            "do you want to agree a draw?",
            "fair enough. what move do you recommend?",
        ),
        draw_offer_what_say=(
            "draw offer — what do you say?",
            "alright, suggest my next move then",
        ),
        i_propose_a_draw=("I propose a draw", "then what should my next move be?"),
        split_the_point=(
            "shall we split the point?",
            "okay, what do you suggest I play then?",
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
    dev=wordings(
        after("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"),
        one_before_that=(
            "take back my last move",
            "and the one before that too",
            "now play d4",
        ),
        can_you_undo=(
            "can you undo that last move?",
            "the one before that as well, please",
            "and now d2-d4",
        ),
        lets_take_back=(
            "let's take back my last move",
            "actually, take back one more",
            "and now play d4 please",
        ),
        bishop_then_knight=(
            "take back my bishop move",
            "take back my knight move too",
            "then play d4",
        ),
        undo_another_one=(
            "undo my last move please",
            "and undo another one",
            "now play d4",
        ),
    ),
    heldout=wordings(
        after("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"),
        previous_as_well=(
            "undo my previous move",
            "undo the move before it as well",
            "go d4 now",
        ),
        undo_by_name=("undo Bc4", "and undo Nf3 too", "now play d4 instead"),
        rewind_one_more=(
            "rewind one move",
            "rewind one more",
            "play the queen's pawn two squares",
        ),
        go_back_one_more=(
            "go back a move",
            "go back one more",
            "and push the d-pawn two squares",
        ),
        want_it_back=(
            "I want to take my last move back",
            "and the previous one too",
            "then go with d4",
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
    dev=wordings(
        at_tier("beginner"),
        harder_harder_back=(
            "make the engine harder",
            "a bit harder still",
            "actually put the difficulty back to what it was when we started",
        ),
        step_up=(
            "step the difficulty up",
            "and another step up",
            "return to the original difficulty",
        ),
        tougher_still_reset=(
            "make it a little tougher",
            "tougher still",
            "reset the difficulty to where we began",
        ),
        stronger_bot=(
            "make the bot stronger",
            "a little stronger again",
            "put the bot back to how strong it was at first",
        ),
        bump_strength=(
            "bump the engine strength up",
            "bump it up once more",
            "put the strength back to the starting level",
        ),
    ),
    heldout=wordings(
        at_tier("beginner"),
        crank_notch_reset=(
            "crank the difficulty up",
            "one more notch up",
            "return the difficulty to its level at the start of this chat",
        ),
        increase_again_originally=(
            "increase the difficulty",
            "increase it again",
            "now set it back to what it was originally",
        ),
        play_harder=(
            "can you play harder?",
            "harder than that",
            "never mind, restore the starting difficulty",
        ),
        raise_the_level=(
            "raise the level",
            "raise it one more step",
            "back to the level we started on, please",
        ),
        harder_opponent=(
            "I want a harder opponent",
            "even harder please",
            "ok, go back to the difficulty we had at the start",
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
    dev=wordings(
        AFTER_E4_E5_NF3_NC6,
        develop_bishop=("undo that and develop my bishop instead", "the one to c4"),
        knight_back_bishop_out=(
            "take back my knight move and bring a bishop out instead",
            "put it on c4",
        ),
        undo_move_bishop_out=(
            "undo my last move and move my bishop out",
            "the one that goes to c4",
        ),
        cancel_knight_bring_bishop=(
            "cancel that knight move and bring out my bishop",
            "c4 square please",
        ),
        scrap_knight_rather_bishop=(
            "scrap that knight move, I'd rather move my bishop",
            "c4 for the bishop",
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5_NF3_NC6,
        light_squared_bishop=(
            "take it back and get my light-squared bishop out",
            "c4 one",
        ),
        undo_nf3_develop=("undo Nf3 and develop the bishop", "the c4 square"),
        rewind_bishop_move=(
            "rewind and play a bishop move instead of the knight",
            "the c4 option",
        ),
        take_back_light_bishop=(
            "take back Nf3 and develop my light bishop",
            "c4 would be my pick",
        ),
        go_back_get_bishop=(
            "go back one move and get the bishop out instead",
            "let's do the c4 one",
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
    dev=wordings(
        AFTER_E4_E5_NF3_NC6,
        night_see_three=(
            "uh can you like take back my night move",
            "put the night on see three instead",
            "um how am i doing now",
        ),
        scratch_night_better=(
            "uh scratch my night move",
            "um play night c three",
            "so uh am i better",
        ),
        knight_thing_c_tree=(
            "take back the uh the knight thing",
            "uh night on c tree",
            "so who's ahead now huh",
        ),
        undo_the_um_knight=(
            "undo the um the knight",
            "nite to sea three then",
            "who's winning uh now",
        ),
        like_knight_see_three=(
            "like take back my last knight move",
            "night to see three",
            "am i doing ok um now",
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5_NF3_NC6,
        night_sea_three=(
            "er undo the last night thing",
            "knight to sea three then",
            "so uh who's ahead",
        ),
        go_back_night_eval=(
            "er go back on that night move",
            "knight to uh c three",
            "uh what's the eval now",
        ),
        the_the_knight_c_three=(
            "can you uh undo the the knight move",
            "uh put the knight on c three this time",
            "er who's better now",
        ),
        nite_c_three_winning=(
            "um take back my nite move",
            "uh nite to c three",
            "so am i winning or what",
        ),
        horse_sea_three=(
            "can you uh like undo my horse move",
            "horse to sea three",
            "am i um ahead",
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


# The ten turns of `long_session` put two moves on the parser's road: turns 1
# and 4 are `Say(..., model=False)` in every wording.

LONG_SESSION = Scenario(
    name="long_session",
    tier=3,
    why=(
        "Ten turns mixing every tool family: moves, a verdict, a setting, a "
        "suggestion played by reference, a named save, a takeback, a verbosity "
        "change and a description. Every turn is a checkpoint."
    ),
    dev=wordings(
        FRESH,
        ten_turns=(
            Say("e4", model=False),
            "how am I doing so far?",
            "make the engine a bit easier for me",
            Say("Nf3", model=False),
            "what would you play here?",
            "play your top pick",
            "save this game as long_game",
            "take back my last move",
            "keep your replies short from now on",
            "just describe where the pieces are",
        ),
        going_lower_pick=(
            Say("e4", model=False),
            "how's my game going, am I better?",
            "go easier on me, lower the level",
            Say("Nf3", model=False),
            "which move would you pick here?",
            "make that move",
            "save under the name long_game",
            "take back my most recent move",
            "be more concise from now on",
            "where are the pieces right now? no evaluation, just the layout",
        ),
        ahead_tone_recommend=(
            Say("e4", model=False),
            "who's ahead?",
            "tone the engine down a little",
            Say("Nf3", model=False),
            "recommend a move",
            "play your recommendation",
            "save this as long_game",
            "undo the last move",
            "keep it brief from now on",
            "list where all the pieces are",
        ),
        evaluate_strength_stockfish=(
            Say("e4", model=False),
            "evaluate my position",
            "lower the engine strength a bit",
            Say("Nf3", model=False),
            "what's Stockfish's top move?",
            "go ahead and play that",
            "save a copy as long_game",
            "rewind my last move",
            "less talking from here on",
            "tell me where every piece is",
        ),
        good_easier_best_it=(
            Say("e4", model=False),
            "is my position good?",
            "make it easier",
            Say("Nf3", model=False),
            "what's the best move now?",
            "play it",
            "save the game as long_game",
            "undo the move I just played",
            "shorter replies from now on",
            "describe the position for me",
        ),
    ),
    heldout=wordings(
        FRESH,
        ten_turns_reworded=(
            Say("e4", model=False),
            "am I winning?",
            "lower the difficulty a little",
            Say("Nf3", model=False),
            "any suggestion for me here?",
            "go with your first choice",
            "store the game as long_game",
            "undo my most recent move",
            "be less wordy from here on",
            "tell me where everything stands on the board, no evaluation",
        ),
        better_weaken_in_my_place=(
            Say("e4", model=False),
            "am I better or worse here?",
            "weaken the engine slightly",
            Say("Nf3", model=False),
            "what would you play in my place?",
            "play that for me",
            "save the current game as long_game",
            "take my last move back",
            "make your answers shorter going forward",
            "describe where the pieces stand",
        ),
        well_notch_got_move=(
            Say("e4", model=False),
            "am I doing well so far?",
            "drop the difficulty a notch",
            Say("Nf3", model=False),
            "got a move for me?",
            "play the move you suggested",
            "save it as long_game",
            "go back one move",
            "short answers from now on, please",
            "describe the board, just the piece positions",
        ),
        winning_weaker_suggest=(
            Say("e4", model=False),
            "who's winning so far?",
            "make the computer a bit weaker",
            Say("Nf3", model=False),
            "suggest my next move",
            "go with your suggestion",
            "save the game, call it long_game",
            "can you undo my last move",
            "please be terse from now on",
            "just tell me the piece placement",
        ),
        eval_easier_suggestion=(
            Say("e4", model=False),
            "what's the evaluation?",
            "easier opponent please",
            Say("Nf3", model=False),
            "your best move suggestion?",
            "alright, play your suggestion",
            "keep a save called long_game",
            "undo that last move",
            "talk less from now on",
            "give me a description of the position, no assessment",
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
    dev=wordings(
        LATE_150,
        worst_move_redo=(
            "review the whole game and tell me which of my moves was the worst",
            "take me back to just before that move",
            "what's the best move in that position?",
            "play it",
        ),
        most_wrong=(
            "do a game review — where did I go most wrong?",
            "take me back to that point, before the mistake",
            "what does the engine want there?",
            "play the engine's move",
        ),
        single_worst=(
            "review my game and point out my single worst move",
            "undo back to right before it",
            "best move in that spot?",
            "go ahead and play it",
        ),
        lost_the_most=(
            "full game review please — which move of mine lost the most?",
            "set the board back to the position before that move",
            "what was best in that position?",
            "play the best one",
        ),
        analyse_find_worst=(
            "analyse the full game and find my worst move",
            "go back to the position right before it",
            "what's the best move there?",
            "play the best move",
        ),
    ),
    heldout=wordings(
        LATE_150,
        biggest_mistake_redo=(
            "go over this game — what was my biggest mistake?",
            "rewind the game to right before I made it",
            "what should I have played there?",
            "make that move",
        ),
        biggest_blunder_review=(
            "which move of mine was the biggest blunder this game? review it",
            "rewind to just before that blunder",
            "what was the right move?",
            "play that",
        ),
        hurt_me_most=(
            "go through my moves and find the one that hurt me most",
            "go back to right before that one",
            "what's the strongest move there?",
            "make it",
        ),
        run_a_review=(
            "run a review and tell me my worst move",
            "take the game back to just before I played it",
            "what would Stockfish have played?",
            "play Stockfish's choice",
        ),
        worst_decision=(
            "look through the whole game: what was my worst decision?",
            "return the board to the moment before that move",
            "what should I have played instead?",
            "play the better move",
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
    dev=wordings(
        LATE_84,
        before_undo=(
            "save this position as before_undo",
            "take back my last three moves",
            "actually restore the game I just saved",
            Say("yes", model=False),
        ),
        save_named_go_back=(
            "make a save named before_undo",
            "go back three moves",
            "restore my save from a moment ago",
            Say("yes", model=False),
        ),
        please_save_rewind=(
            "please save as before_undo",
            "rewind my last three moves",
            "ok, put the saved game back",
            Say("yes", model=False),
        ),
        save_under_previous_three=(
            "save this under before_undo",
            "take back my previous three moves",
            "load the game I just saved",
            Say("yes", model=False),
        ),
        store_a_save=(
            "store a save called before_undo",
            "take three of my moves back",
            "actually, go back to that save",
            Say("yes", model=False),
        ),
    ),
    heldout=wordings(
        LATE_84,
        before_undo_reworded=(
            "store this game as before_undo",
            "undo my three most recent moves",
            "on second thought, load that save back",
            Say("yes", model=False),
        ),
        bring_back_the_save=(
            "save the game as before_undo",
            "undo three of my moves",
            "never mind, bring back the save",
            Say("yes", model=False),
        ),
        name_it_undo_3=(
            "save my game, name it before_undo",
            "undo my last 3 moves",
            "restore the save I just made",
            Say("yes", model=False),
        ),
        create_step_back=(
            "create a save named before_undo",
            "step back three of my moves",
            "bring back what I saved",
            Say("yes", model=False),
        ),
        call_it_changed_mind=(
            "save this game — call it before_undo",
            "undo the last three moves I made",
            "changed my mind, load my save",
            Say("yes", model=False),
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
    dev=wordings(
        AFTER_E4_E5,
        second_choice=(
            "keep it brief from now on",
            "what are the two best moves here?",
            "don't play the top one, play the other",
            "save this as second_choice",
            "how's my position now?",
        ),
        terse_two_options=(
            "terse mode from now on",
            "give me two move options",
            "play the other one, not the best",
            "make a save called second_choice",
            "what's the evaluation now?",
        ),
        answers_short_not_first=(
            "keep your answers short",
            "list the two strongest moves",
            "not the first — play the second",
            "store this game as second_choice",
            "am I winning now?",
        ),
        briefly_second_best=(
            "please answer briefly from now on",
            "what are the best two moves for me?",
            "the second-best one — play that",
            "save this as second_choice",
            "is my position good now?",
        ),
        short_and_sweet_number_two=(
            "short and sweet from now on",
            "name the two top moves here",
            "play number two",
            "save the game with the name second_choice",
            "am I better off now?",
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5,
        runner_up=(
            "shorter answers please, from here on",
            "give me the engine's top two options",
            "skip the first, go with the runner-up",
            "store the game as second_choice",
            "am I better or worse now?",
        ),
        going_forward_next_one=(
            "brief answers going forward",
            "two best moves?",
            "skip the top choice and play the next one",
            "save the game as second_choice",
            "who's ahead now?",
        ),
        less_wordy_runner_up=(
            "less wordy from now on",
            "top two moves please",
            "I'll take the runner-up, play it",
            "save it as second_choice",
            "how do I stand now — better or worse?",
        ),
        short_candidates=(
            "short replies from here, please",
            "what are your two best candidate moves?",
            "go with the second candidate",
            "save the game under second_choice",
            "evaluate my position now",
        ),
        brief_top_two_second=(
            "be brief from now on",
            "show me the top two moves",
            "play the second one",
            "save as second_choice",
            "who's better now?",
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


# --- the open question (#319) ---------------------------------------------------------
#
# A question asked in one turn and answered several turns later, from a record
# the harness keeps (`clarification.py`), not from the narrator's wording. The
# three numbers #319 asks for read off these: a move on the ask turn is the
# ambiguous-request mutation rate, the final pick is successful resolution,
# and a stale question must never be answered by a move.


def asked(index: int) -> Checkpoint:
    """Turn `index` asked which move, and moved nothing doing it."""
    return Checkpoint(
        f"asked_{index}", lambda e: e.turn(index).asked and not e.turn(index).moved
    )


def _played_offered(asked_turn: int, index: int, n: int) -> Callable:
    def check(e: Episode) -> bool:
        ask = e.turn(asked_turn).result("ask_player")
        offered = (ask or {}).get("candidates") or []
        start = len(e.turn(index).before["history"])
        played = e.history(after_turn=index)[start : start + 1]
        return len(offered) >= n and played == offered[n - 1 : n]

    return check


def first_offered(asked_turn: int, index: int) -> Checkpoint:
    """Turn `index`'s first move is the first candidate turn `asked_turn`'s
    `ask_player` offered — read off that ask's own result, so the check
    follows whatever the planner listed (#348 lists all four knight moves for
    a king's-knight ask) rather than a list written here. The move is counted
    from where turn `index` began, not from the start of the game, so a setup
    that places moves first (#352's 1.e4 e5) is graded on the answer."""
    return Checkpoint("played_first_offered", _played_offered(asked_turn, index, 1))


def second_offered(asked_turn: int, index: int) -> Checkpoint:
    """`first_offered`, for the second candidate: a reply that names nothing
    but a place in the question's own list."""
    return Checkpoint("played_second_offered", _played_offered(asked_turn, index, 2))


def picked(index: int, san: str, ply: int = 0) -> Checkpoint:
    """After turn `index`, the move at `ply` of the game is `san`."""
    return Checkpoint(
        f"played_{san}", lambda e: e.history(after_turn=index)[ply : ply + 1] == [san]
    )


KNIGHT_ASK_ASIDE_THEN_PICK = Scenario(
    name="knight_ask_aside_then_pick",
    tier=2,
    why=(
        "A question survives an unrelated read between it and its answer: the "
        "answer arrives two turns after the ask, with an aside in the middle "
        "that must move nothing."
    ),
    dev=wordings(
        FRESH,
        difficulty_aside=(
            "move my kings knight",
            "what difficulty am I on?",
            "the one to f3",
        ),
        voice_aside=(
            "develop my king side knight",
            "is voice output on right now?",
            "put it on f3",
        ),
        kingside_verbose_aside=(
            "develop the kingside knight",
            "how verbose are you set to?",
            "to f3, please",
        ),
        play_kingside_voice_on=(
            "play my kingside knight",
            "is the voice turned on?",
            "the one going to f3",
        ),
        g_knight_voice_aside=(
            "get my g-knight out",
            "are spoken replies on?",
            "the f3 one",
        ),
    ),
    heldout=wordings(
        FRESH,
        verbosity_aside=(
            "bring the king's knight out",
            "how chatty are you set to be?",
            "the f3 square one",
        ),
        g1_out_difficulty_aside=(
            "bring out my g1 knight",
            "what's the current difficulty?",
            "the square f3",
        ),
        knight_from_g1_setting=(
            "move my knight from g1",
            "what difficulty setting are we on?",
            "the f3 square, please",
        ),
        want_kings_knight_voice=(
            "I want to move my king's knight",
            "quick check: is voice on?",
            "let's go f3 with it",
        ),
        g1_knight_level_aside=(
            "move the knight on g1",
            "what level is the engine at?",
            "send it to f3",
        ),
    ),
    checkpoints=(asked(1), still(2), picked(3, "Nf3")),
)

KNIGHT_ASK_LONG_CHAT_THEN_PICK = Scenario(
    name="knight_ask_long_chat_then_pick",
    tier=2,
    why=(
        "Five turns of chat push the question out of the verbatim window "
        "(`conversation.RECENT_TURNS`): Glitch's words naming the candidates "
        "are gone from the planner's view, and the answer names no square, so "
        "only the kept record says what 'the first one' was."
    ),
    dev=wordings(
        FRESH,
        chat=(
            "move my kings knight",
            "hang on, what's your favourite opening?",
            "why do people like it?",
            "tell me a chess joke",
            "who was the best player ever?",
            "fair enough. okay, where were we",
            "the first one you offered",
        ),
        trap_chat=(
            "move the knight on my king's side",
            "sorry, side question: what's an opening trap?",
            "name a famous one",
            "how do I avoid it?",
            "cool",
            "anyway, my move",
            "the first of those",
        ),
        pin_skewer_chat=(
            "move my king's knight",
            "before I decide, what's a pin?",
            "and a skewer?",
            "which is more common?",
            "do grandmasters still fall for them?",
            "ok, back to my move",
            "the first option",
        ),
        tempo_chat=(
            "develop my king-side knight",
            "one sec, what does tempo mean?",
            "why does it matter?",
            "and what's a gambit?",
            "makes sense",
            "ok where was I",
            "let's do the first one",
        ),
        en_passant_chat=(
            "get the g-knight going",
            "pause — what's the en passant rule?",
            "why does that rule exist?",
            "who came up with it?",
            "huh, neat",
            "ok, back to it",
            "pick the first option from before",
        ),
    ),
    heldout=wordings(
        FRESH,
        lesson_chat=(
            "bring the king's knight out",
            "wait, before that: what does castling actually do?",
            "and when should I do it?",
            "what's a fork?",
            "any tips for a beginner?",
            "thanks. right, back to the game",
            "go with the first option you gave me",
        ),
        rating_chat=(
            "play my kingside knight",
            "hold on — what's the Elo rating system?",
            "what's a good rating for a beginner?",
            "and for a club player?",
            "thanks",
            "ok let's continue",
            "first one please",
        ),
        endgame_study_chat=(
            "my king's knight, move it",
            "quick tangent: what's the best way to study endgames?",
            "how long should I study each day?",
            "any books you'd recommend?",
            "thanks for that",
            "right, the move",
            "the first move you suggested",
        ),
        stalemate_chat=(
            "bring out my king's knight",
            "wait, what's stalemate exactly?",
            "is it a draw?",
            "how is it different from checkmate?",
            "got it",
            "back to the game",
            "I'll take the first option",
        ),
        history_chat=(
            "develop my g1 knight",
            "random question: who invented chess?",
            "how old is the game?",
            "when did the queen become powerful?",
            "interesting stuff",
            "alright, my move",
            "go with the first one you listed",
        ),
    ),
    checkpoints=(
        asked(1),
        Checkpoint(
            "chat_moved_nothing",
            lambda e: not any(e.turn(i).moved for i in range(2, 7)),
        ),
        first_offered(1, 7),
    ),
)

KNIGHT_ASK_THEN_BOARD_CHANGES = Scenario(
    name="knight_ask_then_board_changes",
    tier=2,
    why=(
        "Another client changes the board between the question and its "
        "answer. The answer refers only to the old question ('the first one'), "
        "so no candidate may be played from it: ask again or say so."
    ),
    dev=wordings(
        AFTER_E4_E5,
        other_client_moves=(
            "move my kings knight",
            Say("d4", origin="d0", model=False),
            "the first one",
        ),
        other_client_undoes=(
            "develop my king side knight",
            Say("take back the last move", origin="d0"),
            "the second one",
        ),
        other_client_h3=(
            "move my kingside knight",
            Say("h3", origin="d0", model=False),
            "let's go with the first one",
        ),
        other_client_c3=(
            "I'd like to move my king's knight",
            Say("c3", origin="d0", model=False),
            "the first of those",
        ),
        other_client_bc4=(
            "move my king's knight",
            Say("Bc4", origin="d0", model=False),
            "first one",
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5,
        other_client_develops=(
            "bring the king's knight out",
            Say("Nc3", origin="d0", model=False),
            "go with the first option",
        ),
        other_client_undoes_again=(
            "bring my g1 knight out",
            Say("undo the last move", origin="d0"),
            "the first option",
        ),
        other_client_d3=(
            "move the g1 knight",
            Say("d3", origin="d0", model=False),
            "go with the second one",
        ),
        other_client_takes_back=(
            "knight from g1, please",
            Say("take my last move back", origin="d0"),
            "the second of those",
        ),
        other_client_nc3=(
            "develop the kingside knight",
            Say("Nc3", origin="d0", model=False),
            "the second option",
        ),
    ),
    checkpoints=(
        asked(1),
        Checkpoint("board_changed", lambda e: e.turn(2).moved),
        still(3),
    ),
)

TWO_THREADS_SIMILAR_ASKS = Scenario(
    name="two_threads_similar_asks",
    tier=2,
    why=(
        "Two delegate threads each have a question open on one board — a "
        "knight in one, the e-pawn in the other. The second thread's 'the "
        "first one' must pick from its own question, never the first thread's."
    ),
    # The dev pawn ask was "push my e pawn", which the planner plays (e3)
    # rather than asks, so the scenario missed at asked_2 on every arm and never
    # reached the ordinal it is here to measure (#352; the ask miss is #357).
    dev=wordings(
        FRESH,
        knight_then_pawn=(
            Say("move my kings knight", origin="d0"),
            Say("move my e-file pawn", origin="d1"),
            Say("the first one", origin="d1"),
        ),
        g1_knight_advance_e_pawn=(
            Say("develop the g1 knight", origin="d0"),
            Say("advance my e-pawn", origin="d1"),
            Say("the first option", origin="d1"),
        ),
        g_knight_pawn_on_e2=(
            Say("I want to move my g-knight", origin="d0"),
            Say("move my pawn on e2", origin="d1"),
            Say("the first of those", origin="d1"),
        ),
        into_play_e_file=(
            Say("bring my kingside knight into play", origin="d0"),
            Say("move the pawn on the e-file", origin="d1"),
            Say("first option please", origin="d1"),
        ),
        knight_on_g1_front_of_king=(
            Say("move the knight on g1", origin="d0"),
            Say("advance the pawn in front of my king", origin="d1"),
            Say("let's do the first", origin="d1"),
        ),
    ),
    heldout=wordings(
        FRESH,
        knight_then_pawn=(
            Say("bring the king's knight out", origin="d0"),
            Say("move the pawn in front of my king", origin="d1"),
            Say("go with the first option", origin="d1"),
        ),
        play_knight_push_e_pawn=(
            Say("play my king's knight", origin="d0"),
            Say("push the e-pawn", origin="d1"),
            Say("go with the first one", origin="d1"),
        ),
        kingside_knight_e2_pawn=(
            Say("get my kingside knight out", origin="d0"),
            Say("move the e2 pawn", origin="d1"),
            Say("I'll take the first", origin="d1"),
        ),
        kings_knight_kings_pawn=(
            Say("move my king's knight", origin="d0"),
            Say("move my king's pawn", origin="d1"),
            Say("first one", origin="d1"),
        ),
        kings_side_knight_e_pawn=(
            Say("knight on the king's side — move it", origin="d0"),
            Say("I want to move my e-pawn", origin="d1"),
            Say("the first you offered", origin="d1"),
        ),
    ),
    checkpoints=(
        asked(1),
        asked(2),
        first_offered(2, 3),
    ),
)

# --- ordinals only the question resolves (#352) ---------------------------------------
#
# The king's-knight asks above offer their candidates in `legal_moves` order,
# Nh3 first, and before #351 the planner read "the first one" as
# `legal_moves[0]`: the pick landed whether or not it read the question. The
# planner still lists what fits in menu order (queen, bishop and pawn asks
# alike), so these ask about a piece none of whose moves is among the menu's
# first two entries. A pick read off the menu then lands on a knight, and only
# the question says which move "the first one" or "the second one" is.


def fits_sort_late(
    setup: Callable[[EvalApp], None], fits: Callable[[str], bool]
) -> Callable[[EvalApp], None]:
    """`setup`, then #352's premise: two or more moves fit the ask, and neither
    of the first two entries of `legal_moves` is one of them."""

    def check(app: EvalApp) -> None:
        setup(app)
        menu = app.ctx.session.legal_moves()
        assert len([m for m in menu if fits(m)]) >= 2, menu
        assert not any(fits(m) for m in menu[:2]), menu[:2]

    return check


QUEEN_TO_MOVE = fits_sort_late(AFTER_E4_E5, lambda m: m.startswith("Q"))
BISHOP_TO_MOVE = fits_sort_late(AFTER_E4_E5, lambda m: m.startswith("B"))

QUEEN_ASK_ASIDE_THEN_FIRST = Scenario(
    name="queen_ask_aside_then_first",
    tier=2,
    why=(
        "A queen ask is answered 'the first one' after an aside. The queen's "
        "moves sort after the knights' in `legal_moves`, so the pick lands "
        "only if the ordinal is read against the question, not the menu."
    ),
    dev=wordings(
        QUEEN_TO_MOVE,
        difficulty_aside=("move my queen", "what difficulty am I on?", "the first one"),
        voice_aside=(
            "bring my queen out",
            "is voice output on right now?",
            "first one please",
        ),
        queen_please_talk_aside=(
            "queen move, please",
            "how much do you talk by default?",
            "let's go with the first one",
        ),
        move_with_queen=(
            "make a move with my queen",
            "is spoken output enabled?",
            "first of those",
        ),
        queen_move_talk_aside=(
            "play a queen move",
            "are you set to talk out loud?",
            "the first option, please",
        ),
    ),
    heldout=wordings(
        QUEEN_TO_MOVE,
        verbosity_aside=(
            "let's get the queen moving",
            "how chatty are you set to be?",
            "go with the first option",
        ),
        level_aside=(
            "develop my queen",
            "how strong is the engine set right now?",
            "I'll take the first",
        ),
        where_queen_voice_aside=(
            "where can my queen go? move her",
            "is the voice setting on?",
            "the first one you said",
        ),
        move_the_queen_level_aside=(
            "I want to move the queen",
            "what level is the bot on?",
            "first",
        ),
        queen_into_game=(
            "get my queen into the game",
            "what's the difficulty right now?",
            "number one",
        ),
    ),
    checkpoints=(asked(1), still(2), first_offered(1, 3)),
)

ASK_ASIDE_THEN_SECOND = Scenario(
    name="ask_aside_then_second",
    tier=2,
    why=(
        "'The second one', two turns after the ask: the second candidate is a "
        "queen or bishop move, while `legal_moves[1]` is Nf3, so only the "
        "question's own list resolves it."
    ),
    dev=(
        Variant(
            "queen",
            (
                Say("move my queen"),
                Say("what difficulty am I on?"),
                Say("the second one"),
            ),
            QUEEN_TO_MOVE,
        ),
        Variant(
            "bishop",
            (
                Say("move my bishop"),
                Say("is voice output on right now?"),
                Say("second one"),
            ),
            BISHOP_TO_MOVE,
        ),
        Variant(
            "bishop_verbosity_second",
            (
                Say("make a bishop move"),
                Say("what verbosity are you on?"),
                Say("let's take the second one"),
            ),
            BISHOP_TO_MOVE,
        ),
        Variant(
            "queen_verbose_number_two",
            (
                Say("I'd like a queen move"),
                Say("how verbose are you right now?"),
                Say("number two"),
            ),
            QUEEN_TO_MOVE,
        ),
        Variant(
            "bishop_voice_second",
            (
                Say("I want to move a bishop"),
                Say("is voice on?"),
                Say("second of those"),
            ),
            BISHOP_TO_MOVE,
        ),
    ),
    heldout=(
        Variant(
            "queen",
            (
                Say("I want to play a queen move"),
                Say("how chatty are you set to be?"),
                Say("go with the second option"),
            ),
            QUEEN_TO_MOVE,
        ),
        Variant(
            "bishop",
            (
                Say("develop my bishop"),
                Say("how strong is the engine set right now?"),
                Say("I'll take the second"),
            ),
            BISHOP_TO_MOVE,
        ),
        Variant(
            "queen_level_second",
            (
                Say("let's move the queen"),
                Say("what's the engine level?"),
                Say("second option please"),
            ),
            QUEEN_TO_MOVE,
        ),
        Variant(
            "bishop_difficulty_second",
            (
                Say("bring out my bishop"),
                Say("what difficulty is set?"),
                Say("the second option"),
            ),
            BISHOP_TO_MOVE,
        ),
        Variant(
            "queen_spoken_second",
            (
                Say("bring the queen out"),
                Say("are replies spoken aloud right now?"),
                Say("the second one you listed"),
            ),
            QUEEN_TO_MOVE,
        ),
    ),
    checkpoints=(asked(1), still(2), second_offered(1, 3)),
)

QUEEN_ASK_LONG_CHAT_THEN_FIRST = Scenario(
    name="queen_ask_long_chat_then_first",
    tier=2,
    why=(
        "`knight_ask_long_chat_then_pick` with an ask whose candidates sort "
        "late: five turns of chat push Glitch's question out of the verbatim "
        "window, and 'the first one' is neither a square nor `legal_moves[0]`, "
        "so only the kept record (#319) says which move it was."
    ),
    dev=wordings(
        QUEEN_TO_MOVE,
        chat=(
            "bring my queen out",
            "hang on, what's your favourite opening?",
            "why do people like it?",
            "tell me a chess joke",
            "who was the best player ever?",
            "fair enough. okay, where were we",
            "the first one you offered",
        ),
        study_chat=(
            "move my queen",
            "before that, how do I get better at openings?",
            "should I learn the Sicilian?",
            "what's the Italian game?",
            "is it good for beginners?",
            "cool. back to it",
            "the first of the ones you gave me",
        ),
        pawn_rules_chat=(
            "get my queen out",
            "quick question about rules: can a pawn move backwards?",
            "what happens when it reaches the end?",
            "can it become a knight?",
            "neat",
            "back to the game",
            "let's do the first option",
        ),
        champions_chat=(
            "queen move, please",
            "random: who's the current world champion?",
            "how long have they held the title?",
            "who was champion before?",
            "thanks",
            "anyway, back to it",
            "first one you gave me",
        ),
        openings_chat=(
            "let's move the queen",
            "side note: what's the best opening for black?",
            "and for white?",
            "why is that one popular?",
            "do you play it?",
            "alright, back to the board",
            "first option",
        ),
    ),
    heldout=wordings(
        QUEEN_TO_MOVE,
        lesson_chat=(
            "where can my queen go? move it",
            "wait, before that: what does castling actually do?",
            "and when should I do it?",
            "what's a fork?",
            "any tips for a beginner?",
            "thanks. right, back to the game",
            "go with the first option you gave me",
        ),
        history_chat=(
            "let's get the queen moving",
            "quick question, who invented chess?",
            "how old is it?",
            "when did the queen get so strong?",
            "interesting. and castling, when did that start?",
            "okay, let's get on with it",
            "the first option from before",
        ),
        discovered_attack_chat=(
            "time to move my queen",
            "wait, what's a discovered attack?",
            "can you give an example?",
            "how often does it happen?",
            "is it hard to spot?",
            "ok, let's get back to it",
            "the first one",
        ),
        castling_chat=(
            "bring the queen out",
            "before that — how does castling queenside work?",
            "is it riskier than kingside?",
            "when do players choose it?",
            "noted",
            "back to my queen move",
            "the first of them",
        ),
        back_rank_chat=(
            "I'd like to move my queen",
            "hold on, what's a back-rank mate?",
            "how do I prevent it?",
            "what's luft?",
            "good to know",
            "ok, continuing",
            "I'll go with the first one",
        ),
    ),
    checkpoints=(
        asked(1),
        Checkpoint(
            "chat_moved_nothing",
            lambda e: not any(e.turn(i).moved for i in range(2, 7)),
        ),
        first_offered(1, 7),
    ),
)

VERDICT_AS_BLACK_FACING_MATE = Scenario(
    name="verdict_as_black_facing_mate",
    tier=1,
    why=(
        "#320: the verdict from the player's side when the player is Black and "
        "the engine has a forced mate. Until #320 the analysis spoke White's "
        "point of view, and a signed mate distance was one more sign for the "
        "model to flip. The checkpoints are the read; which way the verdict "
        "was said is speech accuracy's advantage class."
    ),
    dev=wordings(
        black_facing_mate,
        am_i_getting_mated="am I getting mated here?",
        how_bad="how bad is it for me right now?",
        hold_or_over="can I still hold this, or is it over?",
        am_i_in_trouble="am I in trouble here?",
        as_bad_as_it_looks="is my position as bad as it looks?",
    ),
    heldout=wordings(
        black_facing_mate,
        am_i_lost="be honest, am I lost?",
        whos_winning="who's winning at this point?",
        whats_the_evaluation="what's the evaluation for me?",
        any_way_i_survive="is there any way I survive this?",
        honest_assessment="give me an honest assessment of my position",
    ),
    checkpoints=(
        Checkpoint("judged_1", lambda e: judged(e.turn(1))),
        still(1),
        completed(1),
    ),
)


def said_backed(family: str) -> Checkpoint:
    """Glitch made a `family` claim and every one he made was backed: the
    scorer's reading (`speech_accuracy`), never the words themselves. For an
    ask no tool answers, so no board or result can grade it."""

    def check(e: Episode) -> bool:
        found = [
            claim
            for record in e.traces
            if (score := score_record(record)) is not None
            for claim in score.claims
            if claim.claim == family
        ]
        return bool(found) and all(claim.backed for claim in found)

    return Checkpoint(f"said_{family}_backed", check)


# 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6: the Ruy Lopez, Morphy Defense (C70).
_MORPHY = ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6")
# 1. e4 e6 2. d4 d5: the French Defense (C00).
_FRENCH = ("e4", "e6", "d4", "d5")
# 1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6: the Najdorf (B90).
_NAJDORF = ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6")
# 1. d4 d5 2. c4 e6: the Queen's Gambit Declined (D30).
_QGD = ("d4", "d5", "c4", "e6")
# #339's six more, each a family the speech reading knows by name, each
# ending with the player (White) to move.
# 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5: the Italian Game, Giuoco Piano (C50).
_ITALIAN = ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
# 1. e4 c6 2. d4 d5: the Caro-Kann Defense (B12).
_CARO_KANN = ("e4", "c6", "d4", "d5")
# 1. e4 d5: the Scandinavian Defense (B01).
_SCANDINAVIAN = ("e4", "d5")
# 1. e4 e5 2. Nf3 Nc6 3. d4 exd4: the Scotch Game (C44).
_SCOTCH = ("e4", "e5", "Nf3", "Nc6", "d4", "exd4")
# 1. d4 f5: the Dutch Defense (A80).
_DUTCH = ("d4", "f5")
# 1. d4 d5 2. c4 c6: the Slav Defense (D10).
_SLAV = ("d4", "d5", "c4", "c6")

NAME_THE_OPENING = Scenario(
    name="name_the_opening",
    tier=1,
    why=(
        "#373: the opening's name. The 12B's memory of opening names is "
        "unreliable; since #373 the state block and the narrator's facts "
        "carry the book's name for the line, and the answer needs no tool. "
        "Graded by the speech class, since nothing else can see it."
    ),
    dev=(
        Variant("what_opening", (Say("what opening is this?"),), after(*_MORPHY)),
        Variant(
            "what_are_we_playing",
            (Say("which opening are we in right now?"),),
            after(*_FRENCH),
        ),
        Variant(
            "name_of_this_line",
            (Say("what's the name of this line?"),),
            after(*_SCANDINAVIAN),
        ),
        Variant(
            "which_opening_on_board",
            (Say("which opening have we got on the board?"),),
            after(*_ITALIAN),
        ),
        Variant(
            "known_opening_called",
            (Say("is this a known opening? what's it called?"),),
            after(*_SCOTCH),
        ),
    ),
    heldout=(
        Variant("name_it", (Say("does this opening have a name?"),), after(*_NAJDORF)),
        Variant(
            "what_is_this_called", (Say("what's this setup called?"),), after(*_QGD)
        ),
        Variant("identify_it", (Say("identify the opening for me"),), after(*_SLAV)),
        Variant(
            "tell_me_the_name",
            (Say("can you tell me the name of this opening?"),),
            after(*_CARO_KANN),
        ),
        Variant(
            "what_did_we_get_into",
            (Say("what opening did we just get into?"),),
            after(*_DUTCH),
        ),
    ),
    checkpoints=(said_backed("opening"), still(1), completed(1)),
)

# Six games before this one: Glitch won three, the player two, one drawn.
# Casual 2–1–0 (Glitch first), advanced 1–1–1. The player has two wins, not
# one, because "once" is a shared hedge the speech reading never reads
# ("once you castle"), so "you've beaten me once" could never be graded.
_PAST_RESULTS = (
    ("opponent", "casual"),
    ("opponent", "casual"),
    ("player", "casual"),
    ("opponent", "advanced"),
    ("player", "advanced"),
    (None, "advanced"),
)


def with_results(app: EvalApp) -> None:
    """A fresh board and a results log of five earlier games, loaded the way
    the app loads one off its save dir at startup."""
    _save_dir(app)
    path = app.ctx.save_dir / RESULTS_FILENAME
    log = ResultsLog(path)
    for index, (winner, level) in enumerate(_PAST_RESULTS):
        log.record(
            f"{index:032x}",
            player_color="white",
            difficulty=level,
            result={"player": "1-0", "opponent": "0-1", None: "1/2-1/2"}[winner],
            winner=winner,
            termination="checkmate" if winner else "stalemate",
            from_setup=False,
        )
    app.ctx.results = ResultsLog.load(path)


RESULTS_SO_FAR = Scenario(
    name="results_so_far",
    tier=1,
    why=(
        "#373: results across games. Nothing recorded a result once a game was "
        "gone; since #373 a results log does, and its tally rides in the state "
        "block and the narrator's facts. The answer needs no tool, so the "
        "speech class grades it."
    ),
    dev=wordings(
        with_results,
        how_many_won="how many games have you won against me?",
        my_record="what's my record against you?",
        overall_tally="what's the overall tally between us?",
        games_lost_to_you="how many games have I lost to you?",
        whos_ahead_overall="who's ahead in our games so far?",
    ),
    heldout=wordings(
        with_results,
        games_played="how many games have we played so far?",
        times_beaten="how many times have I beaten you?",
        games_in_total="how many games have we played in total?",
        my_wins_against_you="how many wins do I have against you?",
        times_you_beat_me="how many times have you beaten me?",
    ),
    checkpoints=(said_backed("results"), still(1), completed(1)),
)


# --- the second brain (#374) -----------------------------------------------------


def gathered_note(index: int, *topics: str) -> Checkpoint:
    """Turn `index` gathered one of `topics` (#451): the note that answers the
    ask is among those the gather step put in front of both phases. Until
    #451 the planner had to call `lookup` and write the query, and this
    checked that lookup's best passage; the gather step searches the player's
    own words, and the narrator reads every note it found, so any place
    counts."""

    def check(e: Episode) -> bool:
        return any(p.get("topic") in topics for p in e.turn(index).gathered)

    return Checkpoint(f"gathered_note_{index}", check)


def nothing_gathered(index: int) -> Checkpoint:
    """Turn `index` gathered no note: the ask was about this game, not the
    notes (#451; until then, that the planner called no `lookup`)."""
    return Checkpoint(f"nothing_gathered_{index}", lambda e: not e.turn(index).gathered)


# The note each variant's ask is answered by, by variant name.
_KNOWLEDGE_ANSWERS = {
    "sicilian_idea": "Sicilian Defense",
    "en_passant": "En passant",
    "bishop_pair": "The bishop pair",
    "first_champion": "Wilhelm Steinitz",
    "zugzwang": "Zugzwang",
    "immortal_game": "The Immortal Game",
    "fianchetto": "Fianchetto",
    "rule_of_the_square": "Rule of the square",
    "fifty_move_rule": "Fifty-move rule",
    "capablanca": "José Raúl Capablanca",
}

KNOWLEDGE_QUESTION = Scenario(
    name="knowledge_question",
    tier=1,
    why=(
        "#374: a question about chess, not about this game. The 12B knows some "
        "of the answer and makes up the rest; since #374 `lookup` answers from "
        "the local notes. Graded on the note the lookup found, never on wording. "
        "Since #451 the gather step searches the player's own words before the "
        "planner runs, and the grade is the note it gathered."
    ),
    dev=(
        Variant(
            "sicilian_idea",
            (Say("what's the main idea behind the Sicilian?"),),
            AFTER_E4_E5,
        ),
        Variant("en_passant", (Say("how does en passant actually work?"),), FRESH),
        Variant(
            "capablanca", (Say("who was Capablanca, and why is he famous?"),), FRESH
        ),
        Variant(
            "rule_of_the_square",
            (Say("how does the rule of the square work in pawn endings?"),),
            FRESH,
        ),
        Variant(
            "fianchetto",
            (Say("what does it mean to fianchetto a bishop?"),),
            AFTER_E4_E5,
        ),
    ),
    heldout=(
        Variant(
            "bishop_pair",
            (Say("why do people say having two bishops is an advantage?"),),
            AFTER_E4_E5,
        ),
        Variant(
            "first_champion",
            (Say("who was the very first world chess champion?"),),
            FRESH,
        ),
        Variant("zugzwang", (Say("what does zugzwang mean?"),), FRESH),
        Variant("immortal_game", (Say("what was the Immortal Game?"),), FRESH),
        Variant("fifty_move_rule", (Say("what's the fifty-move rule?"),), AFTER_E4_E5),
    ),
    checkpoints=(
        Checkpoint(
            "found_the_note",
            lambda e: gathered_note(1, _KNOWLEDGE_ANSWERS.get(e.variant, "")).check(e),
        ),
        still(1),
        completed(1),
    ),
)

# The Ruy Lopez notes a "this opening" ask after the Morphy line can land on.
_RUY_LOPEZ_NOTES = ("Ruy Lopez: Morphy Defense", "Ruy Lopez", "Ruy Lopez: Closed")

THIS_OPENINGS_IDEAS = Scenario(
    name="this_openings_ideas",
    tier=1,
    why=(
        '#374: knowledge about the game on the board. "This opening" names '
        "nothing in the words; the state block names it (#373), and the "
        "planner must carry that name into the lookup. Since #451 there is no "
        "lookup to carry it into: the gather step adds the opening on the board "
        "when its search finds a note about openings in general, and about half "
        "these wordings do."
    ),
    dev=wordings(
        after(*_MORPHY),
        plan_here="what's the usual plan for me in this opening?",
        typical_strategies="what are the typical strategies in this opening?",
        theory_behind="what's the theory behind the opening we're in?",
        aiming_for="what should I be aiming for in this opening?",
        why_people_play_it="why do people play the opening we're playing?",
    ),
    heldout=wordings(
        after(*_MORPHY),
        ideas_of_line="explain the ideas of the opening we're playing",
        main_ideas_behind="what are the main ideas behind this opening?",
        teach_key_plans="teach me the key plans of the opening on the board",
        white_trying_to_do="what's white trying to do in this line?",
        point_for_white="what's the point of this opening for white?",
    ),
    checkpoints=(gathered_note(1, *_RUY_LOPEZ_NOTES), still(1), completed(1)),
)

NOT_A_LOOKUP = Scenario(
    name="not_a_lookup",
    tier=1,
    why=(
        "#374's near misses: the best move here and who is winning are this "
        "game's facts, answered by the engine, not by the notes. A new tool on "
        "the menu must not draw them away. Since #451, nothing may be gathered "
        "for them either: a near-miss note in context is noise."
    ),
    dev=wordings(
        AFTER_E4_E5,
        best_move_here="what's the best move for me in this position?",
        theoretically_best="what's the theoretically best move here?",
        if_you_were_me="if you were me, what would you play now?",
        strongest_continuation="what's the strongest continuation in this position?",
        find_me_a_good_move="can you find me a good move in this position?",
    ),
    heldout=wordings(
        AFTER_E4_E5,
        what_to_play_now="what would a strong player play here?",
        grandmaster_choose="which move would a grandmaster choose here?",
        most_accurate="what's the most accurate move for me here?",
        what_would_magnus_play="what would Magnus play here?",
        best_option_this_move="what's my best option on this move?",
    ),
    checkpoints=(
        Checkpoint("hinted_1", lambda e: bool(e.turn(1).ran("get_best_moves"))),
        nothing_gathered(1),
        still(1),
        completed(1),
    ),
)

KNOWLEDGE_ASIDE_THEN_MOVE = Scenario(
    name="knowledge_aside_then_move",
    tier=2,
    why=(
        "#374 in a thread: a knowledge question mid-game, then a move. The "
        "lookup moves nothing, and the move after it lands as asked. Since "
        "#451 the note is gathered rather than looked up."
    ),
    dev=wordings(
        AFTER_E4_E5_NF3_NC6,
        italian_then_bc4=(
            "quick one: what's the point of the Italian Game?",
            "cool, put my bishop on c4 then",
        ),
        tell_me_then_bishop=(
            "tell me about the Italian Game first",
            "great, let's develop the bishop to c4 then",
        ),
        plan_for_white_then=(
            "what's the plan in the Italian Game for white?",
            "got it. bishop to c4, please",
        ),
        aim_then_play=(
            "what does the Italian Game aim for?",
            "cool, play the bishop to c4",
        ),
        why_good_then_go=(
            "why is the Italian considered a good opening?",
            "nice — I'll go bishop to c4",
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5_NF3_NC6,
        italian_why_then_bc4=(
            "before I move, why do people like the Italian opening?",
            "alright, bishop to c4",
        ),
        idea_then_bishop_c4=(
            "what's the idea of the Italian Game?",
            "ok, then play my bishop to c4",
        ),
        special_then_light_bishop=(
            "what's so special about the Italian Game?",
            "alright then, light bishop to c4",
        ),
        beginners_then_go_for_it=(
            "is the Italian Game good for beginners?",
            "then let's go for it: bishop to c4",
        ),
        explain_quickly_then=(
            "explain the Italian Game to me quickly",
            "thanks — move my bishop to c4",
        ),
    ),
    checkpoints=(
        gathered_note(1, "Italian Game"),
        still(1),
        Checkpoint("played_Bc4", lambda e: e.history(after_turn=2)[4:5] == ["Bc4"]),
        completed(2),
    ),
)

# --- harder scenarios (#339) ----------------------------------------------------------
#
# The solved scenarios above stay: a future model shows on them that it holds
# the line. These nine stand beside them, each one step past a solved task: a
# correction inside the utterance, a takeback conditioned on a verdict, an
# order the utterance sets, a pick by description, the player as Black, three
# settings remembered, a rule explained and then used, a suggestion from before
# a move that was taken back, and a rewind to an event named in words.


def played_in(turn: Turn) -> list[str]:
    """The SANs the turn's successful `make_move` calls played, in order."""
    return [r["result"].get("san") for r in turn.ran("make_move")]


SELF_CORRECTION_MID_UTTERANCE = Scenario(
    name="self_correction_mid_utterance",
    tier=1,
    why=(
        "A correction inside the utterance ('f3, no wait, c3'): the corrected "
        "move lands, the one corrected is never played, and the verdict comes "
        "after the move."
    ),
    dev=wordings(
        AFTER_E4_E5,
        oops_i_mean=(
            "I'll play knight to f3... oops, I mean knight to c3. am I doing well?"
        ),
        no_on_c3_instead=(
            "put the knight on f3, no, on c3 instead, and tell me if that's a good move"
        ),
        nf3_actually_nc3=(
            "play Nf3, actually no, make it Nc3, then tell me if I'm better"
        ),
        wait_no_knight_c3="knight f3, wait no, knight c3, then evaluate the position",
        sorry_i_mean_c3=(
            "move my knight to f3... sorry, I mean c3. am I better after that?"
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5,
        scratch_that_nc3_eval="Nf3 — scratch that, Nc3 — and give me an evaluation",
        b_knight_instead=(
            "develop the g-knight to f3 — no, actually the b-knight "
            "to c3 — and judge it"
        ),
        f3_no_wait_c3="knight to f3 — no wait, c3 — and is that good for me?",
        hmm_no_nc3="let's go Nf3. hmm, no, Nc3. is that strong?",
        double_correction=(
            "knight to c3 — no wait, f3 — no, c3 after all. how am I doing?"
        ),
    ),
    checkpoints=(
        Checkpoint("played_Nc3", lambda e: e.history()[2:3] == ["Nc3"]),
        Checkpoint("never_played_Nf3", lambda e: "Nf3" not in played_in(e.turn(1))),
        Checkpoint(
            "judged_after_the_move", lambda e: verdict_after_the_move(e.turn(1))
        ),
        completed(1),
    ),
)

# 1. e4 e5 2. Qh5 Nc6 3. Qxe5+?? Nxe5: the player's last move gave the queen for
# a pawn, a blunder at any depth. 1. e4 e5 2. Nf3 Nc6: a sound one.
_BLUNDERED = ("e4", "e5", "Qh5", "Nc6", "Qxe5+", "Nxe5")
_SOUND = ("e4", "e5", "Nf3", "Nc6")
AFTER_A_BLUNDER = after(*_BLUNDERED)
AFTER_A_SOUND_MOVE = after(*_SOUND)


def _judged_before_acting(e: Episode) -> bool:
    """A verdict tool ran, and before any takeback: the condition was read
    off a verdict, not guessed."""
    names = e.turn(1).succeeded()
    verdicts = [i for i, name in enumerate(names) if name in VERDICT_TOOLS]
    if not verdicts:
        return False
    return "undo" not in names or verdicts[0] < names.index("undo")


def _acted_on_the_verdict(e: Episode) -> bool:
    """Taken back after the blunder, left alone after the sound move: graded
    on the setup, which no engine depth disagrees with."""
    start = e.start["history"]
    blundered = tuple(start) == _BLUNDERED
    return e.history(after_turn=1) == (start[:-2] if blundered else start)


CONDITIONAL_TAKEBACK = Scenario(
    name="conditional_takeback",
    tier=1,
    why=(
        "A takeback conditioned on a verdict ('if it was a blunder, take it "
        "back'): the move is judged first and undone only if it was one. Half "
        "the wordings follow a blunder and half a sound move, so neither "
        "always undoing nor never undoing passes."
    ),
    dev=(
        Variant(
            "if_i_blundered",
            (
                Say(
                    "if I just blundered, take the move back. if I didn't, "
                    "leave the board alone"
                ),
            ),
            AFTER_A_BLUNDER,
        ),
        Variant(
            "analyse_undo_only_if",
            (Say("analyse the move I just made and undo it only if it's a blunder"),),
            AFTER_A_SOUND_MOVE,
        ),
        Variant(
            "blunder_take_it_back",
            (
                Say(
                    "if that last move of mine was a blunder, take it back; "
                    "otherwise leave it"
                ),
            ),
            AFTER_A_BLUNDER,
        ),
        Variant(
            "check_and_undo_if",
            (Say("check my last move — if it was a blunder, undo it, if not keep it"),),
            AFTER_A_SOUND_MOVE,
        ),
        Variant(
            "did_i_blunder",
            (Say("did I blunder just now? undo it if I did"),),
            AFTER_A_BLUNDER,
        ),
    ),
    heldout=(
        Variant(
            "only_if_engine_says",
            (Say("take back my last move, but only if the engine calls it a blunder"),),
            AFTER_A_SOUND_MOVE,
        ),
        Variant(
            "was_it_a_blunder",
            (Say("was my last move a blunder? if so, take it back"),),
            AFTER_A_BLUNDER,
        ),
        Variant(
            "real_blunder_only",
            (Say("only undo my last move if it was a real blunder"),),
            AFTER_A_SOUND_MOVE,
        ),
        Variant(
            "previous_move_blunder",
            (
                Say(
                    "if my previous move was a blunder, go back; otherwise "
                    "leave things as they are"
                ),
            ),
            AFTER_A_BLUNDER,
        ),
        Variant(
            "rate_and_reverse",
            (Say("rate my last move, and if it's a blunder, reverse it"),),
            AFTER_A_SOUND_MOVE,
        ),
    ),
    checkpoints=(
        Checkpoint("judged_before_acting", _judged_before_acting),
        Checkpoint("acted_on_the_verdict", _acted_on_the_verdict),
        completed(1),
    ),
)

SAVE_BEFORE_MOVE_THEN_JUDGE = Scenario(
    name="save_before_move_then_judge",
    tier=1,
    why=(
        "Four intents in an order the utterance sets: the save comes first, so "
        "it must not hold the move; then the move, a verdict after it, and a "
        "verbosity change that stands."
    ),
    dev=wordings(
        AFTER_E4_E5,
        save_then_nf3_short=(
            "save the game as before_nf3, then play Nf3, tell me if "
            "it's a good move, and keep your replies short from now "
            "on"
        ),
        keep_it_short_now=(
            "from now on keep it short. now: save as before_nf3, "
            "play Nf3, and tell me whether that's good for me"
        ),
        save_first_talk_less=(
            "save first as before_nf3, then Nf3, then the "
            "evaluation — and talk less from now on"
        ),
        before_anything_save=(
            "before you play anything, save as before_nf3; then "
            "play Nf3 and evaluate it. shorter answers from now on "
            "too"
        ),
        less_wordy_only_then=(
            "please be less wordy from now on. also save the game "
            "as before_nf3 and only then play Nf3, then tell me if "
            "it's an improvement"
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5,
        store_before_moving=(
            "store this game as before_nf3 before moving, then "
            "develop my knight to f3 and tell me how good that is; "
            "also, brief answers from now on"
        ),
        brief_first_save=(
            "keep it brief from now on. first save this as "
            "before_nf3, then knight to f3, then tell me if I'm "
            "better"
        ),
        first_make_a_save=(
            "first make a save called before_nf3, then play the "
            "knight to f3, then judge the position, and be terse "
            "from here on"
        ),
        backup_then_g_knight=(
            "save a backup called before_nf3, move the g-knight to "
            "f3, then evaluate — and be concise going forward"
        ),
        shorter_save_verdict=(
            "shorter replies please. save the game under "
            "before_nf3, play Nf3, and give me your verdict"
        ),
    ),
    checkpoints=(
        Checkpoint(
            "saved_before_the_move", lambda e: e.saved("before_nf3") == ["e4", "e5"]
        ),
        Checkpoint("played_Nf3", lambda e: e.history()[2:3] == ["Nf3"]),
        Checkpoint(
            "judged_after_the_move", lambda e: verdict_after_the_move(e.turn(1))
        ),
        Checkpoint("terse", lambda e: e.settings(1)["verbosity"] == "low"),
        completed(1),
    ),
)

# The move each `pick_by_description` wording describes, by name. After 1.e4 e5
# the queen can go to e2, f3, g4 or h5, and each description fits exactly one:
# h5 attacks e5 and is on the edge and furthest up; e2 is in front of the king
# and closest to home; g4 is on the g-file and hits g7 and d7; f3 is on the
# f-file and the third rank.
_DESCRIBED = {
    "attacks_e_pawn": "Qh5",
    "in_front_of_king": "Qe2",
    "on_the_g_file": "Qg4",
    "on_the_f_file": "Qf3",
    "furthest_up": "Qh5",
    "closest_to_home": "Qe2",
    "hits_g_pawn": "Qg4",
    "third_rank": "Qf3",
    "on_the_edge": "Qh5",
    "aims_at_d_pawn": "Qg4",
}

PICK_BY_DESCRIPTION = Scenario(
    name="pick_by_description",
    tier=2,
    why=(
        "A question answered by a description rather than a square or a "
        "place in the list ('the one that attacks your e-pawn'): which move "
        "fits is the model's to work out from the board."
    ),
    dev=wordings(
        QUEEN_TO_MOVE,
        in_front_of_king=("bring my queen out", "the one right in front of my king"),
        third_rank=("develop the queen", "the one on the third rank"),
        aims_at_d_pawn=(
            "I'd like to play a queen move",
            "the one aiming at your d-pawn",
        ),
        furthest_up=(
            "let's move my queen — where can she go?",
            "the one that goes furthest up the board",
        ),
        closest_to_home=("make a queen move", "the one that stays closest to home"),
    ),
    heldout=wordings(
        QUEEN_TO_MOVE,
        on_the_f_file=("queen move please", "put her on the f-file"),
        on_the_g_file=("I want to move the queen", "the one on the g-file"),
        on_the_edge=("get my queen moving", "the one out on the edge of the board"),
        hits_g_pawn=("play my queen", "the one that hits your g-pawn"),
        attacks_e_pawn=("move my queen", "the one that attacks your e-pawn"),
    ),
    checkpoints=(
        asked(1),
        Checkpoint(
            "played_the_described",
            lambda e: e.history(after_turn=2)[2:3] == [_DESCRIBED[e.variant]],
        ),
    ),
)


def as_black_after(*sans: str) -> Callable[[EvalApp], None]:
    """The player as Black, with the moves placed through the session (the
    engine, White, made the first), ending on the player's turn."""

    def setup(app: EvalApp) -> None:
        _save_dir(app)
        app.ctx.replace_session(GameSession(player_color="black"), app.ctx.transcript)
        for san in sans:
            assert app.ctx.session.submit_move(san).legal, san
        assert app.ctx.session.turn == "black"

    return setup


# 1. e4 e5 2. Nf3 Nc6 3. Bc4, the player Black and to move.
AS_BLACK_AFTER_BC4 = as_black_after("e4", "e5", "Nf3", "Nc6", "Bc4")


UNDO_CHAIN_AS_BLACK = Scenario(
    name="undo_chain_as_black",
    tier=2,
    why=(
        "undo_chain_across_turns with the player as Black: 'my last move' is "
        "Black's, the reply in front of it White's, and each takeback lands on "
        "the player's turn."
    ),
    dev=wordings(
        AS_BLACK_AFTER_BC4,
        previous_one_more=("undo my previous move", "undo one more", "then play d5"),
        rewind_rewind=(
            "rewind my last move",
            "rewind one more",
            "push my d-pawn up two",
        ),
        knight_then_e_pawn=(
            "take back my knight move",
            "and take back my e-pawn move too",
            "now push the d-pawn two squares",
        ),
        scratch_scratch=(
            "scratch my last move",
            "scratch the one before as well",
            "now d5 for me",
        ),
        undo_by_name=("undo Nc6", "and undo e5 as well", "now play d5 instead"),
    ),
    heldout=wordings(
        AS_BLACK_AFTER_BC4,
        let_me_take_back=(
            "let me take back my last move",
            "actually one more as well",
            "play the queen's pawn two squares",
        ),
        go_back_another=("go back a move", "go back another one", "play d5 now"),
        can_you_take_back=(
            "can you take back my last move?",
            "and the move before that too, please",
            "then go d5",
        ),
        id_like_to_undo=(
            "I'd like to undo my last move",
            "undo the previous one too",
            "and then d5 please",
        ),
        last_and_before=(
            "take back my last move",
            "and the one before it",
            "now play d5",
        ),
    ),
    checkpoints=(
        Checkpoint(
            "first_takeback", lambda e: e.history(after_turn=1) == ["e4", "e5", "Nf3"]
        ),
        Checkpoint("second_takeback", lambda e: e.history(after_turn=2) == ["e4"]),
        Checkpoint("played_d5", lambda e: e.history()[:2] == ["e4", "d5"]),
        Checkpoint("exactly_one_more_exchange", lambda e: len(e.history()) == 3),
    ),
)

SETTINGS_RESTORE_ALL = Scenario(
    name="settings_restore_all",
    tier=2,
    why=(
        "Three settings changed over three turns, then all of them put back "
        "in one ask: what each was at the start is only in the record of what "
        "changed."
    ),
    dev=wordings(
        FRESH,
        harder_voice_name_all=(
            "make it harder",
            "turn voice on",
            "talk less from now on",
            "put the difficulty, voice and verbosity back to how they started",
        ),
        tougher_aloud_original=(
            "I want a tougher opponent",
            "talk to me out loud",
            "be brief from now on",
            "restore all my original settings",
        ),
        tougher_voice_revert=(
            "tougher engine please",
            "voice on, please",
            "make your replies shorter",
            "revert all the settings to how they were originally",
        ),
        step_speak_undo_changes=(
            "step the difficulty up",
            "speak your answers out loud",
            "keep it short from now on",
            "undo all my settings changes",
        ),
        crank_speaking_starting=(
            "crank the difficulty up one step",
            "start speaking your replies",
            "brief answers from here",
            "return every setting to its starting value",
        ),
    ),
    heldout=wordings(
        FRESH,
        bump_read_started_with=(
            "bump the strength up a bit",
            "read your replies aloud",
            "terse answers from here on",
            "go back to the settings we started this game with",
        ),
        raise_audio_beginning=(
            "raise the level by one",
            "enable audio replies",
            "less wordy please",
            "change everything back to how it was at the beginning",
        ),
        notch_switch_reset=(
            "increase the difficulty a notch",
            "switch voice output on",
            "shorter replies from now on",
            "reset every setting to what it was at the start of our chat",
        ),
        stronger_voice_defaults=(
            "can you play a little stronger?",
            "I'd like voice output on",
            "please be more concise",
            "set everything back to the defaults we had at the start",
        ),
        harder_voice_short_back=(
            "make the engine a bit harder",
            "turn on spoken replies",
            "keep your answers short",
            "now put all my settings back the way they were when we started",
        ),
    ),
    checkpoints=(
        Checkpoint(
            "harder_1", lambda e: strength(e.settings(1)) > strength(e.settings(0))
        ),
        Checkpoint("voice_on_2", lambda e: e.settings(2)["voice_output"] is True),
        Checkpoint("terse_3", lambda e: e.settings(3)["verbosity"] == "low"),
        Checkpoint(
            "difficulty_restored",
            lambda e: strength(e.settings(4)) == strength(e.settings(0)),
        ),
        Checkpoint(
            "voice_restored",
            lambda e: e.settings(4)["voice_output"] == e.settings(0)["voice_output"],
        ),
        Checkpoint(
            "verbosity_restored",
            lambda e: e.settings(4)["verbosity"] == e.settings(0)["verbosity"],
        ),
        Checkpoint("no_moves", lambda e: e.history() == []),
    ),
)

# 1. e4 Nf6 2. e5 d5: Black's d-pawn has just passed e5, so exd6 en passant is
# the player's for this move only.
_EN_PASSANT_ON = ("e4", "Nf6", "e5", "d5")


def en_passant_on(app: EvalApp) -> None:
    after(*_EN_PASSANT_ON)(app)
    assert "exd6" in app.ctx.session.legal_moves()


EN_PASSANT_EXPLAINED_THEN_TAKEN = Scenario(
    name="en_passant_explained_then_taken",
    tier=2,
    why=(
        "A rule asked about on a board where it applies, then used: the "
        "gathered note explains it (a lookup until #451) and nothing moves, "
        "and 'do it' is the en passant capture, legal on this move only."
    ),
    dev=wordings(
        en_passant_on,
        use_it_here=(
            "how does en passant work? can I use it in this position?",
            "ok, capture en passant",
        ),
        can_i_do_it_here=("what's en passant, and can I do it here?", "do it then"),
        legal_for_me=(
            "en passant: what is it, and is it legal for me now?",
            "ok, take it",
        ),
        capture_right_now=(
            "can I capture en passant right now? explain the rule",
            "do the en passant capture",
        ),
        remind_me=(
            "remind me how en passant works and whether I can play it",
            "cool, play the en passant",
        ),
    ),
    heldout=wordings(
        en_passant_on,
        available_to_me=(
            "is en passant available to me here? what is it exactly?",
            "alright, play it",
        ),
        possible_right_now=(
            "explain en passant — is it possible for me right now?",
            "great, take en passant",
        ),
        heard_about_it=(
            (
                "I heard about en passant. can I do it in this "
                "position, and how does it work?"
            ),
            "sure, make that capture",
        ),
        do_i_have_it=(
            "what's this en passant thing, and do I have it now?",
            "then go for it",
        ),
        does_it_apply=(
            "tell me about the en passant rule. does it apply on this board?",
            "nice, let's make that capture",
        ),
    ),
    checkpoints=(
        gathered_note(1, "En passant"),
        still(1),
        Checkpoint(
            "took_en_passant", lambda e: e.history(after_turn=2)[4:5] == ["exd6"]
        ),
        completed(2),
    ),
)

FIRST_SUGGESTION_AFTER_ALL = Scenario(
    name="first_suggestion_after_all",
    tier=2,
    why=(
        "Two suggestions on two boards with a move between them, then 'take "
        "that back and play what you suggested before it': only the "
        "conversation says which suggestion, and its board is back only after "
        "the takeback."
    ),
    dev=wordings(
        AFTER_E4_E5,
        originally_recommended=(
            "which move do you recommend?",
            Say("a3", model=False),
            "and which do you recommend now?",
            "take back a3 and play what you originally recommended",
        ),
        recommended_before_it=(
            "what's your top move here?",
            Say("a3", model=False),
            "and your top move now?",
            "never mind. undo a3 and play the move you recommended before it",
        ),
        right_the_first_time=(
            "what's the best move here?",
            Say("a3", model=False),
            "and what's best now?",
            (
                "you were right the first time: take my a3 back and "
                "play the move you suggested before it"
            ),
        ),
        regret_a3=(
            "best move in this position?",
            Say("a3", model=False),
            "and in this one?",
            "I regret a3. take it back and play your original suggestion",
        ),
        before_my_a3=(
            "suggest a move for me",
            Say("a3", model=False),
            "now what do you suggest?",
            "go back to before my a3 and play what you suggested there",
        ),
    ),
    heldout=wordings(
        AFTER_E4_E5,
        original_after_all=(
            "what should I play here?",
            Say("a3", model=False),
            "and what should I play now?",
            "take back a3; I'll take your original suggestion after all",
        ),
        before_i_played_a3=(
            "got a move for me here?",
            Say("a3", model=False),
            "ok, what's the best move now?",
            "undo my last move and go with the move you suggested before I played a3",
        ),
        strongest_from_before=(
            "what's the strongest move here?",
            Say("a3", model=False),
            "and what's strongest now?",
            "undo my a3 and play the strongest move from before it",
        ),
        should_have_listened=(
            "what would you play here?",
            Say("a3", model=False),
            "what would you play now?",
            "I should have listened. undo a3 and play what you suggested before it",
        ),
        stockfish_liked=(
            "what does Stockfish like here?",
            Say("a3", model=False),
            "and which move does it like now?",
            "undo a3 and play the move Stockfish liked before it",
        ),
    ),
    checkpoints=(
        Checkpoint("consulted_1", lambda e: suggested(e.turn(1)) is not None),
        still(1),
        Checkpoint("consulted_3", lambda e: suggested(e.turn(3)) is not None),
        still(3),
        Checkpoint(
            "took_a3_back",
            lambda e: (
                e.history(after_turn=4)[:2] == ["e4", "e5"]
                and e.history(after_turn=4)[2:3] != ["a3"]
            ),
        ),
        Checkpoint(
            "played_the_first_suggestion",
            lambda e: (
                suggested(e.turn(1)) is not None
                and e.history(after_turn=4)[2:3] == [suggested(e.turn(1))]
            ),
        ),
        Checkpoint("exactly_one_exchange", lambda e: len(e.history(after_turn=4)) == 4),
    ),
)

# The ply each `rewind_to_an_event` wording names, by name: in the 84-ply game
# the player's first capture is 11. fxg5 (ply 20) and first queen move 12. Qb3
# (ply 22). `late_84_events` asserts both.
_FIRST_CAPTURE, _FIRST_QUEEN_MOVE = 20, 22
_EVENTS = {
    "before_first_queen_move": _FIRST_QUEEN_MOVE,
    "queens_first_move": _FIRST_QUEEN_MOVE,
    "queen_came_out": _FIRST_QUEEN_MOVE,
    "undo_to_first_queen_move": _FIRST_QUEEN_MOVE,
    "before_i_moved_the_queen": _FIRST_QUEEN_MOVE,
    "before_first_capture": _FIRST_CAPTURE,
    "captured_anything": _FIRST_CAPTURE,
    "first_capture_of_the_game": _FIRST_CAPTURE,
    "undo_to_first_capture": _FIRST_CAPTURE,
    "set_back_first_capture": _FIRST_CAPTURE,
}


def late_84_events(app: EvalApp) -> None:
    LATE_84(app)
    mine = app.ctx.session.move_history()[::2]
    first_capture = next(i for i, san in enumerate(mine) if "x" in san)
    first_queen = next(i for i, san in enumerate(mine) if san.startswith("Q"))
    assert (2 * first_capture, 2 * first_queen) == (_FIRST_CAPTURE, _FIRST_QUEEN_MOVE)


def _back_before_the_event(e: Episode) -> bool:
    ply = _EVENTS[e.variant]
    return e.history(after_turn=1) == e.start["history"][:ply]


def _played_best_at_the_event(e: Episode) -> bool:
    ply = _EVENTS[e.variant]
    best = suggested(e.turn(2))
    history = e.history(after_turn=3)
    return (
        best is not None
        and history[:ply] == e.start["history"][:ply]
        and history[ply : ply + 1] == [best]
    )


REWIND_TO_AN_EVENT = Scenario(
    name="rewind_to_an_event",
    tier=3,
    why=(
        "An 84-ply game rewound to an event named in words ('before my queen "
        "first moved', 'before my first capture'): the ply is the model's to "
        "find in the move list, and the best move there is then played."
    ),
    dev=wordings(
        late_84_events,
        set_back_first_capture=(
            "set the board back to right before my first capture",
            "what would you have played?",
            "play your move",
        ),
        queen_came_out=(
            "go back to before my queen came out the first time",
            "what does the engine suggest there?",
            "make that move",
        ),
        first_capture_of_the_game=(
            "go back to before my first capture of the game",
            "what does Stockfish recommend there?",
            "play its recommendation",
        ),
        before_first_queen_move=(
            "take me back to just before I first moved my queen",
            "what should I have played there instead?",
            "play that",
        ),
        captured_anything=(
            "rewind to right before I captured anything for the first time",
            "what was best there?",
            "play that one",
        ),
    ),
    heldout=wordings(
        late_84_events,
        undo_to_first_queen_move=(
            "undo everything back to just before my first queen move",
            "best move in that position?",
            "play the best move",
        ),
        undo_to_first_capture=(
            "undo back to just before I captured for the first time",
            "what's the best move in that spot?",
            "make it",
        ),
        before_i_moved_the_queen=(
            "return the game to the moment before I moved my queen for the first time",
            "what's the strongest move there?",
            "go with that",
        ),
        before_first_capture=(
            "take the game back to just before my first capture",
            "what should I have played instead?",
            "play that move",
        ),
        queens_first_move=(
            "rewind to the position right before my queen's first move",
            "what was the best move at that point?",
            "play it",
        ),
    ),
    checkpoints=(
        Checkpoint("back_before_the_event", _back_before_the_event),
        Checkpoint("consulted", lambda e: suggested(e.turn(2)) is not None),
        still(2),
        Checkpoint("played_the_best_there", _played_best_at_the_event),
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
    KNIGHT_ASK_ASIDE_THEN_PICK,
    KNIGHT_ASK_LONG_CHAT_THEN_PICK,
    KNIGHT_ASK_THEN_BOARD_CHANGES,
    TWO_THREADS_SIMILAR_ASKS,
    QUEEN_ASK_ASIDE_THEN_FIRST,
    ASK_ASIDE_THEN_SECOND,
    QUEEN_ASK_LONG_CHAT_THEN_FIRST,
    VERDICT_AS_BLACK_FACING_MATE,
    NAME_THE_OPENING,
    RESULTS_SO_FAR,
    KNOWLEDGE_QUESTION,
    THIS_OPENINGS_IDEAS,
    NOT_A_LOOKUP,
    KNOWLEDGE_ASIDE_THEN_MOVE,
    SELF_CORRECTION_MID_UTTERANCE,
    CONDITIONAL_TAKEBACK,
    SAVE_BEFORE_MOVE_THEN_JUDGE,
    PICK_BY_DESCRIPTION,
    UNDO_CHAIN_AS_BLACK,
    SETTINGS_RESTORE_ALL,
    EN_PASSANT_EXPLAINED_THEN_TAKEN,
    FIRST_SUGGESTION_AFTER_ALL,
    REWIND_TO_AN_EVENT,
)
