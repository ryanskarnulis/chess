"""Speech accuracy, scored offline from trace records (#367).

The claims are `honesty.claims` and the facts `facts.assemble`, both specified
elsewhere; what this file pins is the counting — what a record is scored on,
which claims count, and which are kept out of the accuracy rather than held
against the model.
"""

import chess
import pytest

from chessapp.facts import TurnEvidence, settings_of
from chessapp.game import GameSession
from chessapp.speech_accuracy import (
    LEGACY_SCORED,
    UNSCORED,
    merge,
    names_reply,
    score_record,
    tally,
    unbacked,
)
from chessapp.trace import turn_record


def _session(*sans: str, player_color: str = "white") -> GameSession:
    session = GameSession(player_color=player_color)
    for san in sans:
        assert session.submit_move(san).legal
    return session


def _record(
    draft: str,
    session: GameSession,
    *,
    tools=(),
    reply: str | None = None,
    fen_before: str = chess.STARTING_FEN,
    pending: str | None = None,
    route: str = "brain",
    **fields,
) -> dict:
    tool_results = [{"name": name, "result": result} for name, result in tools]
    evidence = TurnEvidence(
        session=session.to_dict(),
        settings=settings_of(False, "normal", "casual"),
        tool_results=tool_results,
        engine_reply_san=reply,
        fen_before=fen_before,
        pending_reply_fen=pending,
        results=fields.pop("results", None),
    )
    return {
        "ts": "2026-09-26T12:00:00",
        **turn_record(
            utterance="play",
            route=route,
            commentary=draft,
            stop_reason="completed",
            changed=True,
            turn_id=1,
            correlation_id="c0ffee",
            mutations=1,
            fen_before=fen_before,
            fen_after=session.fen(),
            tool_calls=[{} for _ in tool_results],
            tool_results=tool_results,
            draft=draft,
            evidence=evidence.as_trace(),
            **fields,
        ),
    }


def _counts(score) -> list[tuple[str, bool]]:
    return [(claim.claim, claim.backed) for claim in score.claims]


# --- a current record: the draft against the evidence ---------------------------


def test_a_true_capture_and_a_false_one_are_one_backed_and_one_not():
    session = _session("e4", "d5", "exd5")

    score = score_record(
        _record("You took my pawn. You took my queen.", session, reply=None)
    )

    assert not score.legacy
    assert ("capture", False) in _counts(score)
    assert ("capture", True) in _counts(score)


def test_the_facts_are_the_players_side_of_the_session():
    """What only the evidence can say: the player is black, so the pawn that
    left the board was the player's to lose, and Glitch the one who took it."""
    session = _session("e4", "d5", "exd5", player_color="black")

    assert _counts(score_record(_record("I took your pawn.", session))) == [
        ("capture", True)
    ]
    assert _counts(score_record(_record("You took my pawn.", session))) == [
        ("capture", False)
    ]


def test_an_unplayed_reply_counts_only_while_a_reply_was_owed():
    session = _session("e4")
    owed = session.fen()

    named = score_record(_record("My turn. Nf6.", session, pending=owed))
    settled = score_record(_record("My turn. Nf6.", session, pending=None))

    assert ("unplayed_reply", False) in _counts(named)
    assert "unplayed_reply" not in dict(_counts(settled))


def test_a_reply_named_after_it_was_played_is_judged_against_the_real_one():
    """#365: the narrator speaks after the reply and is handed it, so the
    class applies to every turn with a reply — the right name is backed, any
    other move the engine could have played is not."""
    session = _session("e4", "e5")

    right = score_record(_record("I answer e5.", session, reply="e5"))
    wrong = score_record(_record("I answer Nc6.", session, reply="e5"))

    assert ("unplayed_reply", False) not in _counts(right)
    assert ("unplayed_reply", False) in _counts(wrong)


@pytest.mark.parametrize(
    ("draft", "reply", "said"),
    [
        ("Nf6, your move.", "Nf6", True),
        ("Knight to f6.", "Nf6", True),
        ("I take on d5 with the pawn.", "exd5", True),
        ("Mate. Qh4.", "Qh4#", True),
        ("I castle.", "O-O", True),
        ("Short castle, easy.", "O-O-O", True),
        ("Nice move.", "Nf6", False),
        ("I like f5 here.", "Nf6", False),
        ("Nf61 is not a square.", "Nf6", False),
    ],
)
def test_names_reply_reads_san_spoken_squares_and_castles(draft, reply, said):
    assert names_reply(draft, reply) is said


def test_a_turn_that_owed_the_reply_in_words_is_counted():
    session = _session("e4", "e5")
    said = _record("Mirror. e5.", session, reply="e5")
    silent = _record("Mirror.", session, reply="e5")
    before = _record("Mirror.", session, reply="e5", pending=_session("e4").fen())
    no_reply = _record("Mirror.", session, reply=None)

    assert score_record(said).reply_announced is True
    assert score_record(silent).reply_announced is False
    assert score_record(before).reply_announced is None, "spoken before it"
    assert score_record(no_reply).reply_announced is None

    run = tally([said, silent, before, no_reply])
    assert (run.replies_owed, run.replies_announced) == (2, 1)
    assert run.as_dict()["replies"] == {"owed": 2, "announced": 1}
    assert "reply_said=1/2" in run.summary()
    merged = merge([run.as_dict(), {**run.as_dict(), "replies": None}])
    assert merged["replies"] == {"owed": 2, "announced": 1}


def test_a_winner_claim_on_a_live_board_is_the_ending_classes_alone():
    score = score_record(_record("Checkmate, you win.", _session("e4")))

    assert ("ending", False) in _counts(score)
    assert "outcome" not in dict(_counts(score))


def test_the_winner_is_scored_once_the_game_is_over():
    mated = _session("f3", "e5", "g4", "Qh4#")

    wrong = score_record(_record("You win.", mated))
    right = score_record(_record("I win.", mated))

    assert ("outcome", False) in _counts(wrong)
    assert ("outcome", True) in _counts(right)


def test_a_turn_that_never_reached_the_guard_is_not_scored():
    record = _record("Anything.", _session())
    record["evidence"] = None

    assert score_record(record) is None
    assert score_record({"kind": "serving", "schema": 3}) is None


# --- an older record: what it can decide, and nothing it cannot -----------------


def _legacy(commentary: str, **fields) -> dict:
    record = {
        "kind": "turn",
        "schema": 2,
        "route": "brain",
        "model_calls": 2,
        "commentary": commentary,
        "guarded": False,
        "suppressed": "",
        "fen_before": chess.STARTING_FEN,
        "fen_after": chess.STARTING_FEN,
        "engine_reply": None,
        "tools": [],
    }
    record.update(fields)
    return record


def test_a_legacy_record_scores_only_the_families_it_can_decide():
    score = score_record(_legacy("Alright, more detail from now on. Took your queen."))

    assert score.legacy
    assert ("verbosity_change", False) in _counts(score)
    assert "capture" in score.unscored, "no history to back a capture with"
    assert "verbosity_change" not in score.unscored
    assert LEGACY_SCORED.isdisjoint(set(score.unscored) - set(UNSCORED) - {"outcome"})


def test_a_legacy_draft_is_the_suppressed_one_when_the_guard_fired():
    score = score_record(
        _legacy("e5.", guarded=True, suppressed="Game over, bro.", model_calls=3)
    )

    assert _counts(score) == [("ending", False)]


def test_a_legacy_draft_leaves_the_apps_reply_line_out():
    """The reply announcement is the app's words; "Game over: 0-1" there is
    true and was never Glitch's to be scored on."""
    score = score_record(
        _legacy(
            "Nice try.\n\nQh4#. Game over: 0-1 (checkmate).",
            engine_reply={"san": "Qh4#", "uci": "d8h4"},
            outcome={"winner": "opponent", "termination": "checkmate"},
        )
    )

    assert score.claims == ()


def test_a_legacy_turn_the_model_never_spoke_on_is_not_scored():
    assert score_record(_legacy("e4. e5.", model_calls=0)) is None
    assert score_record(_legacy("Resigned.", route="resign")) is None


# --- the tally ------------------------------------------------------------------


def test_the_tally_keeps_the_unscored_out_of_the_accuracy():
    session = _session("e4", "d5", "exd5")
    records = [
        _record("You took my pawn.", session),
        _record("You took my queen.", session),
        _legacy("Took your rook. More detail from now on, promise."),
        {"kind": "speech", "schema": 3},
    ]

    result = tally(records)
    summary = result.as_dict()

    assert (summary["turns"], summary["legacy_turns"]) == (3, 1)
    assert summary["families"]["capture"] == {"made": 2, "backed": 1}
    assert summary["families"]["verbosity_change"] == {"made": 1, "backed": 0}
    assert summary["unscored"]["capture"]["made"] == 1
    assert (summary["made"], summary["backed"]) == (3, 1)
    assert summary["accuracy"] == round(1 / 3, 4)
    assert [u.family for u in result.unbacked] == ["capture", "verbosity_change"]
    assert result.unbacked[0].sentence == "You took my queen."


def test_no_claims_is_no_accuracy_rather_than_a_perfect_one():
    assert tally([_record("Nice.", _session())]).as_dict()["accuracy"] is None


def test_summaries_merge_into_the_run_they_came_from():
    session = _session("e4", "d5", "exd5")
    one = tally([_record("You took my pawn.", session)]).as_dict()
    two = tally([_record("You took my queen.", session)]).as_dict()

    assert (
        merge([one, None, two])
        == tally(
            [
                _record("You took my pawn.", session),
                _record("You took my queen.", session),
            ]
        ).as_dict()
    )


def test_a_partial_record_is_skipped_rather_than_scored():
    """The harnesses' tracers can hold hand-built records with only the fields
    a test needed; scoring must never be what breaks them."""
    assert score_record({"kind": "turn", "route": "brain", "model_calls": 1}) is None
    partial = _legacy("Game over.")
    del partial["fen_after"]
    assert score_record(partial) is None


# --- where the scorer backs more than the guard (#367's first frontier run) -----


# The late_game_review_undo_replay fixture's worst white move, as review_game
# reported it on 2026-09-26: d3, 816 centipawns, Bxc4 was best.
_REVIEW = (
    "review_game",
    {
        "ok": True,
        "critical": [
            {
                "move_number": 7,
                "color": "white",
                "san": "d3",
                "classification": "blunder",
                "cp_loss": 816,
                "best": "Bxc4",
            }
        ],
    },
)


def test_a_reviews_best_move_and_a_rounded_count_are_backed():
    """Both of the run's unbacked lines, verbatim: true, and scored so."""
    session = _session("e4")

    score = score_record(
        _record(
            "You dropped like 800 centipawns there\u2014Bxc4 was the move. "
            "lost like 800 centipawns on that one.",
            session,
            tools=[_REVIEW],
        )
    )

    assert sorted(_counts(score)) == [
        ("evaluation", True),
        ("evaluation", True),
        ("move", True),
    ]


def test_a_count_no_rounding_reaches_is_still_unbacked():
    session = _session("e4")

    score = score_record(
        _record("That cost you 300 centipawns. Qh5 was best.", session, tools=[_REVIEW])
    )

    assert ("evaluation", False) in _counts(score)
    assert ("move", False) in _counts(score)


def test_the_frontier_runs_misread_lines_are_scored_as_true():
    """#384: three true lines the #340 frontier run scored unbacked, verbatim.
    A conditional, a review's lost evaluation and a counterfactual are not an
    ending or a move that was played."""
    session = _session("e4")
    name, review = _REVIEW
    review = {**review, "critical": [{**review["critical"][0], "cp_loss": 817}]}

    score = score_record(
        _record(
            "Yo, starting a new game will end this one. "
            "you lost like 817 centipawns there. "
            "you shoulda played Bxc4",
            session,
            tools=[(name, review)],
        )
    )

    assert all(backed for _, backed in _counts(score))
    assert not {"ending", "owned_move"} & {family for family, _ in _counts(score)}


# --- one turn's misses, for the eval gate and pipeline tests (#368) -------------


def test_unbacked_is_the_scored_misses_and_nothing_else():
    session = _session("e4", "d5", "exd5")

    record = _record("You took my pawn. You took my queen.", session)

    assert [(c.claim, c.sentence) for c in unbacked(record)] == [
        ("capture", "You took my queen.")
    ]
    assert unbacked(_record("You took my pawn.", session)) == ()


def test_unbacked_leaves_out_what_the_scorer_does_not_score():
    """An unscored family is the scorer's doubt, not the model's miss, so it
    never fails a sample either: an older record cannot decide a capture."""
    record = _legacy("Took your queen. Alright, more detail from now on.")

    assert [c.claim for c in unbacked(record)] == ["verbosity_change"]


def test_a_record_with_no_words_to_judge_has_nothing_unbacked():
    assert unbacked({"kind": "serving", "schema": 5}) == ()


# --- #320: who is better, from the player's side --------------------------------


def test_a_direction_is_scored_only_on_a_turn_that_asked_the_engine():
    session = _session("e4", "e5")
    assert _counts(score_record(_record("You're winning.", session))) == []

    evaluated = ("evaluate_position", {"ok": True, "player_advantage_cp": -180})
    score = score_record(_record("You're winning.", session, tools=[evaluated]))
    assert _counts(score) == [("advantage", False)]


def test_a_pre_320_white_pov_score_is_turned_to_the_players_side():
    """The bug #320 is for, re-judged from an old record: White-POV +180 with
    the player on Black, and Glitch telling them they're ahead."""
    session = _session("e4", "e5", player_color="black")
    legacy = ("evaluate_position", {"ok": True, "score_cp": 180, "mate_in": None})
    told_ahead = score_record(_record("You're up 1.8, easy.", session, tools=[legacy]))
    assert ("advantage", False) in _counts(told_ahead)
    told_behind = score_record(_record("I'm ahead.", session, tools=[legacy]))
    assert _counts(told_behind) == [("advantage", True)]


# --- #373: the opening, re-derived from the record's own session ----------------


def test_an_opening_is_scored_against_the_line_the_record_holds():
    session = _session("e4", "e5", "Nf3", "Nc6", "Bb5")
    backed = score_record(_record("Classic Ruy Lopez.", session))
    assert _counts(backed) == [("opening", True)]
    wrong = score_record(_record("Textbook Italian Game.", session))
    assert _counts(wrong) == [("opening", False)]


def test_a_system_named_inside_another_family_is_backed():
    """The deployed trace's "the London's here": the book files 1. d4 d5
    2. Bf4 under the Queen's Pawn Game, as its Accelerated London System."""
    session = _session("d4", "d5", "Bf4")
    assert _counts(score_record(_record("Going for the London, I see.", session))) == [
        ("opening", True)
    ]


# --- #373: the results tally, from the record's evidence ------------------------

_TALLY = {
    "games": 2,
    "player_won": 0,
    "engine_won": 2,
    "drawn": 0,
    "by_difficulty": {
        "casual": {"games": 2, "player_won": 0, "engine_won": 2, "drawn": 0}
    },
}


def test_a_count_is_scored_against_the_tally_the_turn_was_shown():
    session = _session("e4", "e5")
    backed = score_record(_record("I've won 2 games.", session, results=_TALLY))
    assert _counts(backed) == [("results", True)]
    wrong = score_record(_record("You've won 2 games.", session, results=_TALLY))
    assert [
        c.claim
        for c in unbacked(_record("You've won 2 games.", session, results=_TALLY))
    ] == ["results"]
    assert _counts(wrong) == [("results", False)]


def test_a_record_with_no_tally_leaves_a_count_unscored():
    record = _record("I've won 2 games.", _session("e4", "e5"))
    assert "results" in score_record(record).unscored
    assert unbacked(record) == ()


# --- what a lookup said (#374) ----------------------------------------------------

_LOOKUP = (
    "lookup",
    {
        "ok": True,
        "passages": [
            {
                "topic": "Ruy Lopez",
                "text": "1. e4 e5 2. Nf3 Nc6 3. Bb5. White plans c3 and d4, often "
                "with the knight travelling Nb1-d2-f1-g3.",
            }
        ],
    },
)


def test_a_lookups_moves_and_opening_are_backed():
    """Glitch quoting the notes on a turn that looked them up is quoting a
    fact, in a game that is in another opening (here, none at all)."""
    session = _session("d4")

    score = score_record(
        _record(
            "The Ruy Lopez is all about Bb5, pressuring the knight.",
            session,
            tools=[_LOOKUP],
        )
    )

    assert sorted(_counts(score)) == [("move", True), ("opening", True)]


def test_without_the_lookup_the_same_words_are_unbacked():
    session = _session("d4")

    score = score_record(
        _record("The Ruy Lopez is all about Bb5, pressuring the knight.", session)
    )

    assert sorted(_counts(score)) == [("move", False), ("opening", False)]


def test_a_move_the_notes_never_named_is_still_unbacked():
    session = _session("d4")

    score = score_record(
        _record("In the Ruy Lopez, Qh5 is the main idea.", session, tools=[_LOOKUP])
    )

    assert ("move", False) in _counts(score)
    assert ("opening", True) in _counts(score)
