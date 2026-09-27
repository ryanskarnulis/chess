"""Speech accuracy, scored offline from trace records (#367).

The claims are `honesty.claims` and the facts `facts.assemble`, both specified
elsewhere; what this file pins is the counting — what a record is scored on,
which claims count, and which are kept out of the accuracy rather than held
against the model.
"""

import chess

from chessapp.facts import TurnEvidence, settings_of
from chessapp.game import GameSession
from chessapp.speech_accuracy import (
    LEGACY_SCORED,
    UNSCORED,
    merge,
    score_record,
    tally,
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
