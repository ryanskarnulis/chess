"""The opening's name, read off the move line by code (#373)."""

import chess

from chessapp import openings
from chessapp.api import TurnCoordinator, _agent_state_dict, narrator_facts
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import FakeEngine


def _played(*sans: str, **kwargs) -> GameSession:
    session = GameSession(**kwargs)
    for san in sans:
        assert session.submit_move(san).legal, san
    return session


def test_the_book_is_the_vendored_table():
    book = openings.book()
    assert len(book.positions) > 3000
    assert book.max_ply >= 30
    assert "Ruy Lopez" in openings.families()
    assert "Sicilian Defense" in openings.families()


def test_the_ruy_lopez_is_named_with_its_eco_code():
    session = _played("e4", "e5", "Nf3", "Nc6", "Bb5")
    assert openings.opening_of(session) == {"eco": "C60", "name": "Ruy Lopez"}


def test_the_deepest_position_names_the_variation():
    session = _played("e4", "e5", "Nf3", "Nc6", "Bb5", "a6")
    assert openings.opening_of(session) == {
        "eco": "C70",
        "name": "Ruy Lopez: Morphy Defense",
    }


def test_a_transposition_reaches_the_same_name():
    session = _played("Nf3", "Nc6", "e4", "e5", "Bb5")
    assert openings.opening_of(session) == {"eco": "C60", "name": "Ruy Lopez"}


def test_no_move_no_opening():
    assert openings.opening_of(GameSession()) is None
    assert openings.names_on_line(GameSession()) == frozenset()


def test_a_set_up_position_has_no_opening():
    # The Ruy Lopez position itself, set up rather than played to, then one
    # book move on from it.
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):
        board.push_san(san)
    session = GameSession(fen=board.fen())
    assert session.submit_move("a6").legal
    assert openings.opening_of(session) is None


def test_the_name_stops_updating_once_the_line_leaves_the_book(monkeypatch):
    """Nothing past the table's longest line is looked up, so a middlegame
    that happens to meet a book position again keeps the name it had. Shown
    on a table cut to five plies: the Morphy Defense, at ply six, is past
    it."""
    full = openings.book()
    monkeypatch.setattr(
        openings, "book", lambda: openings.Book(full.positions, max_ply=5)
    )
    session = _played("e4", "e5", "Nf3", "Nc6", "Bb5", "a6")
    assert openings.opening_of(session) == {"eco": "C60", "name": "Ruy Lopez"}


def test_a_takeback_takes_the_name_back_too():
    session = _played("e4", "e5", "Nf3", "Nc6", "Bb5", "a6")
    session.undo(1)
    assert openings.opening_of(session) == {"eco": "C60", "name": "Ruy Lopez"}


def test_the_line_keeps_every_name_it_passed_through():
    session = _played("e4", "e5", "Nf3", "Nc6", "Bb5", "a6")
    names = openings.names_on_line(session)
    assert "Ruy Lopez" in names
    assert "Ruy Lopez: Morphy Defense" in names
    assert "King's Pawn Game" in names


def test_both_phases_are_shown_the_opening():
    ctx = ToolContext(
        session=_played("e4", "e5", "Nf3", "Nc6", "Bb5"), engine=FakeEngine()
    )
    expected = {"eco": "C60", "name": "Ruy Lopez"}
    assert _agent_state_dict(ctx)["opening"] == expected
    assert narrator_facts(ctx, TurnCoordinator(ctx))["opening"] == expected
