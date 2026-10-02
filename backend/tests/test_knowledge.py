"""The second brain's retrieval and the `lookup` tool (#374).

Retrieval is deterministic, so it is tested here the way the rest of the core
is: a table of questions players ask, each with the note that answers it,
and a table of asks no note should answer. A wording that stops finding its
note is a ranking regression, caught without a model in the loop.

The corpus is tested for shape too: every file names its sources, every note
is short enough to read out, and every opening note agrees with the Lichess
book the app names openings from — its title is a book name, and the moves it
opens with reach a position the book files under the same family.
"""

import re

import chess
import pytest

from chessapp import handoff, knowledge, openings
from chessapp.game import GameSession
from chessapp.tools import RETRY_DIFFERENT_ARGS, ToolContext, build_registry

INDEX = knowledge.chess_knowledge()

# What a player asks, as the planner would pass it on, and the note that
# answers it: the first hit, always.
ANSWERS = [
    # openings
    ("what's the idea behind the Sicilian", "openings/sicilian-defense"),
    ("Sicilian Najdorf", "openings/sicilian-defense-najdorf-variation"),
    ("Najdorf plans", "openings/sicilian-defense-najdorf-variation"),
    ("Sicilian Dragon Yugoslav attack", "openings/sicilian-defense-dragon-variation"),
    ("Sveshnikov", "openings/sicilian-defense-lasker-pelikan-variation"),
    ("Ruy Lopez: Morphy Defense", "openings/ruy-lopez-morphy-defense"),
    ("plans in the Ruy Lopez Morphy Defense", "openings/ruy-lopez-morphy-defense"),
    ("Spanish opening", "openings/ruy-lopez"),
    ("Berlin wall", "openings/ruy-lopez-berlin-defense"),
    ("Marshall Attack", "openings/ruy-lopez-marshall-attack"),
    ("Italian Game ideas", "openings/italian-game"),
    ("Giuoco Piano", "openings/italian-game"),
    (
        "Fried Liver Attack",
        "openings/italian-game-two-knights-defense-fried-liver-attack",
    ),
    ("Evans Gambit", "openings/italian-game-evans-gambit"),
    ("French Defence main ideas", "openings/french-defense"),
    ("French Winawer", "openings/french-defense-winawer-variation"),
    ("Advance French", "openings/french-defense-advance-variation"),
    ("Caro-Kann", "openings/caro-kann-defense"),
    ("Caro Kann advance variation", "openings/caro-kann-defense-advance-variation"),
    ("Scandinavian defense", "openings/scandinavian-defense"),
    ("how to play against the London System", "openings/london-system"),
    ("Queen's Gambit", "openings/queen-gambit"),
    ("queens gambit declined", "openings/queen-gambit-declined"),
    ("Queen's Gambit Accepted", "openings/queen-gambit-accepted"),
    ("Slav defense", "openings/slav-defense"),
    ("King's Indian Defense plans", "openings/king-indian-defense"),
    ("Nimzo-Indian", "openings/nimzo-indian-defense"),
    ("Grunfeld", "openings/grunfeld-defense"),
    ("Dutch Stonewall", "openings/dutch-defense"),
    ("Catalan", "openings/catalan-opening"),
    ("is the King's Gambit any good", "openings/king-gambit"),
    ("English opening", "openings/english-opening"),
    ("Bongcloud", "openings/bongcloud-attack"),
    ("Benko Gambit", "openings/benko-gambit"),
    ("Alekhine's defence", "openings/alekhine-defense"),
    ("Pirc", "openings/pirc-defense"),
    ("Smith-Morra gambit", "openings/sicilian-defense-smith-morra-gambit"),
    ("what opening should a beginner play", "strategy/choosing-an-opening"),
    ("what is a gambit", "strategy/gambits"),
    # strategy
    ("why is the bishop pair good", "strategy/the-bishop-pair"),
    ("opening principles", "strategy/opening-principles"),
    ("why castle early", "strategy/king-safety"),
    ("isolated queen's pawn", "strategy/isolated-queen-pawn"),
    ("what is an isolated pawn", "strategy/isolated-queen-pawn"),
    ("doubled pawns", "strategy/doubled-pawns"),
    ("passed pawn", "strategy/passed-pawn"),
    ("good bishop bad bishop", "strategy/good-and-bad-bishops"),
    ("knight on the rim is dim", "strategy/knights"),
    ("how much is a rook worth", "strategy/material-values"),
    ("rook on the seventh rank", "strategy/rook-on-the-seventh"),
    ("what does fianchetto mean", "strategy/fianchetto"),
    ("hypermodern", "strategy/hypermodernism"),
    ("prophylaxis", "strategy/prophylaxis"),
    ("when should I trade pieces", "strategy/when-to-trade-pieces"),
    ("how do I stop blundering", "strategy/blunder-check"),
    ("minority attack", "strategy/minority-attack"),
    ("outpost for a knight", "strategy/outposts"),
    ("how to improve at chess", "strategy/how-to-improve-at-chess"),
    # tactics
    ("what's a fork", "tactics/fork"),
    ("absolute pin", "tactics/pin"),
    ("skewer", "tactics/skewer"),
    ("discovered check", "tactics/discovered-check"),
    ("zwischenzug", "tactics/zwischenzug"),
    ("smothered mate", "tactics/smothered-mate"),
    ("back rank mate", "tactics/back-rank-mate"),
    ("Greek gift sacrifice", "tactics/greek-gift-sacrifice"),
    ("scholar's mate", "tactics/scholar-mate"),
    ("fastest checkmate", "tactics/fool-mate"),
    ("Legal's mate", "tactics/legal-mate"),
    ("Elephant trap", "tactics/elephant-trap"),
    ("underpromotion to a knight", "tactics/underpromotion"),
    # endgames
    ("opposition in king and pawn endings", "endgames/opposition"),
    ("how to checkmate with king and rook", "endgames/checkmate-with-king-and-rook"),
    ("checkmate with a queen", "endgames/checkmate-with-king-and-queen"),
    ("bishop and knight checkmate", "endgames/checkmate-with-bishop-and-knight"),
    ("can two knights checkmate", "endgames/two-knights-cannot-force-mate"),
    ("Lucena position", "endgames/lucena-position"),
    ("Philidor position", "endgames/philidor-position"),
    ("explain zugzwang", "endgames/zugzwang"),
    ("rule of the square", "endgames/rule-of-the-square"),
    ("triangulation", "endgames/triangulation"),
    ("opposite coloured bishops endgame", "endgames/opposite-coloured-bishop-endgames"),
    ("tablebases", "endgames/endgame-tablebases"),
    ("wrong rook pawn", "endgames/wrong-rook-pawn"),
    # rules
    ("how does en passant work", "rules/en-passant"),
    ("can I castle out of check", "rules/castling-edge-cases"),
    ("castling rules", "rules/castling"),
    ("what is the fifty move rule", "rules/fifty-move-rule"),
    ("50 move rule", "rules/fifty-move-rule"),
    ("threefold repetition", "rules/threefold-repetition"),
    ("what is stalemate", "rules/stalemate"),
    ("insufficient material", "rules/insufficient-material"),
    ("how do knights move", "rules/how-the-knight-moves"),
    ("can a pawn move backwards", "rules/how-the-pawn-moves"),
    ("can I have two queens", "rules/pawn-promotion"),
    ("touch move rule", "rules/touch-move"),
    ("what is an Elo rating", "rules/chess-ratings"),
    ("how do you become a grandmaster", "rules/chess-titles"),
    ("Chess960", "rules/chess960"),
    ("what is blitz", "rules/time-controls"),
    ("how to read chess notation", "rules/algebraic-notation"),
    # history
    ("who invented chess", "history/origins-of-chess"),
    ("how old is chess", "history/origins-of-chess"),
    ("when did the queen get so strong", "history/modern-chess-and-the-mad-queen"),
    ("when did castling start", "history/history-of-castling"),
    ("who was Capablanca", "history/jose-raul-capablanca"),
    ("tell me about Bobby Fischer", "history/bobby-fischer"),
    ("who is the current world champion", "history/current-world-champion"),
    ("Magnus Carlsen", "history/magnus-carlsen"),
    ("Deep Blue", "history/deep-blue"),
    ("the Immortal Game", "history/the-immortal-game"),
    ("Opera game", "history/the-opera-game"),
    ("first world chess champion", "history/wilhelm-steinitz"),
    ("strongest female chess player", "history/women-in-chess"),
    ("AlphaZero", "history/modern-chess-engines"),
    ("Stockfish", "history/modern-chess-engines"),
    # terms
    ("what does en prise mean", "terms/en-prise"),
    ("ECO codes", "terms/eco-codes"),
    ("what is a simul", "terms/simultaneous-exhibition"),
    ("bughouse", "terms/chess-boxing-and-other-variants"),
]

# Asks that are not about chess knowledge at all: nothing should come back.
NOTHING = [
    "pizza recipe",
    "what time is it",
    "how are you today",
    "nice move",
    "let's play",
    "hello",
    "thanks",
    "do it",
    "weather in Paris",
]


@pytest.mark.parametrize(("query", "expected"), ANSWERS)
def test_the_note_that_answers_comes_first(query, expected):
    hits = INDEX.search(query)
    assert hits, f"nothing found for {query!r}"
    assert hits[0].note.id == expected, [hit.note.id for hit in hits]


@pytest.mark.parametrize("query", NOTHING)
def test_asks_no_note_answers_find_nothing(query):
    assert INDEX.search(query) == []


def test_every_expected_note_exists():
    ids = {note.id for note in INDEX.notes}
    assert {expected for _, expected in ANSWERS} <= ids


def test_a_lookup_returns_at_most_three_notes_best_first():
    hits = INDEX.search("Sicilian Defense")
    assert 1 <= len(hits) <= knowledge.MAX_PASSAGES
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(score >= scores[0] * knowledge.RELATIVE_CUTOFF for score in scores)


def test_a_longer_name_said_beats_its_family():
    # "Sicilian Najdorf" names both notes; the longer, rarer name wins.
    first, second = INDEX.search("Sicilian Najdorf")[:2]
    assert first.note.id == "openings/sicilian-defense-najdorf-variation"
    assert second.note.id == "openings/sicilian-defense"


def test_spelling_and_case_do_not_matter():
    for query in ("GRÜNFELD", "grunfeld", "Grünfeld Defence"):
        assert INDEX.search(query)[0].note.id == "openings/grunfeld-defense"


def test_tokens_drop_fillers_and_meet_on_stems():
    assert knowledge.tokens("um what's the idea, uh?") == ["idea"]
    assert knowledge.tokens("castling") == knowledge.tokens("castle")
    assert knowledge.tokens("pinned") == knowledge.tokens("pins")
    assert knowledge.tokens("1. e4 e5") == ["e4", "e5"]


# --- the corpus ------------------------------------------------------------------


def _files() -> list[tuple[str, str]]:
    return knowledge._files()


def test_every_file_names_its_sources():
    for topic, text in _files():
        preamble = text.split("\n## ", 1)[0]
        assert "Sources" in preamble, f"{topic}.md says nothing of its sources"


def test_every_note_has_a_unique_id_and_readable_length():
    ids = [note.id for note in INDEX.notes]
    assert len(ids) == len(set(ids)), "two notes share a title in one file"
    for note in INDEX.notes:
        words = len(note.text.split())
        assert 20 <= words <= 130, f"{note.id} is {words} words"


def test_the_corpus_is_broad():
    topics = {note.id.split("/")[0] for note in INDEX.notes}
    assert topics >= {"openings", "strategy", "tactics", "endgames", "rules", "history"}
    assert len(INDEX.notes) >= 300


def test_opening_titles_are_book_names():
    """An opening note's title is the name the app reads off the board
    (`openings.book`), so a planner that copies the opening from its state
    block names the note word for word."""
    names = {name for _, name in openings.book().positions.values()}
    for note in INDEX.notes:
        if note.id.startswith("openings/"):
            assert note.title in names or note.title in openings.families(), note.title


_MOVE = re.compile(r"^(?:\d+\.(?:\.\.)?)?([KQRBNOa-h][^\s]*?)[?!]*$")


def _leading_moves(text: str) -> list[str]:
    """The SAN moves a note opens with: "1. e4 e5 2. Nf3 Nc6 3. Bb5." → the
    five moves. Move numbers are skipped; the run ends at the first word that
    is not a move."""
    moves = []
    for word in text.split():
        word = word.rstrip(".,;:")
        if re.fullmatch(r"\d+\.?(?:\.\.)?", word):
            continue
        match = _MOVE.match(word)
        if not match or not re.search(r"\d|O-O", match.group(1)):
            break
        moves.append(match.group(1))
    return moves


def test_opening_lines_are_legal_and_reach_the_named_opening():
    """Every opening note that opens with its moves gives legal moves, and
    the book files the position they reach under the note's own family."""
    checked = 0
    for note in INDEX.notes:
        if not note.id.startswith("openings/") or not note.text.startswith("1."):
            continue
        session = GameSession()
        for san in _leading_moves(note.text):
            assert session.submit_move(san).legal, f"{note.id}: {san}"
        reached = openings.names_on_line(session)
        family = note.title.split(":")[0]
        assert any(name.split(":")[0] == family for name in reached), (
            f"{note.id}: the line reaches {sorted(reached)}"
        )
        checked += 1
    assert checked >= 50


def test_every_move_line_in_the_corpus_parses():
    """Any run of numbered moves anywhere in a note starts from the opening
    position and is legal: "1. e4 e5 2. Bc4 ..." in a trap note, too."""
    for note in INDEX.notes:
        for match in re.finditer(r"(?<![\w.])1\. [^()]*", note.text):
            moves = _leading_moves(match.group(0))
            board = chess.Board()
            for san in moves:
                board.push_san(san)


# --- the tool --------------------------------------------------------------------


@pytest.fixture
def registry():
    return build_registry(ToolContext(session=GameSession()))


def test_lookup_returns_the_passages(registry):
    result = registry.dispatch("lookup", {"query": "how does en passant work"})
    assert result["ok"] is True
    first = result["passages"][0]
    assert first["topic"] == "En passant"
    assert "fifth rank" in first["text"]
    assert set(first) == {"topic", "text"}


def test_lookup_with_no_match_says_nothing_was_found(registry):
    result = registry.dispatch("lookup", {"query": "pizza recipe"})
    assert result == {
        "ok": True,
        "passages": [],
        "summary": "Nothing in the chess notes matches that.",
    }


@pytest.mark.parametrize("query", ["what is it?", "  ", "um, uh"])
def test_a_lookup_that_names_nothing_is_refused(registry, query):
    result = registry.dispatch("lookup", {"query": query})
    assert result["ok"] is False
    assert result["retry"] == RETRY_DIFFERENT_ARGS


def test_an_empty_or_long_query_fails_the_schema(registry):
    assert registry.dispatch("lookup", {"query": ""})["ok"] is False
    assert registry.dispatch("lookup", {"query": "x" * 201})["ok"] is False
    assert registry.dispatch("lookup", {})["ok"] is False


def test_lookup_moves_nothing(registry):
    ctx = registry.context
    before = (ctx.session.fen(), ctx.board_version)
    registry.dispatch("lookup", {"query": "Sicilian"})
    assert (ctx.session.fen(), ctx.board_version) == before


def test_the_handoff_reads_a_lookup_as_consulted():
    result = {"ok": True, "passages": [{"topic": "Fork", "text": "..."}]}
    built = handoff.build([{"name": "lookup", "result": result}])
    assert built.kind == "completed"
    assert [entry.tool for entry in built.consulted] == ["lookup"]
    assert not built.performed
