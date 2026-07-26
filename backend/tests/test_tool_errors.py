"""Structured tool errors: a refusal carries what the agent needs to recover.

Audit item 14. Before this, every "no" the tool layer produced was a bare
sentence — the agent had to infer from prose whether calling again could
possibly help, and an illegal move told it nothing about what *would* have
worked. This file is the spec for the three things every refusal now carries:

- `retry`, one of `RETRY_DIFFERENT_ARGS` / `RETRY_NEVER` — literally "is trying
  again worth a round trip?", answered by code rather than guessed from wording.
- `board_version`, so a refusal says which board it was about (the same counter
  the HTTP precondition uses).
- recovery data: legal `alternatives` for a rejected move, the disambiguated
  candidates for an ambiguous one, the saves that exist for a bad save name.

The confirm gate already worked this way — its refusal carries the exact line to
relay — and these tests pin that pattern spread across the tool surface.
"""

import pytest

from chessapp.coordinator import TurnCoordinator
from chessapp.game import ALTERNATIVES_MAX, GameSession
from chessapp.tools import (
    RETRY_DIFFERENT_ARGS,
    RETRY_NEVER,
    Settings,
    ToolContext,
    ToolRegistry,
    build_registry,
)
from fakes import FakeEngine

# Knights on b1 and f3 with d2 vacated: "Nd2" is ambiguous, "Nbd2"/"Nfd2" are
# the two answers (the queen, bishop and king reach d2 too — they are not
# candidates for a move that said "knight").
TWO_KNIGHTS_TO_D2 = "rnbqkbnr/pppppppp/8/8/8/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1"


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        session=GameSession(),
        engine=FakeEngine(),
        save_dir=tmp_path,
        settings=Settings(),
    )


@pytest.fixture
def registry(ctx):
    return build_registry(ctx)


class TestRejectedMove:
    """An illegal or ambiguous move is a *result* (`ok: True, legal: False`) —
    that has not changed. What it now carries is enough to fix the request."""

    def test_illegal_move_offers_legal_alternatives_for_the_square(self, registry):
        result = registry.dispatch("make_move", {"move": "e2e5"})

        assert result["ok"] is True
        assert result["legal"] is False
        # The player meant e5; the legal way to reach it from the start is e4's
        # follow-up, so nothing does — but the pawn's own moves are the near
        # miss worth naming, not the whole opening book.
        assert result["retry"] == RETRY_DIFFERENT_ARGS
        assert "e5" not in result["alternatives"]
        assert result["alternatives"]  # never an empty gesture

    def test_illegal_move_alternatives_are_moves_to_the_named_square(self, registry):
        # No bishop reaches f3 from the starting position. The piece the request
        # named has nothing there, so the *square* it named is what the
        # alternatives are about — every legal way to reach f3.
        result = registry.dispatch("make_move", {"move": "Bf3"})

        assert result["legal"] is False
        assert result["retry"] == RETRY_DIFFERENT_ARGS
        assert sorted(result["alternatives"]) == ["Nf3", "f3"]

    def test_ambiguous_move_returns_the_candidates(self, ctx, registry):
        ctx.session = GameSession(TWO_KNIGHTS_TO_D2)

        result = registry.dispatch("make_move", {"move": "Nd2"})

        assert result["legal"] is False
        assert "ambiguous" in result["reason"]
        assert sorted(result["alternatives"]) == ["Nbd2", "Nfd2"]
        assert result["retry"] == RETRY_DIFFERENT_ARGS

    def test_alternatives_are_capped(self, registry):
        # Nothing about the string narrows it, so the fallback is the position's
        # own moves — a suggestion list, deliberately not the complete set.
        result = registry.dispatch("make_move", {"move": "banana"})

        assert result["legal"] is False
        assert 0 < len(result["alternatives"]) <= ALTERNATIVES_MAX

    def test_alternatives_are_all_legal(self, registry):
        result = registry.dispatch("make_move", {"move": "Qh5xf7"})

        legal = set(registry.dispatch("get_legal_moves", {})["moves"])
        assert set(result["alternatives"]) <= legal

    def test_rejected_move_carries_the_board_version(self, ctx, registry):
        before = ctx.board_version

        result = registry.dispatch("make_move", {"move": "e2e5"})

        # The board it was about, and the board it left behind: a refusal does
        # not move the version, so the agent's next call can use this one.
        assert result["board_version"] == before == ctx.board_version

    def test_game_over_move_says_do_not_retry(self, ctx, registry):
        ctx.session.resign("white")

        result = registry.dispatch("make_move", {"move": "e4"})

        assert result["legal"] is False
        assert result["retry"] == RETRY_NEVER
        assert result["alternatives"] == []


class TestDispatchErrors:
    """The registry's own two refusals — before any handler runs."""

    def test_unknown_tool_is_not_retriable(self, registry):
        result = registry.dispatch("teleport_king", {})

        assert result["ok"] is False
        assert result["retry"] == RETRY_NEVER

    def test_invalid_args_invite_a_corrected_call(self, registry):
        result = registry.dispatch("make_move", {"mv": "e4"})

        assert result["ok"] is False
        assert result["retry"] == RETRY_DIFFERENT_ARGS

    def test_errors_carry_the_board_version(self, ctx, registry):
        for result in (
            registry.dispatch("teleport_king", {}),
            registry.dispatch("make_move", {"mv": "e4"}),
        ):
            assert result["board_version"] == ctx.board_version

    def test_a_contextless_registry_still_answers(self):
        # `ToolRegistry()` with nothing bound to it — the shape unit tests and
        # the schema fixture build. No board, so no version, and no crash.
        result = ToolRegistry().dispatch("make_move", {"move": "e4"})

        assert result["ok"] is False
        assert result["retry"] == RETRY_NEVER
        assert "board_version" not in result


class TestDomainRefusals:
    def test_confirmation_refusal_keeps_its_relay_line_and_says_do_not_retry(
        self, ctx, registry
    ):
        registry.dispatch("make_move", {"move": "e4"})

        result = registry.dispatch("new_game", {})

        assert result["ok"] is False
        # The line to relay, unchanged — the pattern this slice spreads.
        assert "confirmation required" in result["error"]
        assert result["retry"] == RETRY_NEVER
        assert ctx.pending is not None

    def test_spent_destructive_budget_says_do_not_retry(self, ctx):
        coordinator = TurnCoordinator(ctx)
        registry = build_registry(ctx, coordinator)
        # A command window is what budgets destructive ops — the pipeline opens
        # one per user interaction, and the second op inside it is refused.
        coordinator.begin_command()
        assert registry.dispatch("new_game", {})["ok"] is True

        result = registry.dispatch("new_game", {})

        assert result["ok"] is False
        assert result["retry"] == RETRY_NEVER

    def test_unknown_save_name_lists_the_saves_that_exist(self, ctx, registry):
        registry.dispatch("save_game", {"name": "midgame"})

        result = registry.dispatch("resume_game", {"name": "endgame"})

        assert result["ok"] is False
        assert result["retry"] == RETRY_DIFFERENT_ARGS
        assert result["saved_games"] == ["midgame"]

    def test_saving_without_a_save_dir_is_not_retriable(self, ctx):
        ctx.save_dir = None
        registry = build_registry(ctx)

        result = registry.dispatch("save_game", {"name": "anywhere"})

        assert result["ok"] is False
        assert result["retry"] == RETRY_NEVER

    def test_missing_engine_is_not_retriable(self, ctx):
        ctx.engine = None
        registry = build_registry(ctx)

        result = registry.dispatch("evaluate_position", {})

        assert result["ok"] is False
        assert result["retry"] == RETRY_NEVER

    def test_nothing_to_undo_is_not_retriable(self, registry):
        result = registry.dispatch("undo", {})

        assert result["ok"] is False
        assert result["retry"] == RETRY_NEVER

    def test_bad_difficulty_names_the_tiers_that_work(self, registry):
        # No `ToolError` needed here: the tiers are a schema `enum`, so the
        # candidates are already in the validator's own message. What the
        # refusal adds is that a corrected call is worth making.
        result = registry.dispatch("set_difficulty", {"tier": "impossible"})

        assert result["ok"] is False
        assert result["retry"] == RETRY_DIFFERENT_ARGS
        assert "beginner" in result["error"]


class TestTurnStateRefusals:
    def test_a_second_player_move_mid_turn_is_not_retriable(self, ctx):
        # The split registry leaves the turn open after the player's move, so a
        # second one is the phase machine's "no" — a `TurnStateError`.
        registry = build_registry(ctx, atomic_exchange=False)
        assert registry.dispatch("make_move", {"move": "e4"})["legal"] is True

        result = registry.dispatch("make_move", {"move": "d4"})

        assert result["ok"] is False
        assert result["retry"] == RETRY_NEVER
        assert result["board_version"] == ctx.board_version
