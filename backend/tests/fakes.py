"""Shared test doubles.

`ScriptedBrain` is the one canonical no-LLM `Brain` for the whole suite: it
pops canned responses in order and records what it was shown, so the agent
loop can be exercised deterministically without ever reaching a live model.
`FakeEngine` is the canonical no-Stockfish engine double: it plays a scripted
reply and records every strength setting and MultiPV request, so tests can
pin that difficulty reaches the engine and that replies never take an
analysis detour. Keep the doubles here — not copied into test files.
"""

from chessapp.brain import AgentResponse


class FakeEngine:
    """Engine double: scripted reply + recorders for strength and MultiPV."""

    def __init__(self, reply_uci: str = "e7e5"):
        self.reply_uci = reply_uci
        self.multipv_requests: list[int] = []
        self.skill_levels: list[int] = []
        self.elos: list[int] = []

    def play_move(self, session):
        return session.submit_move(self.reply_uci)

    def choose_move(self, session):
        return self.reply_uci

    def get_best_moves(self, session, n=3):
        self.multipv_requests.append(n)
        return []

    def set_skill_level(self, level: int) -> None:
        self.skill_levels.append(level)

    def set_elo(self, elo: int) -> None:
        self.elos.append(elo)


class ScriptedBrain:
    """Scripted brain: pops canned responses in order, records prompts.

    Phase one (`get_agent_response`) pops the next scripted `AgentResponse`;
    under-scripting raises `IndexError` so a test that asks for more turns than
    it planned fails loudly instead of silently reusing stale output.

    Phase two (`react`) — the game loop's react-from-new-state step — pops the
    next scripted reaction text and records the new board plus what changed.
    When no reaction is scripted it returns a placeholder, so tests that don't
    assert on reaction text don't have to script one.
    """

    def __init__(
        self, *responses: AgentResponse, reactions: tuple[str, ...] = ()
    ) -> None:
        self._responses = list(responses)
        self._reactions = list(reactions)
        self.calls: list[tuple[dict, str]] = []
        self.react_calls: list[tuple[dict, list]] = []

    def get_agent_response(self, board_state: dict, command: str) -> AgentResponse:
        self.calls.append((board_state, command))
        return self._responses.pop(0)

    def react(self, board_state: dict, changes: list) -> str:
        self.react_calls.append((board_state, changes))
        return self._reactions.pop(0) if self._reactions else "(reaction)"
