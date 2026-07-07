"""Shared test doubles.

`ScriptedBrain` is the one canonical no-LLM `Brain` for the whole suite: it
pops canned responses in order and records what it was shown, so the agent
loop can be exercised deterministically without ever reaching a live model.
`StyleAwareFakeEngine` is the canonical no-Stockfish engine double for the
styled-reply path. Keep the doubles here — not copied into test files.
"""

from chessapp.brain import AgentResponse


class StyleAwareFakeEngine:
    """Engine double for the styled-reply path: best move differs from the
    MultiPV pick a biased personality would make, so a test can tell which
    path the code under test took."""

    def __init__(self):
        self.multipv_requests = []

    def play_move(self, session):
        return session.submit_move("e7e5")  # the plain "best move" path

    def choose_move(self, session):
        return "e7e5"

    def get_best_moves(self, session, n=3):
        from chessapp.engine import CandidateMove

        self.multipv_requests.append(n)
        # Best-first, White-POV scores; the weakest eligible (modest pick,
        # i.e. best for White) is a7a6.
        return [
            CandidateMove(uci="e7e5", san="e5", score_cp=-30, mate_in=None),
            CandidateMove(uci="a7a6", san="a6", score_cp=60, mate_in=None),
        ][:n]


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
