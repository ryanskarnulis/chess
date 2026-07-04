"""Shared test doubles.

`ScriptedBrain` is the one canonical no-LLM `Brain` for the whole suite: it
pops canned responses in order and records what it was shown, so the agent
loop can be exercised deterministically without ever reaching a live model.
Keep the brain double here — not copied into individual test files.
"""

from chessapp.brain import AgentResponse


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
