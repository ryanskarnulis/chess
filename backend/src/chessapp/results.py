"""The results log: every game that ended, across games and restarts (#373).

"How many games have you won?" had nothing behind it. The saves can't answer
it: `autosave` is overwritten, and a game nobody saved leaves no file. So code
appends one line to `results.jsonl`, beside `live.json`, whenever the ledger
sees a game end, and the tally of those lines rides in the state block and the
narrator's facts (`api._agent_state_dict`, `api.narrator_facts`).

A takeback of a finished game **withdraws** its result: the board no longer
holds that ending, so neither does the tally. The log stays append-only — a
withdrawal is one more line — and the latest line for a `game_id` is the one
that counts, so a game mated, taken back and mated again is one result.

Fed from the ledger's events rather than hooked into the endings: every road
to an ending already passes through `Ledger.observe` (#372), and a new one
would be a silent gap here too (`ToolContext.follow_results`). An ending the
ledger saw only because a game arrived already over — a finished save
resumed, a rebuilt checkpoint — is marked `restored` there and never counted
again.

Best-effort like the settings file and the live checkpoint: a line that does
not parse is skipped with a warning, and a disk that will not take a write
costs a warning, never the move that ended the game. Without a save dir the
log lives in memory only (tests, the eval harness).
"""

import json
import logging
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "results.jsonl"

# How a result's `winner` is counted: the player's side of it.
_COUNTS = {"player": "player_won", "opponent": "engine_won", None: "drawn"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _entry(data: Any) -> dict[str, Any]:
    """A log line checked for the fields the tally reads; ValueError if not."""
    if not isinstance(data, dict) or not isinstance(data.get("game_id"), str):
        raise ValueError("not a results line")
    if data.get("withdrawn") is True:
        return data
    if data.get("winner") not in _COUNTS or not isinstance(data.get("difficulty"), str):
        raise ValueError("not a results line")
    return data


class ResultsLog:
    """The log's lines and the standing result of each game, latest first
    wins. `path` None keeps it in memory."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._standing: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: Path) -> "ResultsLog":
        log = cls(path)
        try:
            text = path.read_text()
        except FileNotFoundError:
            return log
        except OSError:
            logger.warning("could not read the results log %s", path)
            return log
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                log._apply(_entry(json.loads(line)))
            except ValueError:
                logger.warning("skipping results line %d of %s", number, path)
        return log

    # --- writing ---------------------------------------------------------------

    def record(
        self,
        game_id: str,
        *,
        player_color: str,
        difficulty: str,
        result: str,
        winner: str | None,
        termination: str,
        from_setup: bool,
    ) -> None:
        """A game ended: its result stands until a later line says otherwise."""
        self._append(
            {
                "game_id": game_id,
                "ended_at": _now(),
                "player_color": player_color,
                "difficulty": difficulty,
                "result": result,
                "winner": winner,
                "termination": termination,
                "from_setup": from_setup,
            }
        )

    def withdraw(self, game_id: str) -> None:
        """The ending was taken back. A no-op for a game with no standing
        result, so a takeback in a live game writes nothing."""
        if self.standing(game_id) is None:
            return
        self._append({"game_id": game_id, "withdrawn": True, "at": _now()})

    def _append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._apply(entry)
            if self._path is None:
                return
            try:
                with self._path.open("a") as out:
                    out.write(json.dumps(entry) + "\n")
            except OSError:
                logger.warning("could not write the results log %s", self._path)

    def _apply(self, entry: dict[str, Any]) -> None:
        if entry.get("withdrawn") is True:
            self._standing.pop(entry["game_id"], None)
        else:
            self._standing[entry["game_id"]] = entry

    # --- reading ---------------------------------------------------------------

    def standing(self, game_id: str) -> dict[str, Any] | None:
        return self._standing.get(game_id)

    def tally(self) -> dict[str, Any]:
        """Wins, losses and draws, from the player's side, overall and per
        difficulty: `{"games", "player_won", "engine_won", "drawn",
        "by_difficulty": {label: {...}}}`. Zeros when no game has ended, which
        is a fact too ("none yet")."""
        overall = _empty()
        by_difficulty: dict[str, dict[str, int]] = {}
        with self._lock:
            standing = list(self._standing.values())
        for entry in standing:
            key = _COUNTS[entry["winner"]]
            for counts in (
                overall,
                by_difficulty.setdefault(entry["difficulty"], _empty()),
            ):
                counts["games"] += 1
                counts[key] += 1
        return {**overall, "by_difficulty": by_difficulty}


def _empty() -> dict[str, int]:
    return {"games": 0, "player_won": 0, "engine_won": 0, "drawn": 0}


def counts_of(tally: Mapping[str, Any]) -> frozenset[tuple[str, int]]:
    """Every count a tally states, as `(what, n)`: overall and per difficulty.
    What speech accuracy backs a claimed count with (`honesty`)."""
    out: set[tuple[str, int]] = set()
    for counts in (tally, *dict(tally.get("by_difficulty") or {}).values()):
        for key in ("games", "player_won", "engine_won", "drawn"):
            if isinstance(counts.get(key), int):
                out.add((key, counts[key]))
    return frozenset(out)
