"""How strong Glitch's mover is, per model, tracked over time (#476).

    python scripts/mover_strength.py play gemma-4-26b-a4b --report /tmp/strength.jsonl
    python scripts/mover_strength.py append /tmp/strength.jsonl --label "first numbers"
    python scripts/mover_strength.py trend [--model gemma-4-26b-a4b]

`play` sets the mover (`mover.py`) against Stockfish at fixed UCI Elo levels,
alternating colours, and writes one record per finished game to a report. It
plays through the app's own path: `move_source.GlitchSource` over an engine at
the `glitch` tier, so a mover failure is the app's fallback (Stockfish at
`beginner`), counted, never hidden. Each game is written the moment it ends,
and re-running `play` on the same report plays only the games it lacks, so a
run of several hours survives a crash.

`append` turns a report into one line of `docs/mover-strength-history.jsonl`,
keyed by model and profile (`profile_sha` is the digest of the profile's
`describe()`): the score per level, an estimated Elo, the illegal-attempt rate,
the fallback rate and the mover's seconds per move. `trend` prints the history.

**The Elo is a performance rating.** It is the rating at which the games'
expected score equals the score actually made (maximum likelihood under the
logistic Elo curve), and its range is the one-sided 95% Wilson bounds on the
score read through the same curve. A clean sweep or a whitewash has no
maximum, so it reads as a bound only (`< 1050`). Stockfish's `UCI_Elo` floors
at 1320, so a weak mover is measured against opponents well above it and gets
mostly a ceiling: that is the honest answer at this sample size.

A manual command like the evals, never in CI: it needs the GPU, and an idle
one, since every move is a thinking call (`play` refuses a busy server unless
`--force`). The pure parts are tested off the GPU in
`tests/test_mover_strength.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from chessapp.game import GameSession  # noqa: E402
from chessapp.move_source import CHOOSER_FALLBACK, MoveSource, SourcedMove  # noqa: E402
from chessapp.mover import ILLEGAL  # noqa: E402
from evalstats import wilson_interval  # noqa: E402

HISTORY = Path(__file__).resolve().parents[2] / "docs" / "mover-strength-history.jsonl"
DEFAULT_BASE_URL = "http://127.0.0.1:8200/v1"
DEFAULT_LEVELS = (1320, 1600)
DEFAULT_GAMES = 4
# A game still going after this many plies is adjudicated a draw: two weak
# sides can shuffle for ever, and the GPU hours are better spent on a new game.
MAX_PLIES = 200
HEADER = "strength_header"
GAME = "strength_game"
# The search range for a rating, wide enough that no real score leaves it.
_ELO_RANGE = (-1000.0, 4500.0)


class _Engine(Protocol):
    def choose_move(self, session: GameSession) -> str: ...


# -- playing -----------------------------------------------------------------


def schedule(levels: Sequence[int], games: int) -> list[tuple[int, int, str]]:
    """Every game of a run as `(index, level, glitch's colour)`, colours
    alternating within each level so each level is half white."""
    plan = []
    for level in levels:
        for game in range(games):
            colour = "white" if game % 2 == 0 else "black"
            plan.append((len(plan), level, colour))
    return plan


def _move_record(sourced: SourcedMove) -> dict[str, Any]:
    """One of Glitch's moves: who chose it, the mover's attempts by verdict,
    and its model calls."""
    attempts = (sourced.mover or {}).get("attempts", [])
    return {
        "uci": sourced.uci,
        "chooser": sourced.chooser,
        "fallback_reason": sourced.fallback_reason,
        "attempts": [attempt["verdict"] for attempt in attempts],
        "calls": [
            {
                "status": call.status,
                "ms": call.ms,
                "prompt_tokens": call.prompt_tokens,
                "completion_tokens": call.completion_tokens,
            }
            for call in sourced.calls
        ],
    }


def play_game(
    source: MoveSource,
    opponent: _Engine,
    *,
    index: int,
    level: int,
    colour: str,
    max_plies: int = MAX_PLIES,
) -> dict[str, Any]:
    """One game of Glitch (`source`) as `colour` against `opponent`, both
    moves entering through the session's legality gate, as in the app."""
    session = GameSession()
    moves: list[dict[str, Any]] = []
    while not session.is_game_over() and len(session.move_history()) < max_plies:
        if session.turn == colour:
            sourced = source.choose(session)
            uci = sourced.uci
            moves.append(_move_record(sourced))
        else:
            uci = opponent.choose_move(session)
        result = session.submit_move(uci)
        if not result.legal:
            raise RuntimeError(f"an illegal move reached the board: {uci}")
    outcome = session.outcome()
    if outcome is None:
        result, termination, adjudicated = "1/2-1/2", "max_plies", True
    else:
        result, termination, adjudicated = outcome.result, outcome.termination, False
    return {
        "kind": GAME,
        "index": index,
        "level": level,
        "colour": colour,
        "result": result,
        "score": glitch_score(result, colour),
        "termination": termination,
        "adjudicated": adjudicated,
        "plies": len(session.move_history()),
        "sans": session.move_history(),
        "moves": moves,
    }


def glitch_score(result: str, colour: str) -> float:
    if result == "1/2-1/2":
        return 0.5
    return 1.0 if (result == "1-0") == (colour == "white") else 0.0


def read_report(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pending(
    records: Sequence[dict[str, Any]], header: dict[str, Any]
) -> list[tuple[int, int, str]]:
    """The games of `header`'s run that `records` does not hold yet. A report
    begun under another model, profile or plan is refused, never mixed."""
    old = next((r for r in records if r.get("kind") == HEADER), None)
    if old is not None:
        for key in ("model", "profile_sha", "levels", "games_per_level", "max_plies"):
            if old.get(key) != header.get(key):
                raise SystemExit(
                    f"the report was begun with {key}={old.get(key)!r}, "
                    f"not {header.get(key)!r}; use a new --report"
                )
    done = {r["index"] for r in records if r.get("kind") == GAME}
    plan = schedule(header["levels"], header["games_per_level"])
    return [game for game in plan if game[0] not in done]


def run(
    header: dict[str, Any],
    report: Path,
    source: MoveSource,
    opponent: Any,
    *,
    say: Callable[[str], None] = print,
) -> None:
    """Play the games `report` lacks, appending each as it ends. `opponent`
    is an engine whose `set_elo` picks each game's level."""
    records = read_report(report)
    todo = pending(records, header)
    if not records:
        _append(report, header)
    for index, level, colour in todo:
        opponent.set_elo(level)
        say(f"game {index + 1}: Glitch as {colour} vs Stockfish {level}…")
        record = play_game(
            source,
            opponent,
            index=index,
            level=level,
            colour=colour,
            max_plies=header["max_plies"],
        )
        _append(report, record)
        say(
            f"  {record['result']} ({record['termination']}, {record['plies']} "
            f"plies): Glitch scored {record['score']:g}"
        )


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as out:
        out.write(json.dumps(record) + "\n")


# -- reading -----------------------------------------------------------------


def expected(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent - rating) / 400.0))


def _solve(levels: Sequence[int], points: float) -> float:
    """The rating whose expected total over `levels` is `points`. The total
    rises with the rating, so a bisection finds it."""
    low, high = _ELO_RANGE
    for _ in range(100):
        middle = (low + high) / 2
        if sum(expected(middle, level) for level in levels) < points:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def estimate_elo(games: Iterable[dict[str, Any]]) -> dict[str, int | None]:
    """The performance rating of `games` (each with `level` and `score`) and
    its one-sided 95% bounds. `elo` is None for a 0% or 100% score, which has
    no maximum, and a bound is None where the score's bound is 0 or 1."""
    games = list(games)
    if not games:
        return {"elo": None, "low": None, "high": None}
    levels = [game["level"] for game in games]
    points = sum(game["score"] for game in games)
    n = len(games)
    low_rate, high_rate = wilson_interval(points, n)

    def at(rate: float) -> int | None:
        if rate <= 0.0 or rate >= 1.0:
            return None
        return round(_solve(levels, rate * n))

    return {"elo": at(points / n), "low": at(low_rate), "high": at(high_rate)}


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def rates(games: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The mover's habits over `games`: how often it tried an illegal move,
    how often it failed to a fallback and why, and how long a move took."""
    moves = [move for game in games for move in game["moves"]]
    attempts = [verdict for move in moves for verdict in move["attempts"]]
    fallbacks = [m for m in moves if m["chooser"] == CHOOSER_FALLBACK]
    seconds = [sum(call["ms"] for call in m["calls"]) / 1000 for m in moves]
    return {
        "moves": len(moves),
        "attempts": len(attempts),
        "illegal_attempts": attempts.count(ILLEGAL),
        "moves_with_illegal": sum(ILLEGAL in m["attempts"] for m in moves),
        "fallbacks": len(fallbacks),
        "fallback_kinds": dict(Counter(m["fallback_reason"] for m in fallbacks)),
        "seconds_per_move_p50": _round(statistics.median(seconds) if seconds else None),
        "seconds_per_move_p90": _round(_percentile(seconds, 0.9)),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


_CONFIG_KEYS = (
    "model",
    "profile_sha",
    "mover",
    "git_sha",
    "levels",
    "games_per_level",
    "max_plies",
)


def summarize(
    records: Iterable[dict[str, Any]], *, label: str, date: str
) -> dict[str, Any]:
    """One history line from a run's report."""
    records = list(records)
    header = next((r for r in records if r.get("kind") == HEADER), None)
    if header is None:
        raise SystemExit("the report has no header: was it written by `play`?")
    games = [r for r in records if r.get("kind") == GAME]
    by_level: dict[str, dict[str, float]] = {}
    for level in header["levels"]:
        played = [g for g in games if g["level"] == level]
        by_level[str(level)] = {
            "points": sum(g["score"] for g in played),
            "games": len(played),
        }
    return {
        "date": date,
        "label": label,
        **{key: header.get(key) for key in _CONFIG_KEYS},
        "games": len(games),
        "adjudicated": sum(g["adjudicated"] for g in games),
        "by_level": by_level,
        **estimate_elo(games),
        **rates(games),
    }


def _share(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "—"


def _elo_cell(line: dict[str, Any]) -> str:
    elo, low, high = line.get("elo"), line.get("low"), line.get("high")
    if elo is not None:
        ends = ["…" if end is None else str(end) for end in (low, high)]
        return f"{elo} [{ends[0]}, {ends[1]}]"
    if high is not None:
        return f"< {high}"
    if low is not None:
        return f"> {low}"
    return "—"


def trend(lines: Iterable[dict[str, Any]], model: str | None = None) -> str:
    """The history as a table, oldest first, one row per run."""
    header = (
        "date        model             profile       games  score by level"
        "       Elo                illegal          fallback      s/move p50/p90  label"
    )
    rows = [header]
    for line in lines:
        if model is not None and line["model"] != model:
            continue
        scores = " ".join(
            f"{level}:{cell['points']:g}/{cell['games']}"
            for level, cell in line["by_level"].items()
        )
        rows.append(
            f"{line['date']:<11} {line['model']:<17} {line['profile_sha']:<13} "
            f"{line['games']:<6} {scores:<20} {_elo_cell(line):<18} "
            f"{_share(line['illegal_attempts'], line['attempts']):<16} "
            f"{_share(line['fallbacks'], line['moves']):<13} "
            f"{line['seconds_per_move_p50']}/{line['seconds_per_move_p90']:<10} "
            f"{line['label']}"
        )
    return "\n".join(rows)


# -- the server --------------------------------------------------------------


def busy(client: httpx.Client, base_url: str) -> list[str]:
    """What is generating on the server right now, as readable lines; empty
    when it is idle. Behind llama-swap it asks every loaded model's slots,
    on a bare llama-server its own."""
    root = base_url.rstrip("/").removesuffix("/v1")
    running = client.get(f"{root}/running")
    if running.status_code == 200:
        models = [
            entry.get("model")
            for entry in running.json().get("running", [])
            if isinstance(entry, dict) and entry.get("state") == "ready"
        ]
        urls = {model: f"{root}/upstream/{model}/slots" for model in models}
    else:
        urls = {"server": f"{root}/slots"}
    found = []
    for name, url in urls.items():
        slots = client.get(url)
        if slots.status_code != 200:
            continue
        for slot in slots.json():
            if isinstance(slot, dict) and slot.get("is_processing"):
                found.append(f"{name}: slot {slot.get('id')} is generating")
    return found


# -- the command -------------------------------------------------------------


def profile_sha(described: dict[str, Any]) -> str:
    """A profile's digest: two runs share one only when the mover was asked
    with the same settings, so a history row says what it measured."""
    text = json.dumps(described, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _play(args: argparse.Namespace) -> int:
    from chessapp.engine import EnginePlayer
    from chessapp.move_source import GlitchSource
    from chessapp.mover import Mover
    from chessapp.profiles import MOVER, load_profile
    from chessapp.provider import LlamaCppProvider

    if not args.force:
        with httpx.Client(timeout=10) as client:
            found = busy(client, args.base_url)
        if found:
            print("the server is busy; the timings would be contended:")
            print("\n".join(f"  {line}" for line in found))
            print("wait for it, or pass --force")
            return 1
    profile = load_profile(args.model)
    described = profile.describe()
    header = {
        "kind": HEADER,
        "model": args.model,
        "profile_sha": profile_sha(described),
        "mover": described["phases"][MOVER],
        "git_sha": _git_sha(),
        "levels": args.levels,
        "games_per_level": args.games,
        "max_plies": args.max_plies,
        "base_url": args.base_url,
        "date": datetime.now(UTC).date().isoformat(),
    }
    mover = Mover(LlamaCppProvider(args.base_url, args.model), profile)
    with (
        EnginePlayer(path=args.stockfish, rng=random.Random(args.seed)) as fallback,
        EnginePlayer(path=args.stockfish) as opponent,
    ):
        fallback.set_tier("glitch")
        run(header, args.report, GlitchSource(mover, fallback), opponent)
    games = [r for r in read_report(args.report) if r.get("kind") == GAME]
    print(json.dumps({**estimate_elo(games), **rates(games)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="play the mover against Stockfish")
    play.add_argument("model")
    play.add_argument(
        "--levels",
        type=lambda text: [int(part) for part in text.split(",")],
        default=list(DEFAULT_LEVELS),
        help="Stockfish UCI_Elo levels, comma-separated (floor 1320)",
    )
    play.add_argument("--games", type=int, default=DEFAULT_GAMES, help="per level")
    play.add_argument("--report", type=Path, required=True)
    play.add_argument("--base-url", default=DEFAULT_BASE_URL)
    play.add_argument(
        "--stockfish", default=os.environ.get("CHESSAPP_STOCKFISH", "stockfish")
    )
    play.add_argument("--max-plies", type=int, default=MAX_PLIES)
    play.add_argument("--seed", type=int, default=476, help="the fallback's sampler")
    play.add_argument("--force", action="store_true", help="play on a busy server")

    append = sub.add_parser("append", help="add a report to the history")
    append.add_argument("report", type=Path)
    append.add_argument("--label", required=True)
    append.add_argument("--history", type=Path, default=HISTORY)

    show = sub.add_parser("trend", help="print the history")
    show.add_argument("--model")
    show.add_argument("--history", type=Path, default=HISTORY)

    args = parser.parse_args(argv)
    if args.command == "play":
        return _play(args)
    if args.command == "append":
        line = summarize(
            read_report(args.report),
            label=args.label,
            date=datetime.now(UTC).date().isoformat(),
        )
        _append(args.history, line)
        print(trend([line]))
        return 0
    print(trend(read_report(args.history), args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
