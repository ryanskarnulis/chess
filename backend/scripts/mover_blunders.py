"""How often Glitch's mover blunders, on a fixed set of positions (#495).

    python scripts/mover_blunders.py play gemma-4-12b --report /tmp/blunders.jsonl
    python scripts/mover_blunders.py append /tmp/blunders.jsonl --label "baseline"
    python scripts/mover_blunders.py trend [--model gemma-4-12b]
    python scripts/mover_blunders.py harvest live.json --side black --id g927c \\
        --source "game 927c238e"

The yardstick for `docs/glitch-sight.md` (decision 5). `mover_strength.py`
plays whole games: hours per reading, with Elo bounds hundreds of points
wide, too slow and too coarse to tell two arms apart. This asks the mover for
one move in each position of a fixed corpus and judges each move afterwards.

**The mover sees what it sees in a game.** Each position is stored as a root
FEN and the moves from it, so `Mover.choose` builds its usual request from a
real `GameSession`, move list and all. Nothing else reaches it.

**Stockfish only judges, after the move is chosen.** At a fixed depth, one
thread and a fixed hash, it scores the best move and the move played, from
the mover's side. The loss is the centipawns given up against the best move
(never below 0), with mate scores clamped to ±1000 so one mate does not swamp
the mean. A loss of 150 or more is a **blunder**. A mover failure (deadline,
truncation, retries, provider) is counted apart as a fallback and never
judged: the app would have played Stockfish's `beginner` move, not his.

**Arms are interleaved per position.** Consecutive samples on one llama-server
are correlated, so a run asks every arm at one position before moving on, and
rotates which arm goes first. An arm is a model, or a model with its own
profile file (`label=model@profile.toml`), so a profile key can be compared
on one model.

**The corpus** (`tests/fixtures/mover_blunder_corpus.json`) is grown with
`harvest`, which reads games the mover played (a PGN, a saved session such
as `live.json`, or a `mover_strength.py` report) and keeps the positions,
for the side asked, where the move played gave up 150 or more (`blunder`)
or the other side's move just did (`free`: something to take, so an arm
that only makes him timid scores badly too). Positions already decided (the
best move beyond ±500) are left out: losing faster in a lost position is not
something this can measure.

`play` writes one record per sample the moment it is judged, and re-running
it on the same report asks only for the samples it lacks. `append` keeps one
line per arm in `docs/mover-blunders-history.jsonl`, keyed by model and
`profile_sha`; `trend` prints it. A manual command like the evals, never in
CI: it needs the GPU, and an idle one (`play` refuses a busy server unless
`--force`). The pure parts are tested off the GPU in
`tests/test_mover_blunders.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import chess
import chess.engine
import chess.pgn
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from chessapp.game import GameSession  # noqa: E402
from chessapp.move_source import CHOOSER_FALLBACK  # noqa: E402
from chessapp.mover import ILLEGAL, MoverChoice, MoverFailed  # noqa: E402
from evalstats import wilson_interval  # noqa: E402
from mover_strength import (  # noqa: E402
    DEFAULT_BASE_URL,
    _append,
    _git_sha,
    _percentile,
    _round,
    _share,
    busy,
    profile_sha,
    read_report,
)
from mover_strength import (
    GAME as STRENGTH_GAME,
)

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "docs" / "mover-blunders-history.jsonl"
CORPUS = ROOT / "backend" / "tests" / "fixtures" / "mover_blunder_corpus.json"
DEFAULT_SAMPLES = 3
DEFAULT_DEPTH = 18
# A move giving up this many centipawns or more is a blunder, and a position
# where a move did is worth keeping (`harvest`).
BLUNDER_CP = 150
# Mate scores read as this many centipawns, and no evaluation reads beyond it.
CLAMP_CP = 1000
# A position whose best move is already beyond this is decided: a move there
# can lose faster, but not lose the game, so it measures nothing.
DECIDED_CP = 500
HEADER = "blunders_header"
SAMPLE = "blunders_sample"
PLAYED = "played"
TAG_BLUNDER = "blunder"
TAG_FREE = "free"


# -- judging -----------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """One position's best move and score, from the side to move."""

    best_uci: str
    best_cp: int


class Judge(Protocol):
    def best(self, board: chess.Board) -> Verdict: ...

    def after(self, board: chess.Board, move: chess.Move) -> int: ...


def clamp(cp: int) -> int:
    return max(-CLAMP_CP, min(CLAMP_CP, cp))


class StockfishJudge:
    """Stockfish at a fixed depth, one thread and a fixed hash, so two runs
    judge one move alike. Each position is analysed once and remembered."""

    def __init__(self, path: str, depth: int = DEFAULT_DEPTH) -> None:
        self.depth = depth
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": 1, "Hash": 64})
        self._seen: dict[str, Verdict] = {}

    def __enter__(self) -> StockfishJudge:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.engine.quit()

    def best(self, board: chess.Board) -> Verdict:
        key = board.fen()
        if key not in self._seen:
            # A new game per position clears the hash, so what was analysed
            # before never changes a score.
            info = self.engine.analyse(
                board, chess.engine.Limit(depth=self.depth), game=key
            )
            score = info["score"].pov(board.turn).score(mate_score=100_000)
            self._seen[key] = Verdict(info["pv"][0].uci(), clamp(score))
        return self._seen[key]

    def after(self, board: chess.Board, move: chess.Move) -> int:
        """The score after `move`, from the side that played it."""
        child = board.copy()
        child.push(move)
        if child.is_game_over():
            return _final_cp(child)
        return -self.best(child).best_cp


def _final_cp(board: chess.Board) -> int:
    """A finished position's score for the side that just moved."""
    if board.is_checkmate():
        return CLAMP_CP
    return 0


def judge_move(judge: Judge, board: chess.Board, move: chess.Move) -> dict[str, Any]:
    """The judgement of `move` in `board`: the best move and its score, the
    played move's score, the loss and whether it is a blunder."""
    verdict = judge.best(board)
    best = chess.Move.from_uci(verdict.best_uci)
    played_cp = verdict.best_cp if move == best else judge.after(board, move)
    loss = max(0, verdict.best_cp - played_cp)
    return {
        "best_uci": verdict.best_uci,
        "best_san": board.san(best),
        "best_cp": verdict.best_cp,
        "played_cp": played_cp,
        "loss_cp": loss,
        "blunder": loss >= BLUNDER_CP,
    }


# -- the corpus --------------------------------------------------------------


def session_for(position: dict[str, Any]) -> GameSession:
    """The game a corpus position comes from, at that position, every move
    entering through the session's legality gate."""
    session = GameSession(fen=position["root_fen"])
    for uci in position["moves"]:
        if not session.submit_move(uci).legal:
            raise ValueError(f"{position['id']}: {uci} is not legal")
    return session


def load_corpus(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["positions"]


def corpus_sha(positions: Sequence[dict[str, Any]]) -> str:
    """The digest of what the mover is asked: two runs compare only when it
    matches. The notes kept beside each position for the reader are left
    out, so editing them does not split the history."""
    asked = [
        {key: p[key] for key in ("id", "root_fen", "moves")}
        for p in sorted(positions, key=lambda p: p["id"])
    ]
    return hashlib.sha256(json.dumps(asked).encode()).hexdigest()[:12]


def _ply_id(source_id: str, board: chess.Board) -> str:
    side = "w" if board.turn == chess.WHITE else "b"
    return f"{source_id}-{board.fullmove_number}{side}"


def harvest_game(
    judge: Judge,
    root_fen: str,
    moves: Sequence[str],
    *,
    sides: set[bool],
    source_id: str,
    source: str,
    max_positions: int | None = None,
    not_his: Collection[int] = (),
) -> list[dict[str, Any]]:
    """The positions worth asking about in one game, for the side(s) in
    `sides`: the move played gave up `BLUNDER_CP` or more, or the other
    side's move just did. Decided positions are left out. A move at a ply
    in `not_his` was the fallback's, not the mover's, so it is never his
    blunder. With `max_positions`, the biggest swings are kept."""
    board = chess.Board(root_fen)
    found: list[tuple[int, dict[str, Any]]] = []
    previous_loss = 0
    for index, uci in enumerate(moves):
        move = chess.Move.from_uci(uci)
        judged = judge_move(judge, board, move)
        if board.turn in sides and abs(judged["best_cp"]) <= DECIDED_CP:
            tags = []
            if judged["blunder"] and index not in not_his:
                tags.append(TAG_BLUNDER)
            if previous_loss >= BLUNDER_CP:
                tags.append(TAG_FREE)
            if tags:
                position = {
                    "id": _ply_id(source_id, board),
                    "source": source,
                    "root_fen": root_fen,
                    "moves": list(moves[:index]),
                    "side": "white" if board.turn == chess.WHITE else "black",
                    "tags": tags,
                    "played": board.san(move),
                    "best": judged["best_san"],
                    "best_cp": judged["best_cp"],
                    "played_cp": judged["played_cp"],
                }
                found.append((max(judged["loss_cp"], previous_loss), position))
        previous_loss = judged["loss_cp"]
        board.push(move)
    if max_positions is not None:
        found = sorted(found, key=lambda item: -item[0])[:max_positions]
        found.sort(key=lambda item: len(item[1]["moves"]))
    return [position for _, position in found]


def read_games(path: Path) -> list[tuple[str, list[str], frozenset[int]]]:
    """Games as `(root FEN, UCI moves, fallback plies)`: from a PGN, a saved
    session (a `live.json` or a bare `GameSession.to_dict()`), or the games
    of a `mover_strength.py` report, the one source that says which of
    Glitch's moves were the fallback's."""
    if path.suffix == ".pgn":
        games = []
        with path.open() as handle:
            while (game := chess.pgn.read_game(handle)) is not None:
                board = game.board()
                moves = [m.uci() for m in game.mainline_moves()]
                games.append((board.fen(), moves, frozenset()))
        return games
    if path.suffix == ".jsonl":
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        return [
            (
                chess.STARTING_FEN,
                _uci_moves(chess.STARTING_FEN, line["sans"]),
                _fallback_plies(line),
            )
            for line in lines
            if line.get("kind") == STRENGTH_GAME
        ]
    data = json.loads(path.read_text())
    session = data.get("session", data)
    return [(session["root_fen"], session["moves"], frozenset())]


def _fallback_plies(game: dict[str, Any]) -> frozenset[int]:
    """The plies of a strength-report game where Glitch's move was the
    fallback's: his moves are every other ply, from his colour's first."""
    first = 0 if game["colour"] == "white" else 1
    return frozenset(
        first + 2 * number
        for number, move in enumerate(game["moves"])
        if move["chooser"] == CHOOSER_FALLBACK
    )


def _uci_moves(root_fen: str, sans: Sequence[str]) -> list[str]:
    board = chess.Board(root_fen)
    return [board.push_san(san).uci() for san in sans]


def merge_corpus(
    existing: Sequence[dict[str, Any]], found: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """`existing` plus the positions of `found` it lacks (by position, so one
    position reached twice is asked once), and how many were new."""
    seen = {_position_key(p) for p in existing}
    merged = list(existing)
    for position in found:
        key = _position_key(position)
        if key in seen:
            continue
        seen.add(key)
        merged.append(position)
    return merged, len(merged) - len(existing)


def _position_key(position: dict[str, Any]) -> str:
    board = chess.Board(position["root_fen"])
    for uci in position["moves"]:
        board.push_uci(uci)
    return board.epd()


# -- playing -----------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One way of asking: a label, a model, and the mover built for it."""

    label: str
    model: str
    profile_source: str
    profile_sha: str
    mover_settings: dict[str, Any]
    mover: Any

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "profile_source": self.profile_source,
            "profile_sha": self.profile_sha,
            "mover": self.mover_settings,
        }


def schedule(
    positions: Sequence[dict[str, Any]], arms: Sequence[str], samples: int
) -> list[tuple[int, str, str]]:
    """Every sample of a run as `(sample, position id, arm)`: each round asks
    every position once per arm, all arms at one position together, with the
    first arm rotating so none always goes first on a fresh prefix."""
    plan = []
    for sample in range(samples):
        for index, position in enumerate(positions):
            shift = (index + sample) % len(arms)
            for arm in [*arms[shift:], *arms[:shift]]:
                plan.append((sample, position["id"], arm))
    return plan


def _calls(calls: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "status": call.status,
            "ms": call.ms,
            "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens,
        }
        for call in calls
    ]


def ask(
    arm: Arm, position: dict[str, Any], sample: int, judge: Judge
) -> dict[str, Any]:
    """One sample: the arm's mover asked at `position`, its move judged."""
    session = session_for(position)
    record: dict[str, Any] = {
        "kind": SAMPLE,
        "arm": arm.label,
        "position": position["id"],
        "tags": position["tags"],
        "sample": sample,
    }
    try:
        choice: MoverChoice = arm.mover.choose(session)
    except MoverFailed as failed:
        attempts = [attempt.verdict for attempt in failed.attempts]
        return {
            **record,
            "outcome": str(failed.kind),
            "attempts": attempts,
            "calls": _calls(failed.calls),
        }
    board = chess.Board(session.fen())
    return {
        **record,
        "outcome": PLAYED,
        "uci": choice.uci,
        "san": choice.san,
        "intent": choice.intent,
        "attempts": [attempt.verdict for attempt in choice.attempts],
        "calls": _calls(choice.calls),
        **judge_move(judge, board, chess.Move.from_uci(choice.uci)),
    }


def pending(
    records: Sequence[dict[str, Any]],
    header: dict[str, Any],
    positions: Sequence[dict[str, Any]],
) -> list[tuple[int, str, str]]:
    """The samples of `header`'s run that `records` lacks. A report begun
    under other arms, another corpus or another judge is refused, never
    mixed."""
    old = next((r for r in records if r.get("kind") == HEADER), None)
    if old is not None:
        for key in ("arms", "corpus_sha", "samples", "depth", "blunder_cp"):
            if old.get(key) != header.get(key):
                raise SystemExit(
                    f"the report was begun with {key}={old.get(key)!r}, "
                    f"not {header.get(key)!r}; use a new --report"
                )
    done = {
        (r["sample"], r["position"], r["arm"])
        for r in records
        if r.get("kind") == SAMPLE
    }
    labels = [arm["label"] for arm in header["arms"]]
    plan = schedule(positions, labels, header["samples"])
    return [item for item in plan if item not in done]


def run(
    header: dict[str, Any],
    report: Path,
    positions: Sequence[dict[str, Any]],
    arms: Sequence[Arm],
    judge: Judge,
    *,
    say: Callable[[str], None] = print,
) -> None:
    """Ask for the samples `report` lacks, appending each as it is judged."""
    records = read_report(report)
    todo = pending(records, header, positions)
    if not records:
        _append(report, header)
    by_id = {position["id"]: position for position in positions}
    by_label = {arm.label: arm for arm in arms}
    for count, (sample, position_id, label) in enumerate(todo, 1):
        record = ask(by_label[label], by_id[position_id], sample, judge)
        _append(report, record)
        if record["outcome"] == PLAYED:
            mark = " BLUNDER" if record["blunder"] else ""
            said = (
                f"{record['san']} (best {record['best_san']}, "
                f"-{record['loss_cp']}){mark}"
            )
        else:
            said = f"fallback: {record['outcome']}"
        say(f"[{count}/{len(todo)}] {label} {position_id} #{sample}: {said}")


# -- reading -----------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 1) if values else None


def rates(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """One arm's samples, read: the blunder rate and its one-sided 95% Wilson
    bounds, the centipawn loss, the same per tag, the fallbacks and why,
    the illegal attempts and the seconds per move."""
    samples = list(samples)
    played = [s for s in samples if s["outcome"] == PLAYED]
    blunders = sum(s["blunder"] for s in played)
    low, high = wilson_interval(blunders, len(played))
    losses = [s["loss_cp"] for s in played]
    attempts = [verdict for s in samples for verdict in s["attempts"]]
    fallbacks = [s for s in samples if s["outcome"] != PLAYED]
    seconds = [sum(call["ms"] for call in s["calls"]) / 1000 for s in samples]
    by_tag: dict[str, dict[str, Any]] = {}
    for tag in sorted({tag for s in played for tag in s["tags"]}):
        tagged = [s for s in played if tag in s["tags"]]
        by_tag[tag] = {
            "moves": len(tagged),
            "blunders": sum(s["blunder"] for s in tagged),
            "mean_loss_cp": _mean([s["loss_cp"] for s in tagged]),
        }
    return {
        "samples": len(samples),
        "moves": len(played),
        "blunders": blunders,
        "blunder_rate": round(blunders / len(played), 3) if played else None,
        "blunder_low": round(low, 3),
        "blunder_high": round(high, 3),
        "mean_loss_cp": _mean(losses),
        "median_loss_cp": statistics.median(losses) if losses else None,
        "by_tag": by_tag,
        "attempts": len(attempts),
        "illegal_attempts": attempts.count(ILLEGAL),
        "fallbacks": len(fallbacks),
        "fallback_kinds": dict(Counter(s["outcome"] for s in fallbacks)),
        "seconds_per_move_p50": _round(statistics.median(seconds) if seconds else None),
        "seconds_per_move_p90": _round(_percentile(seconds, 0.9)),
    }


_CONFIG_KEYS = ("git_sha", "corpus_sha", "positions", "samples", "depth", "blunder_cp")


def summarize(
    records: Iterable[dict[str, Any]], *, label: str, date: str
) -> list[dict[str, Any]]:
    """The history lines of a run's report: one per arm."""
    records = list(records)
    header = next((r for r in records if r.get("kind") == HEADER), None)
    if header is None:
        raise SystemExit("the report has no header: was it written by `play`?")
    samples = [r for r in records if r.get("kind") == SAMPLE]
    lines = []
    for arm in header["arms"]:
        lines.append(
            {
                "date": date,
                "label": label,
                "arm": arm["label"],
                "model": arm["model"],
                "profile_sha": arm["profile_sha"],
                "profile_source": arm["profile_source"],
                "mover": arm["mover"],
                **{key: header.get(key) for key in _CONFIG_KEYS},
                **rates(s for s in samples if s["arm"] == arm["label"]),
            }
        )
    return lines


def _rate_cell(line: dict[str, Any]) -> str:
    if line["blunder_rate"] is None:
        return "—"
    return (
        f"{line['blunders']}/{line['moves']} {line['blunder_rate']:.0%} "
        f"[{line['blunder_low']:.0%}, {line['blunder_high']:.0%}]"
    )


def trend(lines: Iterable[dict[str, Any]], model: str | None = None) -> str:
    """The history as a table, oldest first, one row per arm per run."""
    header = (
        "date        model             profile       corpus        "
        "blunders              cp loss mean/med  fallback      "
        "s/move p50/p90  label"
    )
    rows = [header]
    for line in lines:
        if model is not None and line["model"] != model:
            continue
        loss = f"{line['mean_loss_cp']}/{line['median_loss_cp']}"
        rows.append(
            f"{line['date']:<11} {line['model']:<17} {line['profile_sha']:<13} "
            f"{line['corpus_sha']:<13} {_rate_cell(line):<21} {loss:<17} "
            f"{_share(line['fallbacks'], line['samples']):<13} "
            f"{line['seconds_per_move_p50']}/{line['seconds_per_move_p90']:<10} "
            f"{line['label']}"
            + ("" if line["arm"] == line["model"] else f" [{line['arm']}]")
        )
    return "\n".join(rows)


# -- the command -------------------------------------------------------------


def parse_arm(text: str) -> tuple[str, str, Path | None]:
    """`model`, `label=model` or `label=model@profile.toml`, as
    `(label, model, profile path)`."""
    label, _, rest = text.rpartition("=")
    model, _, path = rest.partition("@")
    if not model:
        raise argparse.ArgumentTypeError(f"no model in arm {text!r}")
    return label or model, model, Path(path) if path else None


def _build_arm(text: str, base_url: str) -> Arm:
    from chessapp.mover import Mover
    from chessapp.profiles import MOVER, load_profile, parse_profile
    from chessapp.provider import LlamaCppProvider

    label, model, path = parse_arm(text)
    if path is None:
        profile = load_profile(model)
    else:
        profile = parse_profile(model, path.read_text(encoding="utf-8"), str(path))
    described = profile.describe()
    return Arm(
        label=label,
        model=model,
        profile_source=profile.source,
        profile_sha=profile_sha(described),
        mover_settings=described["phases"][MOVER],
        mover=Mover(LlamaCppProvider(base_url, model), profile),
    )


def _play(args: argparse.Namespace) -> int:
    if not args.force:
        with httpx.Client(timeout=10) as client:
            found = busy(client, args.base_url)
        if found:
            print("the server is busy; the timings would be contended:")
            print("\n".join(f"  {line}" for line in found))
            print("wait for it, or pass --force")
            return 1
    positions = load_corpus(args.corpus)
    arms = [_build_arm(text, args.base_url) for text in args.arms]
    if len({arm.label for arm in arms}) != len(arms):
        raise SystemExit("two arms share a label; name them with label=model")
    header = {
        "kind": HEADER,
        "arms": [arm.describe() for arm in arms],
        "corpus_sha": corpus_sha(positions),
        "positions": len(positions),
        "samples": args.samples,
        "depth": args.depth,
        "blunder_cp": BLUNDER_CP,
        "git_sha": _git_sha(),
        "base_url": args.base_url,
        "date": datetime.now(UTC).date().isoformat(),
    }
    with StockfishJudge(args.stockfish, args.depth) as judge:
        run(header, args.report, positions, arms, judge)
    samples = [r for r in read_report(args.report) if r.get("kind") == SAMPLE]
    for arm in arms:
        print(arm.label)
        mine = (s for s in samples if s["arm"] == arm.label)
        print(json.dumps(rates(mine), indent=2))
    return 0


def _harvest(args: argparse.Namespace) -> int:
    sides = {
        "white": {chess.WHITE},
        "black": {chess.BLACK},
        "both": {chess.WHITE, chess.BLACK},
    }[args.side]
    corpus = load_corpus(args.corpus) if args.corpus.exists() else []
    found: list[dict[str, Any]] = []
    with StockfishJudge(args.stockfish, args.depth) as judge:
        games = read_games(args.games)
        for number, (root_fen, moves, not_his) in enumerate(games):
            game_id = args.id if len(games) == 1 else f"{args.id}g{number + 1}"
            found += harvest_game(
                judge,
                root_fen,
                moves,
                sides=sides,
                source_id=game_id,
                source=args.source,
                max_positions=args.max_per_game,
                not_his=not_his,
            )
    merged, added = merge_corpus(corpus, found)
    args.corpus.write_text(json.dumps({"positions": merged}, indent=2) + "\n")
    for position in found:
        print(
            f"{position['id']:<16} {','.join(position['tags']):<13} "
            f"played {position['played']:<7} best {position['best']:<7} "
            f"({position['best_cp']:+} -> {position['played_cp']:+})"
        )
    print(f"{added} new of {len(found)}; the corpus holds {len(merged)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    stockfish = os.environ.get("CHESSAPP_STOCKFISH", "stockfish")

    play = sub.add_parser("play", help="ask the mover at every corpus position")
    play.add_argument(
        "arms", nargs="+", help="model, label=model or label=model@profile.toml"
    )
    play.add_argument("--report", type=Path, required=True)
    play.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    play.add_argument("--corpus", type=Path, default=CORPUS)
    play.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    play.add_argument("--base-url", default=DEFAULT_BASE_URL)
    play.add_argument("--stockfish", default=stockfish)
    play.add_argument("--force", action="store_true", help="ask a busy server")

    append = sub.add_parser("append", help="add a report to the history")
    append.add_argument("report", type=Path)
    append.add_argument("--label", required=True)
    append.add_argument("--history", type=Path, default=HISTORY)

    show = sub.add_parser("trend", help="print the history")
    show.add_argument("--model")
    show.add_argument("--history", type=Path, default=HISTORY)

    harvest = sub.add_parser("harvest", help="add a game's positions to the corpus")
    harvest.add_argument("games", type=Path, help=".pgn, saved session .json, .jsonl")
    harvest.add_argument("--side", choices=("white", "black", "both"), required=True)
    harvest.add_argument("--id", required=True, help="short id for position ids")
    harvest.add_argument("--source", required=True, help="where the game came from")
    harvest.add_argument("--max-per-game", type=int)
    harvest.add_argument("--corpus", type=Path, default=CORPUS)
    harvest.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    harvest.add_argument("--stockfish", default=stockfish)

    args = parser.parse_args(argv)
    if args.command == "play":
        return _play(args)
    if args.command == "harvest":
        return _harvest(args)
    if args.command == "append":
        lines = summarize(
            read_report(args.report),
            label=args.label,
            date=datetime.now(UTC).date().isoformat(),
        )
        for line in lines:
            _append(args.history, line)
        print(trend(lines))
        return 0
    print(trend(read_report(args.history), args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
