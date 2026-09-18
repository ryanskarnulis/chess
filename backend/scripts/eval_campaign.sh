#!/usr/bin/env bash
# Harness-level A/B for a planner change: alternating blocks of five between
# two trees on ONE llama-server, so server drift lands on both arms alike.
#
#   scripts/eval_campaign.sh --a /path/to/main --b /path/to/branch \
#       --k 'ambiguous_knight_then_selection or ambiguous_move' --blocks 4 [--fresh-per-block]
#
# Each block runs `tests/test_agent_evals.py -k <expr>` once per arm with
# CHESSAPP_EVAL_RUNS=5 CHESSAPP_EVAL_MAX_RUNS=5, the `chessapp` under test
# selected by PYTHONPATH=<tree>/backend/src. The tests dir is THIS checkout's
# (the campaign worktree serves both arms), the venv is this checkout's, and
# every path is absolute — a gate that cannot prove which tree it measured
# measured nothing (docs/agent-evals.md). Before each arm-block the log records
# the tree's HEAD, the sha of its personality.py, `chessapp.__file__` as Python
# resolves it, and llama-swap /running. The arm order flips every block.
#
# --fresh-per-block calls llama-swap /unload before each arm-block after the
# probe's pre-flight (refuses while a slot is processing or another job holds
# the card). Reports land in --out (default ./campaign-<utc>/) as
# block-<n>-<arm>.jsonl plus a log, and the last line is the joined table.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$(cd "$HERE/.." && pwd)"
PYTHON="$BACKEND/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python)"

A=""; B=""; K=""; BLOCKS=4; FRESH=0; OUT=""; RUNS=5
BASE_URL="${LLAMACPP_BASE_URL:-http://127.0.0.1:8200/v1}"
MODEL="${LLAMACPP_MODEL:-gemma-4-12b}"
while [ $# -gt 0 ]; do
  case "$1" in
    --a) A="$2"; shift 2 ;;
    --b) B="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --blocks) BLOCKS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --fresh-per-block) FRESH=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$A" ] && [ -n "$B" ] && [ -n "$K" ] || { echo "need --a, --b and --k" >&2; exit 2; }
A="$(cd "$A" && pwd)"; B="$(cd "$B" && pwd)"
[ -d "$A/backend/src/chessapp" ] || { echo "$A is not a chess tree" >&2; exit 2; }
[ -d "$B/backend/src/chessapp" ] || { echo "$B is not a chess tree" >&2; exit 2; }
OUT="${OUT:-$PWD/campaign-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
LOG="$OUT/campaign.log"
SWAP_ROOT="${BASE_URL%/v1}"

log() { echo "$*" | tee -a "$LOG"; }

identity() {  # $1 = arm name, $2 = tree
  local head sha file running
  head="$(git -C "$2" rev-parse --short HEAD)"
  sha="$(sha256sum "$2/backend/src/chessapp/personality.py" | cut -c1-12)"
  file="$(cd "$BACKEND" && PYTHONPATH="$2/backend/src" "$PYTHON" -c 'import chessapp; print(chessapp.__file__)')"
  running="$(curl -s -m 5 "$SWAP_ROOT/running" || echo '{}')"
  log "  arm $1: tree $2 HEAD $head personality $sha chessapp $file running $running"
  case "$file" in "$2"/*) ;; *) log "  !! chessapp resolved outside $2; aborting"; exit 1 ;; esac
}

fresh() {
  (cd "$BACKEND" && "$PYTHON" scripts/probe_planner.py --preflight-only --base-url "$BASE_URL" --model "$MODEL") \
    || { log "  pre-flight refused the unload; aborting"; exit 1; }
  curl -s -m 10 "$SWAP_ROOT/unload" >/dev/null
  log "  unloaded $(date -u +%H:%M:%SZ); the first request reloads"
}

run_block() {  # $1 = block number, $2 = arm name, $3 = tree
  local report="$OUT/block-$1-$2.jsonl"
  log "block $1 arm $2 start $(date -u +%H:%M:%SZ)"
  [ "$FRESH" = 1 ] && fresh
  identity "$2" "$3"
  (cd "$BACKEND" && PYTHONPATH="$3/backend/src" CHESSAPP_AGENT_EVALS=1 \
     CHESSAPP_EVAL_RUNS="$RUNS" CHESSAPP_EVAL_MAX_RUNS="$RUNS" CHESSAPP_EVAL_REPORT="$report" \
     LLAMACPP_BASE_URL="$BASE_URL" LLAMACPP_MODEL="$MODEL" \
     "$PYTHON" -m pytest "$BACKEND/tests/test_agent_evals.py" -k "$K" -s 2>&1 | tee -a "$OUT/block-$1-$2.out" | grep -E '^\[eval\]|passed|failed' || true)
  log "block $1 arm $2 end $(date -u +%H:%M:%SZ)"
  SPECS+=("$2=$report")
}

SPECS=()
log "campaign $(date -u +%FT%TZ) tests $BACKEND/tests k='$K' blocks=$BLOCKS runs=$RUNS fresh=$FRESH"
for ((i = 1; i <= BLOCKS; i++)); do
  if (( i % 2 )); then run_block "$i" a "$A"; run_block "$i" b "$B"
  else run_block "$i" b "$B"; run_block "$i" a "$A"; fi
done
log ""
(cd "$BACKEND" && "$PYTHON" scripts/campaign_report.py "${SPECS[@]}") | tee -a "$LOG"
