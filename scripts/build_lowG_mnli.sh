#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
PCT=0.25
CKPT="roberta_base"
OUTPUT_DIR=${OUTPUT_DIR:-release}
PRED_FILE=${PRED_FILE:-"${OUTPUT_DIR}/mnli_groundedness_predictions.jsonl"}
SUBSET_FILE=${SUBSET_FILE:-"subset/lowG_mnli_v1.jsonl"}
STATS_FILE=${STATS_FILE:-"artifacts/lowG_mnli_stats.json"}
BASELINE_FILE=${BASELINE_FILE:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pct)
      PCT="$2"; shift 2 ;;
    --ckpt)
      CKPT="$2"; shift 2 ;;
    --pred)
      PRED_FILE="$2"; shift 2 ;;
    --subset)
      SUBSET_FILE="$2"; shift 2 ;;
    --stats)
      STATS_FILE="$2"; shift 2 ;;
    --baseline)
      BASELINE_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

mkdir -p "$(dirname "$SUBSET_FILE")"
mkdir -p "$(dirname "$STATS_FILE")"

if [[ ! -f "$PRED_FILE" ]]; then
  echo "Predictions not found at $PRED_FILE. Run scripts/score_groundedness.py first." >&2
  exit 1
fi

if [[ -n "$BASELINE_FILE" && ! -f "$BASELINE_FILE" ]]; then
  echo "Baseline file $BASELINE_FILE missing" >&2
  exit 1
fi

CMD=("$PYTHON_BIN" "$(dirname "$0")/build_lowG_mnli.py" "$PRED_FILE" "--pct" "$PCT" "--output" "$SUBSET_FILE" "--stats" "$STATS_FILE")
if [[ -n "$BASELINE_FILE" ]]; then
  CMD+=("--baseline" "$BASELINE_FILE")
fi
"${CMD[@]}"

echo "Low-groundedness subset written to $SUBSET_FILE"
