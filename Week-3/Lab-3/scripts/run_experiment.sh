#!/bin/bash
# The whole Lab 3 experiment: for bf16 and then fp8, start the server,
# measure speed, score the test set, and stop the server.
# Usage: bash scripts/run_experiment.sh OUTPUT_DIR [LAB2_CONFIG]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="${1:?usage: run_experiment.sh OUTPUT_DIR [LAB2_CONFIG]}"
TASK_CONFIG="${2:-../../Week-2/Lab-2/configs/roihu.json}"
trap 'bash scripts/stop_server.sh >/dev/null 2>&1 || true' EXIT

for LABEL in bf16 fp8; do
  echo "=== $LABEL ==="
  bash scripts/start_server.sh "configs/$LABEL.json"
  python3 benchmark.py --server-config "configs/$LABEL.json" --output-dir "$OUTPUT/$LABEL"
  python3 score_task.py --server-config "configs/$LABEL.json" --task-config "$TASK_CONFIG" --output-dir "$OUTPUT/$LABEL"
  if [[ "$LABEL" == "bf16" ]]; then
    python3 evaluate_structured.py --server-config "configs/$LABEL.json" --task-config "$TASK_CONFIG" --output-dir "$OUTPUT/$LABEL"
  fi
  bash scripts/stop_server.sh
done

python3 summarize_results.py "$OUTPUT"
