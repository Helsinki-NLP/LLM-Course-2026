#!/bin/bash
# Run the controlled bf16/FP8 comparison inside one Roihu GPU allocation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT="${1:-results/roihu}"

cleanup() {
  bash scripts/stop.sh >/dev/null 2>&1 || true
}
trap cleanup EXIT

bash scripts/restart.sh bf16
bash scripts/measure.sh bf16 "$OUTPUT_ROOT/bf16" --structured

bash scripts/restart.sh fp8
bash scripts/measure.sh fp8 "$OUTPUT_ROOT/fp8"

python3 summarize_results.py \
  --bf16-dir "$OUTPUT_ROOT/bf16" \
  --fp8-dir "$OUTPUT_ROOT/fp8" \
  --structured-dir "$OUTPUT_ROOT/bf16" \
  --output-dir "$OUTPUT_ROOT"

echo "Experiment completed: $OUTPUT_ROOT"
