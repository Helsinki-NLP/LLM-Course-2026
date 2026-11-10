#!/bin/bash
# Optional: run the core lab steps without typing them (Hugging Face, then vLLM).
# Usage: bash scripts/run_experiment.sh RESULTS_DIR [LAB2_CONFIG]
# The interactive steps in README.md do the same thing one by one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:?usage: run_experiment.sh RESULTS_DIR [LAB2_CONFIG]}"
TASK="${2:-../../Week-2/Lab-2/configs/roihu.json}"
trap 'bash scripts/stop_server.sh >/dev/null 2>&1 || true' EXIT
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
export HF_HOME="${HF_HOME:-/scratch/project_2021893/${USER}/hf}"

echo "=== Hugging Face ==="
(
  module purge
  module load python-pytorch/2.13
  python3 benchmark.py --backend hf --output-dir "$OUT/hf"
  python3 score_task.py --backend hf --task-config "$TASK" --output-dir "$OUT/hf"
)

echo "=== vLLM ==="
bash scripts/start_server.sh configs/model.json
python3 benchmark.py --backend server --output-dir "$OUT/vllm"
python3 score_task.py --backend server --task-config "$TASK" --output-dir "$OUT/vllm"
bash scripts/stop_server.sh

python3 summarize_results.py "$OUT"
