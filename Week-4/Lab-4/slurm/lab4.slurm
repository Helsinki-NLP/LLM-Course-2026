#!/bin/bash
#SBATCH --job-name=lab4
#SBATCH --account=project_2021893
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

# Lab 4 as a batch job, for when you cannot get an interactive GPU session.
# It runs README steps 3, 4 and 5, and your own experiment (step 6) if you
# add it below. Submit it from the lab folder on roihu-gpu.csc.fi:
#
#   cd ~/LLM-Course-2026/Week-4/Lab-4
#   mkdir -p logs
#   sbatch slurm/lab4.slurm ../../Week-2/Lab-2/configs/my-task.json          # everything
#
# or as two shorter jobs, one per backend (they can run in either order):
#
#   sbatch slurm/lab4.slurm ../../Week-2/Lab-2/configs/my-task.json hf
#   sbatch slurm/lab4.slurm ../../Week-2/Lab-2/configs/my-task.json vllm
#
# Follow a job with:  tail -f logs/lab4-JOBID.out
# Results go to results/batch/ (a new run replaces the old one). The
# comparison at the end appears once both parts have run.
#
# gputest jobs start quickly but may run at most 15 minutes. If you add a long
# experiment, use --partition=gpumedium and a longer --time instead.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: not a GH200 node. Submit from ssh roihu-gpu.csc.fi." >&2
  exit 1
fi
TASK="${1:-../../Week-2/Lab-2/configs/roihu.json}"   # your Lab 2 task config
PART="${2:-all}"                                       # all, hf or vllm
case "$PART" in all|hf|vllm) ;; *) echo "ERROR: the second argument must be all, hf or vllm" >&2; exit 1;; esac
OUT="results/batch"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
export HF_HOME="${HF_HOME:-/scratch/project_2021893/${USER}/hf}"
if [[ "$PART" != hf ]]; then   # only the job that starts the server stops it
  trap 'bash scripts/stop_server.sh >/dev/null 2>&1 || true' EXIT   # never leave a server running
fi
echo "Task: $TASK   Part: $PART   Results: $OUT"

if [[ "$PART" != vllm ]]; then
echo "=== Step 3: Hugging Face ==="
module purge
module load python-pytorch/2.13
python3 benchmark.py --backend hf --output-dir "$OUT/hf"
python3 score_task.py --backend hf --task-config "$TASK" --output-dir "$OUT/hf"

# Step 6, Hugging Face options: remove the # in front of the one you chose.
# E: python3 tokens_per_word.py --task-config "$TASK" --compare my-translations.txt
# F: python3 score_labels.py --task-config "$TASK" --output-dir "$OUT/hf-scoring"
fi

if [[ "$PART" != hf ]]; then
echo "=== Step 4: vLLM ==="
bash scripts/start_server.sh
python3 benchmark.py --backend server --output-dir "$OUT/vllm"
python3 score_task.py --backend server --task-config "$TASK" --output-dir "$OUT/vllm"

# Step 6, vLLM options (they need the server, so they go here):
# A: python3 benchmark.py --backend server --output-dir "$OUT/vllm-knee" --concurrency 1 2 4 8 16 32 64
# B: python3 benchmark.py --backend server --output-dir "$OUT/vllm-short" --prompts my-short-prompts.txt
#    python3 benchmark.py --backend server --output-dir "$OUT/vllm-long" --prompts my-long-prompts.txt
# D: python3 benchmark.py --backend server --output-dir "$OUT/vllm-mine" --prompts my-prompts.txt --max-tokens 64

bash scripts/stop_server.sh
fi

echo "=== Step 5: Compare ==="
python3 summarize_results.py "$OUT"
if [[ -f "$OUT/hf/eval-answers.jsonl" && -f "$OUT/vllm/eval-answers.jsonl" ]]; then
  python3 show_differences.py "$OUT/hf" "$OUT/vllm"
else
  echo "Run the other part too, then the differences between hf and vllm are shown here."
fi
echo "Done. Results in $OUT"
