#!/bin/bash
#SBATCH --job-name=lab3
#SBATCH --account=project_2021893
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Usage: sbatch slurm/lab3.slurm [LAB2_CONFIG]
# for example: sbatch slurm/lab3.slurm ../../Week-2/Lab-2/configs/my-task.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: this job is not on a GH200 node. Submit it from ssh roihu-gpu.csc.fi." >&2
  exit 1
fi

TASK_CONFIG="${1:-../../Week-2/Lab-2/configs/roihu.json}"
bash scripts/run_experiment.sh "results/lab3-${SLURM_JOB_ID}" "$TASK_CONFIG"
