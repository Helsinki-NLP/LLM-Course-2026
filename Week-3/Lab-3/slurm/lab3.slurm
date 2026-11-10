#!/bin/bash
#SBATCH --job-name=lab3
#SBATCH --account=project_2021893
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

# Optional: the core lab steps as an unattended job.
# Usage: sbatch slurm/lab3.slurm [LAB2_CONFIG]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: not a GH200 node. Submit from ssh roihu-gpu.csc.fi." >&2
  exit 1
fi
bash scripts/run_experiment.sh "results/job-${SLURM_JOB_ID}" "${1:-../../Week-2/Lab-2/configs/roihu.json}"
