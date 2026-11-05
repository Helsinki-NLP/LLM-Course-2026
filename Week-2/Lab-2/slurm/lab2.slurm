#!/bin/bash
#SBATCH --job-name=lab2-lora
#SBATCH --account=project_2021893
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"

module purge
module load python-pytorch/2.13

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 was not found after 'module load python-pytorch/2.13'." >&2
  echo "Submit the job from the GPU side: ssh USERNAME@roihu-gpu.csc.fi" >&2
  echo "(the browser Jupyter terminal runs on x86 and cannot submit this job)." >&2
  exit 1
fi

# Downloaded models and datasets go to scratch, not to the home directory.
export HF_HOME="${HF_HOME:-/scratch/project_2021893/${USER}/hf}"
mkdir -p "${HF_HOME}"

CONFIG="${1:-configs/roihu.json}"
srun python3 run_experiment.py --config "${CONFIG}" --fresh
