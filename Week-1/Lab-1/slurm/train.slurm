#!/bin/bash
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

CONFIG="${1:-configs/smoke.yaml}"
srun python3 train_tiny_gpt.py --config "${CONFIG}" --compile
