#!/bin/bash
#SBATCH --job-name=lab3-serve-measure
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/lab3-%j.out
#SBATCH --error=logs/lab3-%j.err
#SBATCH --account=project_2021893

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"

# These short defaults validate the workflow. Override the partition and time
# on the sbatch command line for the complete benchmark.
bash scripts/run_roihu_experiment.sh "results/roihu-${SLURM_JOB_ID}"
