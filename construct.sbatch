#!/bin/bash
#SBATCH --job-name=construct
#SBATCH --output=logs/construct.out
#SBATCH --error=logs/construct.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10

set -euo pipefail

module purge || true

if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  conda deactivate || true
fi

python train