#!/bin/bash
#SBATCH --job-name=tefu_with_div
#SBATCH --output=logs/tefu_with_div.out
#SBATCH --error=logs/tefu_with_div.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10

set -euo pipefail

module purge || true

if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  conda deactivate || true
fi

make train