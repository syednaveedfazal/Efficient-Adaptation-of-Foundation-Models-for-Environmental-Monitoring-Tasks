#!/bin/bash
#SBATCH --job-name=v2_diag
#SBATCH --partition=A100devel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/v2_diag_%j.out
#SBATCH --error=logs/v2_diag_%j.err

# Phase 1 diagnostics on the v1 s4_full checkpoint (evaluation only, no training).
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29
export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH=.

python -u scripts/scope_v2/diagnose.py "$@"
