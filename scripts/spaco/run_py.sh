#!/bin/bash
#SBATCH --job-name=spaco_py
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/spaco_py_%x_%j.out
#SBATCH --error=logs/spaco_py_%x_%j.err
# Run a SpaCo analysis script on a GPU node. Usage: sbatch -J <tag> scripts/spaco/run_py.sh <script.py> [args...]
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29
export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH=.
python -u "$@"
