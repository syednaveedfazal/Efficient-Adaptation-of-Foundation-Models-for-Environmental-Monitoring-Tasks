#!/bin/bash
#SBATCH --job-name=spaco_eval
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/spaco_eval_%x_%j.out
#SBATCH --error=logs/spaco_eval_%x_%j.err

# Score a checkpoint on the given sets. Usage: sbatch -J <tag> scripts/spaco/eval.sh <config> <ckpt> <out_json> <sets...>
CFG=$1; CKPT=$2; OUT=$3; shift 3
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29
export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH=.
python -u scripts/spaco/evaluate.py --config "$CFG" --ckpt "$CKPT" --out_json "$OUT" --sets "$@"
