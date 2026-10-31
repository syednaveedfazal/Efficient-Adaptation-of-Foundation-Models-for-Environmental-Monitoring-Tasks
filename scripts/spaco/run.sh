#!/bin/bash
#SBATCH --job-name=spaco
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/spaco_%x_%j.out
#SBATCH --error=logs/spaco_%x_%j.err

# Train one SpaCo study run (configs/spaco/<name>.yaml) on split_010pct with a training seed, then score dev.
# Usage: sbatch -J <name>_s<seed> scripts/spaco/run.sh <name> <seed>
NAME=$1; SEED=$2
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29
export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH=.

RUN="${NAME}_s${SEED}_010pct"
python -u scripts/train.py \
    --config "configs/spaco/${NAME}.yaml" \
    --split_json data/splits/seed_42/split_010pct.json \
    --seed "$SEED" \
    --run_name "$RUN" \
    --out_dir results/spaco/metrics \
    --ckpt_dir results/spaco/checkpoints || exit 1

BEST=$(python -c "import json;print(json.load(open('results/spaco/metrics/${RUN}.json'))['training']['best_ckpt'])")
python -u scripts/spaco/evaluate.py --config "configs/spaco/${NAME}.yaml" --ckpt "$BEST" \
    --sets dev --out_json "results/spaco/eval/${RUN}.json"
