#!/bin/bash
#SBATCH --job-name=v2_run
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/v2_run_%x_%j.out
#SBATCH --error=logs/v2_run_%x_%j.err

# Train one SCOPE v2 run (Stage B, warm start + distill) on split_010pct, then evaluate its best checkpoint.
# Usage: sbatch -J <name> scripts/scope_v2/run_v2.sh <name> [keep:share ...]
NAME=$1; shift
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29
export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH=.

CFG="configs/scope_v2/${NAME}.yaml"
RUN="prithvi_scope_${NAME}_seed_42_split_010pct"
python -u scripts/train.py \
    --config "$CFG" \
    --split_json data/splits/seed_42/split_010pct.json \
    --seed 42 \
    --out_dir results/scope_v2/metrics \
    --ckpt_dir results/scope_v2/checkpoints || exit 1

BEST=$(python -c "import json;print(json.load(open('results/scope_v2/metrics/${RUN}.json'))['training']['best_ckpt'])")
python -u scripts/scope_v2/evaluate_v2.py --config "$CFG" --ckpt "$BEST" \
    --out_json "results/scope_v2/eval/${NAME}.json" --settings "$@"
