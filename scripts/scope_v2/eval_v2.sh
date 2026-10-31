#!/bin/bash
#SBATCH --job-name=v2_eval
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/v2_eval_%x_%j.out
#SBATCH --error=logs/v2_eval_%x_%j.err
# Re-evaluate the best checkpoint of an already trained SCOPE v2 run.
# Usage: sbatch -J <name> scripts/scope_v2/eval_v2.sh <name> [keep:share ...]
NAME=$1; shift
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29
export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH=.
RUN="prithvi_scope_${NAME}_seed_42_split_010pct"
BEST=$(python -c "import json;print(json.load(open('results/scope_v2/metrics/${RUN}.json'))['training']['best_ckpt'])")
python -u scripts/scope_v2/evaluate_v2.py --config "configs/scope_v2/${NAME}.yaml" --ckpt "$BEST" \
    --out_json "results/scope_v2/eval/${NAME}.json" --settings "$@"
