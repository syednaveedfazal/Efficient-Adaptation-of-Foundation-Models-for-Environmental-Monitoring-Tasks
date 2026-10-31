#!/bin/bash
#SBATCH --job-name=v2_lora010
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v2_lora010_%j.out
#SBATCH --error=logs/v2_lora010_%j.err

# Retrains the plain LoRA r=8 baseline (existing config, unchanged) on split_010pct,
# seed 42, because the v1 checkpoint was deleted. It is the SCOPE v2 warm start and teacher.
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29
export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH=.

python -u scripts/train.py \
    --config configs/prithvi_fcn_lora_r8.yaml \
    --split_json data/splits/seed_42/split_010pct.json \
    --seed 42 \
    --out_dir results/scope_v2/metrics \
    --ckpt_dir results/scope_v2/checkpoints
