#!/bin/bash
#SBATCH --job-name=v2_lora_eval
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/v2_lora_eval_%j.out
#SBATCH --error=logs/v2_lora_eval_%j.err
# Re-evaluate the plain LoRA r=8 10% checkpoint with the v2 evaluator and FLOP convention.
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29
export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH=.
python -u scripts/scope_v2/evaluate_v2.py --plain --config configs/prithvi_fcn_lora_r8.yaml \
    --ckpt results/scope_v2/checkpoints/lora_r8_seed42_split_010pct.ckpt --out_json results/scope_v2/eval/lora.json
