#!/bin/bash
#SBATCH --job-name=scope_phase5
#SBATCH --partition=A100medium
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase5_exp_%j.out
#SBATCH --error=logs/phase5_exp_%j.err

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29

export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

export PYTHONPATH=.

OUT_DIR="results/scope_smoke/metrics"
CKPT_DIR="results/scope_smoke/checkpoints"
mkdir -p "$OUT_DIR"
mkdir -p "$CKPT_DIR"
mkdir -p logs

SPLIT="data/splits/seed_42/split_010pct.json"

echo "=========================================================="
echo "Phase 5 Mini Experiment on split_010pct (54 scenes)"
echo "Starting at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "=========================================================="

# Check if checkpoint exists to resume
CKPT=$(find "$CKPT_DIR/prithvi_scope_s4_full_seed_42_split_010pct" -name "*.ckpt" 2>/dev/null | head -n 1)
RESUME_ARG=""
if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
    echo "Resuming s4_full from $CKPT"
    RESUME_ARG="--resume $CKPT"
fi

# 1. Train SCOPE-full
echo "----------------------------------------------------------"
echo "1. Training SCOPE-full (configs/scope/s4_full.yaml)"
echo "----------------------------------------------------------"
python -u scripts/train.py \
    --config configs/scope/s4_full.yaml \
    --split_json "$SPLIT" \
    --max_epochs 40 \
    --batch_size 4 \
    --seed 42 \
    --out_dir "$OUT_DIR" \
    --ckpt_dir "$CKPT_DIR" \
    $RESUME_ARG

status=$?
if [ $status -ne 0 ]; then
    echo "ERROR: s4_full training failed with exit $status"
    exit $status
fi

# Find best checkpoint for s4_full
BEST_CKPT=$(find "$CKPT_DIR/prithvi_scope_s4_full_seed_42_split_010pct" -name "*.ckpt" | head -n 1)
echo "Best checkpoint found: $BEST_CKPT"

# 2. Run inference sweep over (keep_ratio, share_frac)
if [ -n "$BEST_CKPT" ] && [ -f "$BEST_CKPT" ]; then
    echo "----------------------------------------------------------"
    echo "2. Running inference grid sweep on best checkpoint"
    echo "----------------------------------------------------------"
    python -u scripts/evaluate_sweep.py \
        --config configs/scope/s4_full.yaml \
        --split_json "$SPLIT" \
        --ckpt "$BEST_CKPT" \
        --out_json "results/scope_smoke/sweep_results.json"
fi

# 3. Train SCOPE Ablation (no prior)
echo "----------------------------------------------------------"
echo "3. Training SCOPE Ablation No-Prior (configs/scope/ablation_no_prior.yaml)"
echo "----------------------------------------------------------"
python -u scripts/train.py \
    --config configs/scope/ablation_no_prior.yaml \
    --split_json "$SPLIT" \
    --max_epochs 40 \
    --batch_size 4 \
    --seed 42 \
    --out_dir "$OUT_DIR" \
    --ckpt_dir "$CKPT_DIR" \
    --delete_ckpt

status=$?
if [ $status -ne 0 ]; then
    echo "ERROR: ablation_no_prior training failed with exit $status"
    exit $status
fi

# 4. Generate Pareto Plot
echo "----------------------------------------------------------"
echo "4. Generating Pareto Plot"
echo "----------------------------------------------------------"
python -u scripts/plot_pareto.py

echo "=========================================================="
echo "Phase 5 experiment completed successfully at: $(date)"
echo "=========================================================="
