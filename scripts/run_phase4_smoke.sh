#!/bin/bash
#SBATCH --job-name=scope_phase4
#SBATCH --partition=A100medium
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/phase4_smoke_%j.out
#SBATCH --error=logs/phase4_smoke_%j.err

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29

export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

export PYTHONPATH=.

OUT_DIR="results/scope_smoke/metrics"
mkdir -p "$OUT_DIR"
mkdir -p logs
SPLIT="data/splits/seed_42/split_001pct.json"

echo "=========================================================="
echo "Phase 4 Smoke Test: 6 configs on split_001pct, 2 epochs"
echo "Starting at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "=========================================================="

configs=(
    "configs/prithvi_fcn_lora_r8.yaml"
    "configs/scope/b1_random.yaml"
    "configs/scope/s1_share.yaml"
    "configs/scope/s2_gate.yaml"
    "configs/scope/s3_exit.yaml"
    "configs/scope/s4_full.yaml"
)

for cfg in "${configs[@]}"; do
    echo "----------------------------------------------------------"
    echo "Running smoke test for config: $cfg"
    python -u scripts/train.py \
        --config "$cfg" \
        --split_json "$SPLIT" \
        --max_epochs 2 \
        --batch_size 2 \
        --seed 42 \
        --out_dir "$OUT_DIR" \
        --delete_ckpt
    status=$?
    echo "Finished $cfg with exit code $status"
    if [ $status -ne 0 ]; then
        echo "ERROR: Run failed for $cfg"
        exit $status
    fi
done

echo "=========================================================="
echo "All smoke tests completed at: $(date)"
echo "=========================================================="

