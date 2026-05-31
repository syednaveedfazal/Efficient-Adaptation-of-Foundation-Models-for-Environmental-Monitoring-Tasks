#!/bin/bash
#SBATCH --job-name=p29_unet
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/job_%j.out     # Path to stdout log (%j is replaced by Job ID)
#SBATCH --error=logs/job_%j.err      # Path to stderr log

echo "=========================================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"
echo "=========================================================="

# Conda
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29
# Navigate to project root
cd /home/s93nsyed/Efficient-Adaptation-of-Foundation-Models-for-Environmental-Monitoring-Tasks
echo "Env: $CONDA_DEFAULT_ENV"
nvidia-smi
echo "=========================================================="

# Step 1: Prepare dataset (skipped if already done)
if [ ! -f data/processed/stats.json ]; then
    echo "Running prepare_dataset.py ..."
    python -u scripts/prepare_dataset.py \
        --raw_dir   data/raw/hls_burn_scars \
        --out_dir   data/processed \
        --split_dir data/splits
else
    echo "Dataset already prepared, skipping."
fi

echo "=========================================================="
echo "Starting UNet training ..."

python -u scripts/train.py \
    --config     configs/unet_baseline.yaml \
    --split_json data/splits/seed_42/split_100pct.json

echo "=========================================================="
echo "Finished: $(date)"
echo "=========================================================="
