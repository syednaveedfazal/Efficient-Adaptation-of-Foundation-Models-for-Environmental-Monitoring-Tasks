#!/bin/bash
#SBATCH --job-name=p29_train_unet     # Name of the job in the queue
#SBATCH --partition=A40devel         # Partition (development queue, max 1 hour)
#SBATCH --gpus=1                     # Request 1 physical GPU
#SBATCH --nodes=1                    # Run on a single compute node
#SBATCH --ntasks-per-node=1          # Number of tasks per node
#SBATCH --cpus-per-task=4            # CPUs requested (useful for PyTorch DataLoader num_workers=4)
#SBATCH --mem=32G                    # Total RAM requested
#SBATCH --time=01:00:00              # Max runtime — 1 hr (A40devel max)
#SBATCH --output=logs/job_%j.out     # Path to stdout log (%j is replaced by Job ID)
#SBATCH --error=logs/job_%j.err      # Path to stderr log

# --- 1. Print diagnostic metadata for debugging ---
echo "=========================================================="
echo "Job ID:             $SLURM_JOB_ID"
echo "Running on node:    $SLURMD_NODENAME"
echo "Starting at:        $(date)"
echo "=========================================================="

# --- 2. Initialize Conda for non-interactive shell ---
export TMPDIR="/home/s93nsyed/p29_tmp"
mkdir -p "$TMPDIR"

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29

echo "Active environment: $CONDA_DEFAULT_ENV"
echo "GPU Status:"
nvidia-smi
echo "=========================================================="

# --- 3. Execute training ---
echo "Starting Python execution for Prithvi + UNet decoder..."

# Plug-and-play: change SPLIT to switch seed/budget, then sbatch this file.
#   data/splits/seed_42/split_001pct.json   (1%,   ~5 chips)
#   data/splits/seed_42/split_010pct.json   (10%, ~54 chips)
#   data/splits/seed_42/split_050pct.json   (50%)
#   data/splits/seed_123/split_010pct.json  (10%, seed 123)
#   data/splits/seed_456/split_010pct.json  (10%, seed 456)
# For 100% budget: comment out the --split_json line below
SPLIT="data/splits/seed_42/split_010pct.json"

python -u scripts/train.py \
    --config configs/prithvi_unet_lora_r8.yaml \
    # --split_json "$SPLIT"

# --- Clean up temporary files ---
rm -rf "$TMPDIR"/*

echo "=========================================================="
echo "Job finished at:    $(date)"
echo "=========================================================="
