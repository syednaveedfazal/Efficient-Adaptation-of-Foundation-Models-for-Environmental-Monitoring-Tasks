#!/bin/bash
#SBATCH --job-name=p29_eval          # Name of the job in the queue
#SBATCH --partition=A40devel         # Partition (development queue, max 1 hour)
#SBATCH --gpus=1                     # Request 1 physical GPU
#SBATCH --nodes=1                    # Run on a single compute node
#SBATCH --ntasks-per-node=1          # Number of tasks per node
#SBATCH --cpus-per-task=4            # CPUs requested (useful for PyTorch DataLoader num_workers=4)
#SBATCH --mem=32G                    # Total RAM requested
#SBATCH --time=00:20:00              # Validation-only — a few minutes per checkpoint
#SBATCH --output=logs/job_%j.out     # Path to stdout log (%j is replaced by Job ID)
#SBATCH --error=logs/job_%j.err      # Path to stderr log

# --- 1. Print diagnostic metadata for debugging ---
echo "=========================================================="
echo "Job ID:             $SLURM_JOB_ID"
echo "Running on node:    $SLURMD_NODENAME"
echo "Starting at:        $(date)"
echo "=========================================================="

# --- 2. Initialize Conda for non-interactive shell ---
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29

echo "Active environment: $CONDA_DEFAULT_ENV"
echo "GPU Status:"
nvidia-smi
echo "=========================================================="

# --- 3. Backfill results/metrics/ for the completed Prithvi+LoRA r8 runs ---
echo "Evaluating 1% budget (best: epoch=21, burn_iou=0.3934)..."
python -u scripts/evaluate.py \
    --config     configs/prithvi_lora_r8.yaml \
    --split_json data/splits/seed_42/split_001pct.json \
    --ckpt       "results/checkpoints/prithvi_lora_r8_seed_42_split_001pct/epoch=21-val/burn_iou=0.3934.ckpt"

echo "Evaluating 10% budget (best: epoch=55, burn_iou=0.6695)..."
python -u scripts/evaluate.py \
    --config     configs/prithvi_lora_r8.yaml \
    --split_json data/splits/seed_42/split_010pct.json \
    --ckpt       "results/checkpoints/prithvi_lora_r8_seed_42_split_010pct/epoch=55-val/burn_iou=0.6695.ckpt"

echo "=========================================================="
echo "Job finished at:    $(date)"
echo "=========================================================="
