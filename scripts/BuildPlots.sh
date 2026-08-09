#!/bin/bash
#SBATCH --job-name=p29_plots         # Build the comparison figures from results/metrics/
#SBATCH --partition=A100devel        # CPU-only matplotlib job; short devel queue is fine
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Regenerates results/plots/ from every results/metrics/*.json. Typically
# submitted by scripts/submit_dinov2.sh with a dependency on all training jobs,
# so the combined comparison graph is built automatically once they finish.
# Can also be run standalone at any time.

echo "Building result plots at $(date)"
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29

python -u scripts/plot_results.py
echo "Done at $(date)"
