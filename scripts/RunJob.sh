#!/bin/bash
#SBATCH --job-name=p29_train         # Name of the job in the queue
#SBATCH --partition=A100short        # A100 (80GB), 8-hour limit — fits 100% budget w/o resume
#SBATCH --gpus=1                     # Request 1 physical GPU
#SBATCH --nodes=1                    # Run on a single compute node
#SBATCH --ntasks-per-node=1          # Number of tasks per node
#SBATCH --cpus-per-task=4            # CPUs requested (PyTorch DataLoader num_workers=4)
#SBATCH --mem=32G                    # Total RAM requested
#SBATCH --time=08:00:00              # Max runtime (A100short max)
#SBATCH --output=logs/job_%j.out     # stdout log (%j = Job ID)
#SBATCH --error=logs/job_%j.err      # stderr log
# NOTE: --partition and --time here are defaults; submit_dinov2.sh overrides them
# on the sbatch command line (env vars PARTITION / TIME), which takes precedence.

# ---------------------------------------------------------------------------
# Generic, plug-and-play training launcher.
#
# Pass the config and (optionally) the split via environment variables at
# submit time — no need to edit this file per run:
#
#   sbatch --export=ALL,CONFIG=configs/dinov2_fcn_lora_r8.yaml,\
#          SPLIT=data/splits/seed_42/split_010pct.json scripts/RunJob.sh
#
#   # 100% budget: omit SPLIT (or set SPLIT="")
#   sbatch --export=ALL,CONFIG=configs/dinov2_fcn_full_ft.yaml scripts/RunJob.sh
#
# Optional: RESUME=<ckpt path>  to resume a time-limited run.
# ---------------------------------------------------------------------------

echo "=========================================================="
echo "Job ID:             $SLURM_JOB_ID"
echo "Running on node:    $SLURMD_NODENAME"
echo "Starting at:        $(date)"
echo "Config:             $CONFIG"
echo "Split:              ${SPLIT:-ALL (100%)}"
echo "Resume:             ${RESUME:-<none>}"
echo "=========================================================="

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29

# Redirect the system temp dir onto the SAME (NFS) filesystem as the project,
# so Lightning's atomic "write temp then os.rename" checkpoint save stays on one
# device. The cluster's /tmp is a local disk (/dev/sda3) while the project lives
# on NFS, so the default causes OSError: [Errno 18] Invalid cross-device link.
# Must ALSO stay short: DataLoader workers create AF_UNIX sockets under TMPDIR,
# and Unix-socket paths are capped at ~108 chars — the deep project path blew
# that limit ("OSError: AF_UNIX path too long"). $HOME/p29_tmp is short AND on
# the same NFS mount as the checkpoints, satisfying both constraints.
export TMPDIR="$HOME/p29_tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Active environment: $CONDA_DEFAULT_ENV"
echo "TMPDIR:             $TMPDIR"
nvidia-smi
echo "=========================================================="

if [ -z "$CONFIG" ]; then
    echo "ERROR: CONFIG environment variable not set."
    echo "Example: sbatch --export=ALL,CONFIG=configs/dinov2_fcn_lora_r8.yaml scripts/RunJob.sh"
    exit 1
fi

# Build argument list dynamically so 100% budget (no SPLIT) works cleanly.
# SEED is needed for the 100% budget (no split JSON to infer the seed from);
# for sub-100% budgets it simply matches the split's own seed (harmless).
# DELETE_CKPT=1 removes the (large) checkpoint dir after metrics are exported —
# essential for big sweeps on a quota-limited NFS home.
ARGS=(--config "$CONFIG")
[ -n "$SPLIT" ]       && ARGS+=(--split_json "$SPLIT")
[ -n "$RESUME" ]      && ARGS+=(--resume "$RESUME")
[ -n "$SEED" ]        && ARGS+=(--seed "$SEED")
[ -n "$DELETE_CKPT" ] && ARGS+=(--delete_ckpt)

echo "Starting: python -u scripts/train.py ${ARGS[*]}"
python -u scripts/train.py "${ARGS[@]}"
status=$?

echo "=========================================================="
echo "Job finished at:    $(date)  (python exit=$status)"
echo "=========================================================="
exit $status
