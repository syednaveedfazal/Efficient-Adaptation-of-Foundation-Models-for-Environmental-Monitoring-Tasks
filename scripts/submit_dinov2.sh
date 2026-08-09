#!/bin/bash
# ---------------------------------------------------------------------------
# submit_dinov2.sh — batch-submit the full DINOv2 experiment matrix to SLURM.
#
# Sweeps: adaptation × label-budget × seed, one SLURM job each, via RunJob.sh.
# Each job trains, validates the best checkpoint, exports results/metrics/<run>.json,
# and regenerates results/plots/. After all training jobs are submitted, a final
# plotting job (BuildPlots.sh) is queued with a dependency so the combined
# comparison graph is rebuilt automatically once everything finishes.
#
# DINOv2 uses only the 3 RGB bands of each scene (see src/models/dinov2_fcn.py).
#
# Usage:
#   bash scripts/submit_dinov2.sh                          # full matrix (default below)
#   ADAPTATIONS="lora_r8" BUDGETS="001 010" SEEDS="42" bash scripts/submit_dinov2.sh
#
# Axes (override via environment variables):
#   ADAPTATIONS  lora_r8 lora_r16 linear_probe full_ft
#   BUDGETS      pct codes: 001 010 025 100
#   SEEDS        42 123 456
# ---------------------------------------------------------------------------
set -euo pipefail

ADAPTATIONS="${ADAPTATIONS:-lora_r8 lora_r16 linear_probe full_ft}"
BUDGETS="${BUDGETS:-001 010 025 100}"
SEEDS="${SEEDS:-42 123 456}"

# Cluster resources — override via env. A100short = A100 80GB, 8-hour limit.
PARTITION="${PARTITION:-A100short}"
TIME="${TIME:-08:00:00}"

declare -A CONFIG_FOR=(
    [lora_r8]="configs/dinov2_fcn_lora_r8.yaml"
    [lora_r16]="configs/dinov2_fcn_lora_r16.yaml"
    [linear_probe]="configs/dinov2_fcn_linear_probe.yaml"
    [full_ft]="configs/dinov2_fcn_full_ft.yaml"
)

job_ids=()
n=0
for adapt in $ADAPTATIONS; do
    cfg="${CONFIG_FOR[$adapt]:-}"
    if [ -z "$cfg" ]; then
        echo "WARNING: unknown adaptation '$adapt' — skipping."
        continue
    fi
    for seed in $SEEDS; do
        for b in $BUDGETS; do
            # DELETE_CKPT=1 (env) → auto-delete checkpoints after metrics export.
            dc="${DELETE_CKPT:-}"
            if [ "$b" = "100" ]; then
                # 100% budget: no split JSON — SEED tells train.py which seed to use.
                export_vars="ALL,CONFIG=${cfg},SPLIT=,SEED=${seed},DELETE_CKPT=${dc}"
                split_desc="ALL (100%)"
            else
                split="data/splits/seed_${seed}/split_${b}pct.json"
                export_vars="ALL,CONFIG=${cfg},SPLIT=${split},SEED=${seed},DELETE_CKPT=${dc}"
                split_desc="$split"
            fi
            jid=$(sbatch --parsable --partition="$PARTITION" --time="$TIME" \
                         --export="${export_vars}" scripts/RunJob.sh)
            job_ids+=("$jid")
            n=$((n+1))
            echo "[$jid] adapt=${adapt} seed=${seed} budget=${b}%  ${split_desc}"
        done
    done
done

# Final plot-build trigger: runs after ALL training jobs finish (success or not).
if [ "$n" -gt 0 ]; then
    dep=$(IFS=:; echo "${job_ids[*]}")
    plot_jid=$(sbatch --parsable --dependency=afterany:"${dep}" scripts/BuildPlots.sh)
    echo "-----------------------------------------------------------"
    echo "Submitted ${n} training jobs."
    echo "[$plot_jid] BuildPlots — runs after all training jobs finish."
else
    echo "No jobs submitted."
fi

echo "Monitor with: squeue -u \$USER"
