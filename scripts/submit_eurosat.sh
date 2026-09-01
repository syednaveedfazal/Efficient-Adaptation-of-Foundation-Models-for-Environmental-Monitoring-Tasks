#!/bin/bash
# ---------------------------------------------------------------------------
# submit_eurosat.sh — batch-submit the full EuroSAT classification matrix.
#
# Sweeps: model × adaptation × label-budget × seed, one SLURM job each, via the
# shared RunJob.sh (unchanged). Each job trains a land-cover classifier, validates
# the best checkpoint, and exports results/metrics/<run>.json (accuracy/macro_f1).
#
# EuroSAT plots are NOT auto-built (plot_results auto-run is burn-scar only);
# after the sweep, run:  python scripts/plot_results.py --dataset eurosat
#
# Usage:
#   MODELS="prithvi dinov2" DELETE_CKPT=1 bash scripts/submit_eurosat.sh   # full 96
#   MODELS="dinov2" ADAPTATIONS="lora_r8" BUDGETS="010" SEEDS="42" bash scripts/submit_eurosat.sh
#
# Axes (override via environment variables):
#   MODELS       prithvi dinov2
#   ADAPTATIONS  lora_r8 lora_r16 linear_probe full_ft
#   BUDGETS      pct codes: 001 010 025 100
#   SEEDS        42 123 456
#
# NOTE: unlike burn-scar, EuroSAT's 100% budget still passes split_100pct.json —
# all tiles share one directory, so the split JSON is what separates train from
# the held-out val set. Never run EuroSAT with an empty SPLIT (it would train on
# the val tiles too).
# ---------------------------------------------------------------------------
set -euo pipefail

MODELS="${MODELS:-prithvi dinov2}"
ADAPTATIONS="${ADAPTATIONS:-lora_r8 lora_r16 linear_probe full_ft vpt}"
BUDGETS="${BUDGETS:-001 010 025 100}"
SEEDS="${SEEDS:-42 123 456}"

PARTITION="${PARTITION:-A100short}"
TIME="${TIME:-08:00:00}"

job_ids=()
n=0
for model in $MODELS; do
    for adapt in $ADAPTATIONS; do
        cfg="configs/eurosat_${model}_${adapt}.yaml"
        if [ ! -f "$cfg" ]; then
            echo "WARNING: config $cfg not found — skipping."
            continue
        fi
        for seed in $SEEDS; do
            for b in $BUDGETS; do
                dc="${DELETE_CKPT:-}"
                # Always pass a split (incl. 100%): the JSON holds the TRAIN-only ids.
                split="data/splits/eurosat/seed_${seed}/split_${b}pct.json"
                export_vars="ALL,CONFIG=${cfg},SPLIT=${split},SEED=${seed},DELETE_CKPT=${dc}"
                
                run_name="eurosat_${model}_${adapt}_seed_${seed}_split_${b}pct"

                # Check if this experiment has already run and exported metrics
                if [ -f "results/metrics/${run_name}.json" ]; then
                    echo "Skipping: ${run_name} (already completed)"
                    continue
                fi

                jid=$(sbatch --parsable --partition="$PARTITION" --time="$TIME" \
                             --export="${export_vars}" scripts/RunJob.sh)
                job_ids+=("$jid")
                n=$((n+1))
                echo "[$jid] model=${model} adapt=${adapt} seed=${seed} budget=${b}%  ${split}"
            done
        done
    done
done

echo "-----------------------------------------------------------"
echo "Submitted ${n} EuroSAT training jobs."
echo "After they finish: python scripts/plot_results.py --dataset eurosat"
echo "Monitor with: squeue -u \$USER"
