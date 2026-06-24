#!/usr/bin/env python
"""
scripts/run_unet_experiments.py

Orchestrates batch submissions of UNet-from-scratch experiments on the SLURM cluster.
Sweeps across label budgets and seeds (UNet has no adaptation-strategy variants,
since it's trained from scratch rather than adapted from a foundation model).

Supports sequential execution (one-by-one dependency queue) or parallel queueing.
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep UNet-from-scratch experiments on SLURM.")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1, 10, 25, 100],
        help="Data budgets to sweep (matches LABEL_BUDGETS in prepare_dataset.py)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="unet_scratch",
        help="Config basename (without .yaml) in configs/ to use for all runs"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Seeds to sweep"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        default=True,
        help="Run jobs sequentially one-by-one using SLURM job dependencies (default: True)"
    )
    parser.add_argument(
        "--parallel",
        action="store_false",
        dest="sequential",
        help="Run all jobs in parallel (disables sequential dependencies)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show generated batch scripts and commands without submitting"
    )
    return parser.parse_args()


def submit_job(sbatch_content: str, prev_job_id: str = None, dry_run: bool = False) -> str:
    """
    Submits a job to SLURM. If prev_job_id is provided, sets an afterany dependency.
    Returns the submitted Job ID.
    """
    cmd = ["sbatch"]
    if prev_job_id:
        cmd.append(f"--dependency=afterany:{prev_job_id}")

    if dry_run:
        print(f"Would execute: {' '.join(cmd)}")
        print("---------- SBATCH SCRIPT ----------")
        print(sbatch_content.strip())
        print("-----------------------------------\n")
        return "DRY_RUN_ID"

    # Execute sbatch with piped input script
    res = subprocess.run(cmd, input=sbatch_content, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"[ERROR] sbatch submission failed:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse Job ID from stdout: "Submitted batch job 123456"
    match = re.search(r"Submitted batch job (\d+)", res.stdout)
    if not match:
        print(f"[ERROR] Could not parse Job ID from sbatch output:\n{res.stdout}", file=sys.stderr)
        sys.exit(1)

    job_id = match.group(1)
    return job_id


def main():
    args = parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    print("=" * 60)
    print("      UNET-FROM-SCRATCH EXPERIMENT SWEEP ORCHESTRATOR")
    print("=" * 60)
    print(f"Config:      {args.config}.yaml")
    print(f"Budgets:     {args.budgets}%")
    print(f"Seeds:       {args.seeds}")
    print(f"Execution:   {'Sequential (one-by-one dependency)' if args.sequential else 'Parallel'}")
    print(f"Dry-run:     {args.dry_run}")
    print("=" * 60)

    prev_job_id = None
    submit_count = 0

    for budget in args.budgets:
        for seed in args.seeds:
            # 1. Determine split json argument
            if budget == 100:
                split_arg = ""
                budget_str = "100pct"
            else:
                split_path = f"data/splits/seed_{seed}/split_{budget:03d}pct.json"
                split_arg = f"--split_json {split_path}"
                budget_str = f"{budget:03d}pct"

            # 2. Formulate Job Name and Run Name
            # NOTE: must match the run_name format produced by scripts/train.py
            run_name = f"{args.config}_seed_{seed}_split_{budget_str}"
            job_name = f"p29_train_{run_name}"
            metrics_file = Path(f"results/metrics/{run_name}.json")

            # # Check if this experiment has already run and exported metrics
            # if metrics_file.exists():
            #     print(f"Skipping: {args.config} | Seed {seed} | Budget {budget}% (already completed)")
            #     print("-" * 50)
            #     continue

            # 3. Construct the SBATCH Shell Script Content
            delete_arg = "--delete_ckpt" if budget < 100 else ""
            sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=A40devel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# --- Configure temporary directory to avoid node local /tmp partition overflow ---
export TMPDIR="/home/s93nsyed/p29_tmp"
mkdir -p "$TMPDIR"

# --- Initialize Conda environment ---
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate p29

# --- Execute training ---
python -u scripts/train.py \\
    --config configs/{args.config}.yaml \\
    --seed {seed} \\
    {split_arg} \\
    {delete_arg}

# --- Clean up temporary files ---
rm -rf "$TMPDIR"/*
"""

            # 4. Submit the job
            print(f"Queueing: {args.config} | Seed {seed} | Budget {budget}%")
            job_id = submit_job(sbatch_content, prev_job_id=prev_job_id if args.sequential else None, dry_run=args.dry_run)

            if args.dry_run:
                print(f"Queued job dynamically with dry-run placeholder.")
            else:
                print(f"--> Submitted batch job {job_id}" + (f" (depends on {prev_job_id})" if prev_job_id else ""))

            if args.sequential:
                prev_job_id = job_id

            submit_count += 1
            print("-" * 50)

    print(f"\n[ORCHESTRATOR] Successfully processed {submit_count} jobs.")
    if not args.dry_run:
        print("Use 'squeue -u $USER' to monitor the execution queue.")


if __name__ == "__main__":
    main()