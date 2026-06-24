# P29 — Efficient Adaptation of Foundation Models for Environmental Monitoring

Comparing parameter-efficient fine-tuning (PEFT) strategies on the **Prithvi-EO-2.0-300M** geospatial foundation model for burn-scar segmentation from satellite imagery.

**Task:** Binary semantic segmentation (burned / not burned) on 512×512 HLS satellite chips  
**Dataset:** ~800 labelled scenes from NASA IMPACT (6 spectral bands, 30 m/pixel)  
**Primary method:** LoRA r=8 — only 1.9% of parameters trained, matches full fine-tuning performance

---

## Results so far

| Model | Label budget | Best val burn IoU |
|---|---|---|
| **UNet (scratch)** | 100% | 0.696 |
| **Prithvi + LoRA r=8** | 1% | 0.39 |
| **Prithvi + LoRA r=8** | 10% | 0.67 |
| **Prithvi + LoRA r=8** | 100% | **0.81** |

---

## Folder structure

```
project/
│
├── configs/                    # One YAML file per model/strategy
│   ├── unet_baseline.yaml      # UNet trained from scratch
│   └── prithvi_lora_r8.yaml   # Prithvi + LoRA r=8 (primary method)
│
├── data/
│   ├── raw/
│   │   └── hls_burn_scars/     # Downloaded satellite images + masks
│   │       ├── training/       # ~540 scene-mask pairs
│   │       └── validation/     # ~264 scene-mask pairs
│   ├── processed/
│   │   └── stats.json          # Per-band mean/std for normalisation
│   └── splits/                 # Label budget subsets (pre-generated)
│       ├── seed_42/
│       │   ├── split_001pct.json   # 1%  of training scenes
│       │   ├── split_005pct.json
│       │   ├── split_010pct.json   # 10% of training scenes
│       │   ├── split_025pct.json
│       │   ├── split_050pct.json
│       │   └── split_100pct.json
│       ├── seed_123/           # Same budgets, different random seed
│       ├── seed_456/           # Same budgets, third random seed
│       └── val_scenes.json     # Fixed validation set (never changes)
│
├── logs/                       # SLURM job logs (stdout + stderr)
│   └── job_<ID>.out / .err
│
├── models/
│   └── pretrained/
│       ├── prithvi/            # Prithvi-EO-2.0-300M backbone
│       │   ├── Prithvi_EO_V2_300M.pt   # ~1.2 GB weights
│       │   ├── prithvi_mae.py          # Model architecture source
│       │   └── config.json
│       └── dinov2/             # DINOv2-Base (future baseline)
│
├── notebooks/                  # Jupyter notebooks for analysis/plots
│
├── results/
│   ├── checkpoints/            # Saved model checkpoints
│   │   └── prithvi_lora_r8_seed_42_split_010pct/
│   │       └── epoch=55-val/
│   │           └── burn_iou=0.6695.ckpt
│   ├── metrics/                # Exported CSV/JSON results
│   └── plots/                  # Figures for the report
│
├── scripts/
│   ├── ActivateGPU.sh          # SLURM job script — edit and sbatch to train
│   ├── train.py                # Main training entry point
│   ├── download_data.py        # Download HLS burn scars dataset
│   ├── download_models.py      # Download Prithvi / DINOv2 weights
│   └── prepare_dataset.py      # Generate stats.json and split JSONs
│
├── src/
│   ├── datasets/
│   │   └── burn_scar.py        # Dataset + DataModule for HLS chips
│   ├── models/
│   │   ├── registry.py         # Maps config name → model class
│   │   ├── unet.py             # UNet from scratch (baseline)
│   │   └── prithvi_seg.py      # Prithvi + LoRA segmentor (SegDecoder)
│   ├── training/
│   │   ├── module.py           # PyTorch Lightning training loop (shared)
│   │   ├── losses.py           # CE + Dice combined loss
│   │   └── metrics.py          # IoU / burn_iou / bg_iou
│   └── evaluation/
│       └── metrics.py          # Evaluation utilities
│
├── environment.yml             # Conda environment (install with this)
└── requirements.txt            # pip packages
```

---

## Setup

### 1. Install the conda environment

```bash
conda env create -f environment.yml
conda activate p29
```

### 2. Download data and models (first time only)

```bash
python scripts/download_data.py
python scripts/download_models.py --model prithvi
```

### 3. Prepare dataset splits

```bash
python scripts/prepare_dataset.py
```

This generates `data/processed/stats.json` and all the split JSON files under `data/splits/`.

---

## How to connect VS Code to the HPC cluster

This lets you edit files, browse the project, and run terminals — all inside VS Code on your laptop, with the code actually running on the HPC server.

### Step 1 — Install the Remote SSH extension

Open VS Code → Extensions (Ctrl+Shift+X) → search **Remote - SSH** → Install

### Step 2 — Add the HPC server

Press `Ctrl+Shift+P` → type **Remote-SSH: Open SSH Configuration File** → select it.

Add this block (replace with your actual login node address):

```
Host bender-hpc
    HostName bender.hpc.uni-bonn.de
    User s93nsyed
    ForwardAgent yes
```

### Step 3 — Connect

Press `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → select `bender-hpc` → enter your password.

VS Code reopens connected to the server. Now open the project folder:

**File → Open Folder** → navigate to:
```
/home/s93nsyed/Efficient-Adaptation-of-Foundation-Models-for-Environmental-Monitoring-Tasks
```

### Step 4 — Select the conda environment as the Python interpreter

Press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one that says `p29`.

If it's not listed, paste the path manually:
```
/home/s93nsyed/.conda/envs/p29/bin/python
```

You can now edit code with full IntelliSense, run terminals (`Ctrl+\`` `), and open Jupyter notebooks directly in VS Code.

---

## How to run a training job

### Option A — Submit to the GPU cluster (recommended)

Edit `scripts/ActivateGPU.sh` — change the `SPLIT` variable to choose your label budget:

```bash
# 10% of training data, seed 42
SPLIT="data/splits/seed_42/split_010pct.json"

# Different seed, same budget
SPLIT="data/splits/seed_123/split_010pct.json"

# 100% of training data (omit the --split_json line)
```

Then submit:

```bash
sbatch scripts/ActivateGPU.sh
```

Monitor live output:

```bash
# Find your job ID
squeue -u $USER

# Watch the log
tail -f logs/job_<ID>.out
```

### Option B — Run locally (small tests, no GPU needed)

```bash
conda activate p29
python scripts/train.py --config configs/prithvi_lora_r8.yaml \
    --split_json data/splits/seed_42/split_001pct.json
```

---

## Plug-and-play: switching models

The entire pipeline is config-driven. To switch models, just change `--config`:

```bash
# UNet baseline
python scripts/train.py --config configs/unet_baseline.yaml

# Prithvi + LoRA r=8
python scripts/train.py --config configs/prithvi_lora_r8.yaml
```

To add a new model: create `src/models/your_model.py`, register it in `src/models/registry.py`, write a config YAML. Nothing else changes.

---

## Checking results

Checkpoint filenames encode the metric — no script needed to read results:

```
results/checkpoints/prithvi_lora_r8_seed_42_split_010pct/
  epoch=55-val/
    burn_iou=0.6695.ckpt    ← val burn IoU = 0.6695 at epoch 55
```

Compare all runs at once:

```bash
find results/checkpoints -name "*.ckpt" | sort
```

### Resume a job that was cancelled

If a job hits the SLURM time limit, resume it from the latest checkpoint:

```bash
# In ActivateGPU.sh, add:
RESUME="results/checkpoints/<run_name>/epoch=<N>-val/burn_iou=<X>.ckpt"

python -u scripts/train.py \
    --config     configs/prithvi_lora_r8.yaml \
    --split_json "$SPLIT" \
    --resume     "$RESUME"
```

---

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Backbone | Prithvi-EO-2.0-300M | Trained on same HLS sensor as our data |
| Adaptation | LoRA r=8 (PEFT) | 1.9% trainable params, matches full fine-tuning |
| LoRA targets | attn.qkv, attn.proj, mlp.fc1, mlp.fc2 | All attention + FFN layers in 24 ViT blocks |
| Decoder | 4× conv+upsample | Reshapes 32×32 patch tokens back to 512×512 |
| Loss | CE + Dice (50/50) | Dice handles class imbalance (burn scars are rare) |
| Class weights | [1.0, 8.0] | Burn scar ~8× rarer than background |
