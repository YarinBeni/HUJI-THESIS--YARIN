#!/bin/bash
#SBATCH --job-name=train_mlm
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/linear_probing/logs/train_mlm_%j.out

echo "=== Retrain Akkadian MLM ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

# --- Environment setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/lititure-review

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/models/baseline_retrained

# --- Verify training data ---
echo "=== Verifying training data ==="
python3 -c "
import pandas as pd, json, sys
from pathlib import Path

train = pd.read_parquet('v_1/data/training_ready/train_fragments.parquet')
val   = pd.read_parquet('v_1/data/training_ready/val_fragments.parquet')
eval_ = pd.read_parquet('v_1/data/training_ready/eval_subset.parquet')
with open('v_1/data/training_ready/vocab.json') as f:
    vocab = json.load(f)
vocab_size = vocab['vocab_size']

print(f'  train fragments : {len(train):,}')
print(f'  val fragments   : {len(val):,}')
print(f'  eval subset     : {len(eval_):,}')
print(f'  vocab size      : {vocab_size:,}')

assert len(train) > 30000, f'Expected >30k train fragments, got {len(train)}'
assert len(val)   >  3000, f'Expected >3k val fragments, got {len(val)}'
assert vocab_size > 10000, f'Expected vocab_size>10k, got {vocab_size}'
print('  [OK] training data verified')
" || { echo "FAILED: training data verification"; exit 1; }

# --- Train ---
echo "=== Starting training (output -> v_1/models/baseline_retrained/) ==="
python v_1/src/archive/baseline_mlm/02_train.py \
    --data_dir   v_1/data/training_ready \
    --output_dir v_1/models/baseline_retrained \
    --epochs     10 \
    --batch_size 16 \
    --lr         3e-4 \
    --seed       42 \
    --num_workers 4 \
    || { echo "FAILED: MLM training"; exit 1; }

# --- Verify checkpoint ---
echo "=== Verifying checkpoint ==="
python3 -c "
from pathlib import Path
import torch

ckpt = Path('v_1/models/baseline_retrained/baseline_best.pt')
assert ckpt.exists(), f'Checkpoint not found: {ckpt}'
data = torch.load(ckpt, map_location='cpu')
print(f'  best val_loss : {data[\"val_loss\"]:.4f}')
print(f'  epoch         : {data[\"epoch\"]}')
print(f'  [OK] checkpoint verified')
if data['val_loss'] > 3.020:
    print(f'  WARNING: val_loss {data[\"val_loss\"]:.4f} does not beat previous 3.020')
else:
    print(f'  [PASS] val_loss beats previous 3.020')
" || { echo "FAILED: checkpoint verification"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
