#!/bin/bash
#SBATCH --job-name=ft_FT4_q8b
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/finetune/logs/FT4_%A_%a.out
#SBATCH --array=0-3

# ===========================================================================
# FT4 — MAIN depth ablation: Qwen3-8B CPT on Akkadian NTP.
# 36 blocks -> cuts {0, 12, 24, 32} (fig4 peak at hidden-state 16 ~ block 15
# sits between cuts 12 and 24). Full FT of unfrozen blocks; 2x H100 because
# the cut=0 arm needs ~96GB for fp32 AdamW states (device_map auto splits).
# Submit AFTER the FT3 gate passes. Needs FT1 data.
# ===========================================================================
set -uo pipefail

CUTS=(0 12 24 32)
CUT="${CUTS[$SLURM_ARRAY_TASK_ID]}"
CUT2=$(printf "%02d" "$CUT")
echo "=== FT4 qwen3_8b cut=$CUT ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/finetune/logs v_1/src/finetune/results/train_summaries

OUT=v_1/models/finetune/qwen3_8b/cut${CUT2}
python -u v_1/src/finetune/train_ntp.py \
    --model-id Qwen/Qwen3-8B \
    --unfreeze-from "$CUT" \
    --output-dir "$OUT" \
    --seq-len 2048 --epochs 3 --lr 1e-5 \
    --global-batch 64 --micro-batch 2 \
    || { echo "FAILED: train cut=$CUT"; exit 1; }

cp "$OUT/train_summary.json" \
   v_1/src/finetune/results/train_summaries/qwen3_8b_cut${CUT2}.json

git add v_1/src/finetune/results/train_summaries/qwen3_8b_cut${CUT2}.json
git commit -m "finetune FT4: qwen3_8b cut${CUT2} train summary (job ${SLURM_JOB_ID:-NA})" || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
