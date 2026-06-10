#!/bin/bash
#SBATCH --job-name=ft_FT2_q1b7
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/finetune/logs/FT2_%A_%a.out
#SBATCH --array=0-3

# ===========================================================================
# FT2 — PILOT depth ablation: Qwen3-1.7B CPT on Akkadian NTP.
# 28 blocks -> cuts {0, 9, 19, 25} = full / ~33% / ~67% / ~90% depth
# (meeting rule; fig4 peak for 1b7 is hidden-state 9 ~ block 8).
# Full FT of unfrozen blocks, lr 1e-5, 3 epochs over ~9M tokens.
# Needs FT1 data. Checkpoints -> v_1/models/finetune/qwen3_1b7/cutNN/best
# ===========================================================================
set -uo pipefail

CUTS=(0 9 19 25)
CUT="${CUTS[$SLURM_ARRAY_TASK_ID]}"
CUT2=$(printf "%02d" "$CUT")
echo "=== FT2 qwen3_1b7 cut=$CUT ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/finetune/logs v_1/src/finetune/results/train_summaries

OUT=v_1/models/finetune/qwen3_1b7/cut${CUT2}
python -u v_1/src/finetune/train_ntp.py \
    --model-id Qwen/Qwen3-1.7B \
    --unfreeze-from "$CUT" \
    --output-dir "$OUT" \
    --seq-len 2048 --epochs 3 --lr 1e-5 \
    --global-batch 64 --micro-batch 4 \
    || { echo "FAILED: train cut=$CUT"; exit 1; }

cp "$OUT/train_summary.json" \
   v_1/src/finetune/results/train_summaries/qwen3_1b7_cut${CUT2}.json

git add v_1/src/finetune/results/train_summaries/qwen3_1b7_cut${CUT2}.json
git commit -m "finetune FT2: qwen3_1b7 cut${CUT2} train summary (job ${SLURM_JOB_ID:-NA})" || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
