#!/bin/bash
#SBATCH --job-name=ft_FT4b_q32b
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=36:00:00
#SBATCH --output=v_1/src/finetune/logs/FT4b_%A_%a.out
#SBATCH --array=0-3

# ===========================================================================
# FT4b — depth ablation: Qwen3-32B CPT on Akkadian NTP (completeness across
# the full Qwen scale ladder 1.7B/8B/32B).
# 64 blocks -> cuts {0, 21, 43, 58} = 0/33/67/90% of depth (base fig4 peak is
# early, ~L6 maximal). 8x H100: the cut=0 full-FT arm needs the 32B bf16
# weights (~64GB) + fp32 AdamW states (~256GB) + grads — exceeds 4 GPUs, so
# size for the worst case; upper-cut arms simply under-use. Needs FT1 data.
# ===========================================================================
set -uo pipefail

CUTS=(0 21 43 58)
CUT="${CUTS[$SLURM_ARRAY_TASK_ID]}"
CUT2=$(printf "%02d" "$CUT")
echo "=== FT4b qwen3_32b cut=$CUT ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/finetune/logs v_1/src/finetune/results/train_summaries
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=v_1/models/finetune/qwen3_32b/cut${CUT2}
python -u v_1/src/finetune/train_ntp.py \
    --model-id Qwen/Qwen3-32B \
    --unfreeze-from "$CUT" \
    --output-dir "$OUT" \
    --seq-len 2048 --epochs 3 --lr 1e-5 \
    --global-batch 64 --micro-batch 1 \
    || { echo "FAILED: train cut=$CUT"; exit 1; }

cp "$OUT/train_summary.json" \
   v_1/src/finetune/results/train_summaries/qwen3_32b_cut${CUT2}.json

git add v_1/src/finetune/results/train_summaries/qwen3_32b_cut${CUT2}.json
git commit -m "finetune FT4b: qwen3_32b cut${CUT2} train summary (job ${SLURM_JOB_ID:-NA})" || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
