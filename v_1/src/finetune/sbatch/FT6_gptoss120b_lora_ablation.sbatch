#!/bin/bash
#SBATCH --job-name=ft_FT6_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=v_1/src/finetune/logs/FT6_%A_%a.out
#SBATCH --array=0-3

# ===========================================================================
# FT6 — gpt-oss-120b depth ablation via LoRA-above-the-cut.
# 36 blocks -> cuts {0, 12, 24, 32}. MXFP4 -> bf16 dequantized (~240GB across
# 8x H100, naive model parallel via device_map=auto). LoRA r=32 on attention
# projections of blocks >= cut, lr 2e-4, 3 epochs (~7.7M gpt-oss tokens/epoch).
# Full FT of a 120B MoE is out of budget; this preserves the depth design.
# Submit AFTER FT3 gate + FT0 (base curve). Needs FT1 data.
# ===========================================================================
set -uo pipefail

CUTS=(0 12 24 32)
CUT="${CUTS[$SLURM_ARRAY_TASK_ID]}"
CUT2=$(printf "%02d" "$CUT")
echo "=== FT6 gpt-oss-120b LoRA cut=$CUT ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/finetune/logs v_1/src/finetune/results/train_summaries

python -c "import peft" 2>/dev/null || pip install -q peft

OUT=v_1/models/finetune/gpt_oss_120b/cut${CUT2}
python -u v_1/src/finetune/train_ntp.py \
    --model-id openai/gpt-oss-120b \
    --unfreeze-from "$CUT" \
    --lora --lora-r 32 --lora-alpha 64 \
    --output-dir "$OUT" \
    --seq-len 2048 --epochs 3 --lr 2e-4 \
    --global-batch 32 --micro-batch 1 \
    || { echo "FAILED: train cut=$CUT"; exit 1; }

cp "$OUT/train_summary.json" \
   v_1/src/finetune/results/train_summaries/gpt_oss_120b_cut${CUT2}.json

git add v_1/src/finetune/results/train_summaries/gpt_oss_120b_cut${CUT2}.json
git commit -m "finetune FT6: gpt-oss-120b cut${CUT2} LoRA train summary (job ${SLURM_JOB_ID:-NA})" || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
