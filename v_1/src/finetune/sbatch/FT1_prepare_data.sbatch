#!/bin/bash
#SBATCH --job-name=ft_FT1_data
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/finetune/logs/FT1_%j.out

# ===========================================================================
# FT1 — build the Akkadian NTP fine-tune corpus (Task 5).
# Fragment texts from unified train/val splits (tier0), written to
# v_1/data/finetune/. CPU-only, ~2 min. Run before FT2/FT4/FT6.
# ===========================================================================
set -uo pipefail
echo "=== FT1 prepare NTP data ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/finetune/logs

python -u v_1/src/finetune/prepare_ntp_data.py \
    || { echo "FAILED: prepare_ntp_data.py"; exit 1; }

git add v_1/data/finetune/metadata.json 2>/dev/null || true
git commit -m "finetune FT1: NTP data metadata (job ${SLURM_JOB_ID:-NA})" || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
