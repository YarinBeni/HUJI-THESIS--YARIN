#!/bin/bash
#SBATCH --job-name=P1a_tokaudit
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/chronorank/autopsy/logs/P1a_%j.out

# ===========================================================================
# Pillar 1 — 1a Tokenization audit (CPU). Isolates factor (T).
# Per tokenizer (Thalesian/uMT5, Qwen3-8B, gpt-oss-120b), per corpus
# (orcc/seal/letters/archibab/oracc_1mill/ebl): fertility, UNK, byte-fallback,
# char-probe, ORCC category split + tier0-vs-maximal detail. Tokenizers are
# already in the HF cache from prior extraction runs.
# ===========================================================================
set -uo pipefail
echo "=== P1a tokenization audit ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p v_1/src/chronorank/autopsy/logs

python -u v_1/src/chronorank/autopsy/tokenization_audit.py --sample 3000 \
    || { echo "FAILED: tokenization_audit"; exit 1; }

git add v_1/src/chronorank/autopsy/results/tokenization_audit.json \
        v_1/src/chronorank/autopsy/results/tokenization_audit.csv \
        v_1/src/chronorank/autopsy/results/figures/*.png 2>/dev/null || true
git commit -m "P1a: tokenization audit (fertility/UNK/byte per tokenizer x corpus) (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
