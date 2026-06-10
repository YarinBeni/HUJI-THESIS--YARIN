#!/bin/bash
#SBATCH --job-name=ft_FT0b_gptoss_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/finetune/logs/FT0b_%j.out

# ===========================================================================
# FT0b — balanced MC probes (200 draws) for gpt-oss-120b BASE, tier0+maximal,
# all layers. CPU-only (activations from FT0). The new method 'gpt_oss_120b'
# is auto-registered by run_mc_probes.py from the activation dirs on disk.
# Output is ISOLATED in v_1/src/finetune/results/probes (tags mc_balanced /
# mc_balanced_maximal) — never collides with canonical Round-3 results.
# ===========================================================================
set -uo pipefail
echo "=== FT0b gpt-oss-120b base probes ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p v_1/src/finetune/logs v_1/src/finetune/results/probes

SUBSET=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
OUT=v_1/src/finetune/results/probes
PROBES="gpt_oss_120b_pls,gpt_oss_120b_cls_numeric,gpt_oss_120b_cls"

for CLEAN in tier0 maximal; do
    TAG=mc_balanced; [ "$CLEAN" = "maximal" ] && TAG=mc_balanced_maximal
    python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
        --draws-matrix   "$SUBSET/draws_matrix.npy" \
        --fragment-order "$SUBSET/corpus_fragment_order.json" \
        --output-dir     "$OUT" \
        --method-tag     "$TAG" \
        --probes         "$PROBES" \
        --cleaning       "$CLEAN" \
        --pooling        mean \
        --layers         all \
        --pls-k          1,2,3,5 \
        --n-jobs         "${SLURM_CPUS_PER_TASK:-32}" \
        || { echo "FAILED: probes ${CLEAN}"; exit 1; }
done

git add "$OUT"/*__summary.json 2>/dev/null || true
git commit -m "finetune FT0b: gpt-oss-120b base probe summaries (job ${SLURM_JOB_ID:-NA})" || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
