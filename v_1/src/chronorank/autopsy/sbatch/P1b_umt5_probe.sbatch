#!/bin/bash
#SBATCH --job-name=P1b_umt5_probe
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=v_1/src/chronorank/autopsy/logs/P1b_%j.out

# ===========================================================================
# Pillar 1 — 1b Control-ladder probe (THE cheap star). NO training.
#  (1) GPU: extract vanilla google/umt5-base on ORCC (tier0+maximal, MEAN, all
#      encoder layers) with the SAME seq2seq-encoder extractor that produced the
#      on-disk Thalesian activations (round2_phase3/extract_enc_activations.py).
#  (2) CPU: probe the full ladder under the identical maximal-balanced PLS:
#      umt5_base (new) + thalesian_cunei400m/akk300m + qwen3_8b + gpt_oss_120b
#      + random.  Decision regime = maximal+mean; tier0+mean = robustness column.
#  (3) Build the factor table -> ladder_table.csv + T/A/O/F decision reads.
# Ladder: Thalesian vs uMT5 -> (F);  uMT5 vs Qwen3-8B -> (A)+(T).
# ===========================================================================
set -uo pipefail
echo "=== P1b umt5 control-ladder ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p v_1/src/chronorank/autopsy/logs v_1/src/chronorank/autopsy/results/probes

ACTS=v_1/src/linear_probing/results/orcc__embed/activations
SUBSET=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
OUT=v_1/src/chronorank/autopsy/results/probes

# ---- 1) extract vanilla uMT5-base (GPU, encoder-only, mean pool) -----------
for CLEAN in tier0 maximal; do
    OUTDIR=${ACTS}/umt5_base_${CLEAN}_mean
    if ls "$OUTDIR"/layer_*.npz >/dev/null 2>&1; then
        echo "umt5_base ${CLEAN} already extracted — skip"; continue
    fi
    python -u v_1/src/linear_probing/round2_phase3/extract_enc_activations.py \
        --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
        --text-col text_${CLEAN} \
        --model google/umt5-base \
        --pooling mean \
        --batch-size 16 \
        --output-dir "$OUTDIR" \
        || { echo "FAILED: extract umt5_base ${CLEAN}"; exit 1; }
done

# ---- 2) probe the ladder (CPU), PLS year, mean pooling, all layers --------
PROBES=umt5_base_pls,thalesian_cunei400m_pls,thalesian_akk300m_pls,qwen3_8b_pls,gpt_oss_120b_pls,random_pls
echo "probes: $PROBES"

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

# ---- 3) factor table + decision -------------------------------------------
python -u v_1/src/chronorank/autopsy/build_ladder_table.py \
    --probes-dir "$OUT" \
    --out v_1/src/chronorank/autopsy/results \
    || echo "WARN ladder table failed"

git add "$OUT"/*__summary.json \
        v_1/src/chronorank/autopsy/results/ladder_table.csv \
        ${ACTS}/umt5_base_tier0_mean/metadata.json \
        ${ACTS}/umt5_base_maximal_mean/metadata.json 2>/dev/null || true
git commit -m "P1b: vanilla uMT5 extracted + control-ladder PLS + factor table (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
