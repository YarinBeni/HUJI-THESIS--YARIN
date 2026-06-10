#!/bin/bash
#SBATCH --job-name=ft_FT5_q8b_probe
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=v_1/src/finetune/logs/FT5_%j.out

# ===========================================================================
# FT5 — extract + probe the 4 Qwen3-8B checkpoints (after FT4).
# Same structure as FT3. Methods qwen3_8b_ft{00,12,24,32}.
# ===========================================================================
set -uo pipefail
echo "=== FT5 qwen3_8b FT probing ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p v_1/src/finetune/logs v_1/src/finetune/results/probes

ACTS=v_1/src/linear_probing/results/orcc__embed/activations
SUBSET=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
OUT=v_1/src/finetune/results/probes

for CUT2 in 00 12 24 32; do
    CKPT=v_1/models/finetune/qwen3_8b/cut${CUT2}/best
    [ -d "$CKPT" ] || { echo "MISSING checkpoint $CKPT — did FT4 finish?"; exit 1; }
    for CLEAN in tier0 maximal; do
        python -u v_1/src/linear_probing/03_extract_seal_activations.py \
            --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
            --text-col text_${CLEAN} \
            --model "$CKPT" \
            --pooling mean \
            --batch-size 4 \
            --output-dir ${ACTS}/qwen3_8b_ft${CUT2}_${CLEAN}_mean \
            || { echo "FAILED: extract cut${CUT2} ${CLEAN}"; exit 1; }
    done
done

PROBES=""
for CUT2 in 00 12 24 32; do
    m=qwen3_8b_ft${CUT2}
    PROBES="${PROBES:+$PROBES,}${m}_pls,${m}_cls_numeric,${m}_cls"
done
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

python -u v_1/src/finetune/build_scoreboard.py || echo "WARN scoreboard failed"

git add "$OUT"/*__summary.json v_1/src/finetune/results/scoreboard* 2>/dev/null || true
git commit -m "finetune FT5: qwen3_8b FT probe summaries + scoreboard (job ${SLURM_JOB_ID:-NA})" || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
