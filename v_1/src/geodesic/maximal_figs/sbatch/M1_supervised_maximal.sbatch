#!/bin/bash
#SBATCH --job-name=maxfig_M1
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/geodesic/maximal_figs/logs/M1_%j.out

# ===========================================================================
# M1 — maximal / mean / balanced supervised probes + fig1(A,C,D), fig2(A,B),
#      fig4(A). 8 models x {PLS year, Ridge year, ruler-CLS}, ALL layers.
#
# Reuses the tested balanced-MC machinery (run_mc_probes.py) with --cleaning
# maximal and --layers all. Writes an ISOLATED method-tag + dir so it never
# collides with the canonical tier0 `mc_balanced` results.
#
# CPU-only (activations already extracted). Bottleneck = qwen3_32b all-layers
# x 200 draws; if it walltimes, re-run JUST 32b with --probes qwen3_32b_pls,
# qwen3_32b_cls_numeric,qwen3_32b_cls into the same dir/tag, then re-run the
# plotting step at the bottom.
# ===========================================================================
set -uo pipefail
echo "=== M1 maximal supervised ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

MF=v_1/src/geodesic/maximal_figs
SUBSET=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
mkdir -p "$MF/probes" "$MF/logs"

# 8 models x 3 probes
PROBES=""
for m in tfidf mlm thalesian_akk300m thalesian_cunei400m qwen3_1b7 qwen3_8b qwen3_32b random; do
  PROBES="${PROBES:+$PROBES,}${m}_pls,${m}_cls_numeric,${m}_cls"
done
echo "probes: $PROBES"

python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix   "$SUBSET/draws_matrix.npy" \
    --fragment-order "$SUBSET/corpus_fragment_order.json" \
    --output-dir     "$MF/probes" \
    --method-tag     mc_balanced_maximal \
    --probes         "$PROBES" \
    --cleaning       maximal \
    --pooling        mean \
    --layers         all \
    --pls-k          1,2,3,5 \
    --n-jobs         "${SLURM_CPUS_PER_TASK:-32}"

echo ""; echo "--- rendering fig1(A,C,D) / fig2(A,B) / fig4(A) ---"
python -u "$MF/make_maximal_figs.py" \
    --probes-dir "$MF/probes" --tag mc_balanced_maximal \
    --tables-out "$MF/tables" --fig-out "$MF/figures"

echo ""; echo "=== Committing ==="
git add "$MF/figures"/*.png "$MF/tables"/*.csv "$MF/make_maximal_figs.py" \
        "$MF/README.md" "$MF/sbatch"/*.sbatch
git commit -m "maximal_figs M1: supervised probes + fig1(ACD)/fig2(AB)/fig4(A) (job ${SLURM_JOB_ID:-NA})" \
    || echo "Nothing to commit"
git pull --rebase origin main || echo "WARN pull failed"
git push origin main || echo "WARN push failed — outputs on cluster fs."
echo "=== Done. $(date) ==="
