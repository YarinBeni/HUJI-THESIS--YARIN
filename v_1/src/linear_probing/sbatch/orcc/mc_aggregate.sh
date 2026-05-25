#!/bin/bash
#SBATCH --job-name=mc_agg
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=v_1/src/linear_probing/logs/mc_agg_%x_%j.out
# Rebuild the full summary JSONs for one model from ALL 200 draw files, then
# commit + push everything. Submit with --dependency=afterok:<all chunk jobs>.
#   --export=MODEL=qwen3_32b

set -euo pipefail
export OMP_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARN: git pull failed"

PROBES="${PROBES:-${MODEL}_pls,${MODEL}_cls,${MODEL}_cls_numeric}"
echo "=== MC aggregate: ${MODEL} (${PROBES}) ==="

# All draw files already exist -> every draw is SKIPPED, only summaries rebuilt
# from the complete 0..199 set.
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix   v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy \
    --fragment-order v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json \
    --output-dir     v_1/src/linear_probing/results/orcc_round2_phase0/probes \
    --probes         "$PROBES" \
    --layers         all \
    --draws-range    "0-199" \
    --n-jobs         1 \
    || { echo "FAILED: aggregate ${MODEL}"; exit 1; }

# Report draw-file counts as a completeness check.
for P in pls cls cls_numeric; do
    N=$(ls v_1/src/linear_probing/results/orcc_round2_phase0/probes/${MODEL}_${P}__mc_balanced__draw*.json 2>/dev/null | wc -l)
    echo "  ${MODEL}_${P}: ${N}/200 draws"
done

git add v_1/src/linear_probing/results/orcc_round2_phase0/probes/${MODEL}_*
git commit -m "Phase E1 MC (parallel fan-out): ${MODEL} balanced probes (job ${SLURM_JOB_ID})" \
    || echo "Nothing new to commit"
git push origin main || echo "WARN: git push failed"

echo "=== Aggregate done: ${MODEL}  End: $(date) ==="
