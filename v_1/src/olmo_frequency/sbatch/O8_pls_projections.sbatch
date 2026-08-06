#!/bin/bash
#SBATCH --job-name=O8_pls_proj
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/world_models/logs/O8_pls_proj_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# O8 — per-entity PLS predictions for cell A, so the frequency dose-response can
# be asked under the PLS read-out as well as under ridge.
#
#   0 olmo2_7b        1 olmo2_7b_random
#
# WHY THIS EXISTS. The dose-response figure correlates each entity's exposure with
# each entity's dating error, so it needs a prediction PER ENTITY. probe_wm.py writes
# those for ridge (results/projections/); probe_eng_pls.py recorded only aggregate
# r2/spearman per k, which cannot be decomposed back to entities. It now also writes
# the best-k predictions to results/projections_pls/, which is what this job produces.
#
# That matters because the deck reports PLS for the fragment cells and ridge for cell
# A; a frequency claim read only under ridge is a claim about one of the two read-outs
# the thesis uses.
#
# CPU only, no re-extraction: it walks the cell-A npz already on this filesystem.
#
#   sbatch v_1/src/olmo_frequency/sbatch/O8_pls_projections.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b olmo2_7b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== O8 pls projections ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python -u v_1/src/world_models/probe_eng_pls.py --method "$M" \
    || { echo "FAILED eng_pls ${M}"; exit 1; }

N=$(ls v_1/src/world_models/results/projections_pls/${M}/*.csv.gz 2>/dev/null | wc -l)
echo "[check] ${M}: ${N} PLS projection files"
[ "$N" -gt 0 ] || { echo "FAILED: ${M} produced no PLS projections"; exit 1; }

commit_push "O8: cell-A PLS per-entity projections ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/results/eng_pls/${M} \
    v_1/src/world_models/results/projections_pls/${M}
echo "=== Done ${M}. $(date) ==="
