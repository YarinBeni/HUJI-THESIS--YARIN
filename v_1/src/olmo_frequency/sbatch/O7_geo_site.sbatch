#!/bin/bash
#SBATCH --job-name=O7_geo_site
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/O7_geo_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# O7 — WAg's by-site geo probe for the OLMo arms: predict the find-spot's
# latitude/longitude from whole fragments, 200 draws holding out whole sites.
#
#   0 olmo2_7b        1 olmo2_7b_random
#
# Fills the two dashes OLMo currently shows on the deck's fragment-place slides
# (EN gloss and raw Akkadian). CPU only — walks the npz O4 already extracted.
# Same script and settings as WAg gave the other arms.
#
#   sbatch v_1/src/olmo_frequency/sbatch/O7_geo_site.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b olmo2_7b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== O7 geo-site ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python -u v_1/src/world_models/akkadian/probe_geo_site.py --method "$M" \
    || { echo "FAILED geo-site ${M}"; exit 1; }

N=$(ls v_1/src/world_models/akkadian/results/probes_geosite/${M}/*.json 2>/dev/null | wc -l)
echo "[check] ${M}: ${N} geo-site files"
[ "$N" -gt 0 ] || { echo "FAILED: ${M} produced no geo-site files"; exit 1; }

commit_push "O7: by-site geo R2 ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/probes_geosite/${M}
echo "=== Done ${M}. $(date) ==="
