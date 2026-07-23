#!/bin/bash
#SBATCH --job-name=WAg_geosite
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WAg_geosite_%A_%a.out
#SBATCH --array=0-10
# ===========================================================================
# WAg — by-SITE geo probe reported in the paper's R² (space analog of the year MC).
# Reuses the EXISTING geo activations (last+mean) — CPU only, no re-extraction.
# 10 merged sites, cap 21, 200 draws, StratifiedKFold-by-site; akk_maximal+eng_tier0.
#   0 qwen3_1b7 1 qwen3_8b 2 qwen3_32b 3 random 4 llama2_7b 5 llama2_13b
#   6 llama2_70b 7 llama2_7b_random 8 llama2_13b_random 9 llama2_70b_random 10 tfidf
#   sbatch v_1/src/world_models/akkadian/sbatch/WAg_geo_site.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b random llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random tfidf)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WAg geo-site ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u v_1/src/world_models/akkadian/probe_geo_site.py --method "$M" \
    || { echo "FAILED geo-site ${M}"; exit 1; }
commit_push "WAg: by-site geo R² ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/probes_geosite/${M}
echo "=== Done ${M}. $(date) ==="
