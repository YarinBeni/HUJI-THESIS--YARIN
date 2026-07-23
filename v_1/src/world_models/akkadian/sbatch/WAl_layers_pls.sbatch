#!/bin/bash
#SBATCH --job-name=WAl_layers_pls
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WAl_layers_pls_%A_%a.out
#SBATCH --array=0-9
# ===========================================================================
# WAl — per-layer R² curve + best-layer PLS-k sweep (k up to 64) over the EXISTING
# Akkadian acts (eng_tier0 + akk_maximal, year+geo, last+mean). CPU, no re-extract.
#   0 qwen3_1b7 1 qwen3_8b 2 qwen3_32b 3 random 4 llama2_7b 5 llama2_13b
#   6 llama2_70b 7 llama2_7b_random 8 llama2_13b_random 9 llama2_70b_random
#   sbatch v_1/src/world_models/akkadian/sbatch/WAl_layers_pls.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b random llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WAl layers+pls ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u v_1/src/world_models/akkadian/probe_layers_pls.py --method "$M" \
    || { echo "FAILED layers_pls ${M}"; exit 1; }
commit_push "WAl: per-layer R² + best-layer PLS-k ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/layers_pls/${M}
echo "=== Done ${M}. $(date) ==="
