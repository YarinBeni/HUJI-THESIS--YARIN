#!/bin/bash
#SBATCH --job-name=WA1b_extract_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WA1b_extract_gptoss_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# WA1b — gpt-oss-120B extraction for the Akkadian mimic (gpu:8). One task per
# text variant.  sbatch v_1/src/world_models/akkadian/sbatch/WA1b_extract_gptoss.sbatch
# ===========================================================================
set -uo pipefail
VARIANTS=(akk_maximal eng_maximal); V="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"; M=gpt_oss_120b
echo "=== WA1b extract ${M}/${V} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/world_models/akkadian/extract_akk.py --method "$M" --variant "$V" --batch-size 8 \
    || { echo "FAILED extract ${M}/${V}"; exit 1; }

commit_push "WA1b: akk acts ${M}/${V} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/activations/${M}/${V}/metadata.json
echo "=== Done ${M}/${V}. $(date) ==="
