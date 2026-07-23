#!/bin/bash
#SBATCH --job-name=WA2t_tier0
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WA2t_tier0_%A_%a.out
#SBATCH --array=0-10
# ===========================================================================
# WA2t — ridge probes (holdout/mc/loro) over the NEW eng_tier0 activations only.
# Same method map as WA2; --variant eng_tier0 leaves the akk_maximal/eng_maximal
# results untouched. Launch after WA1t/WA1dt land. Index 3 (gpt_oss) has no akk
# acts and no-ops.
#   0 qwen3_1b7 1 qwen3_8b 2 qwen3_32b 3 gpt_oss_120b 4 random
#   5 llama2_7b 6 llama2_13b 7 llama2_70b 8 llama2_7b_random 9 llama2_13b_random
#   10 llama2_70b_random
#   sbatch --array=0-2,4-10 v_1/src/world_models/akkadian/sbatch/WA2t_probe_tier0.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b random llama2_7b llama2_13b \
         llama2_70b llama2_7b_random llama2_13b_random llama2_70b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WA2t probe ${M}/eng_tier0 === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u v_1/src/world_models/akkadian/probe_akk.py --method "$M" --variant eng_tier0 \
    || { echo "FAILED probe ${M}/eng_tier0"; exit 1; }
commit_push "WA2t: akk probes ${M}/eng_tier0 (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/probes/${M}
echo "=== Done ${M}/eng_tier0. $(date) ==="
