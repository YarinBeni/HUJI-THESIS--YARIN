#!/bin/bash
#SBATCH --job-name=WA2_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WA2_probe_%A_%a.out
#SBATCH --array=0-10
# ===========================================================================
# WA2 — ridge probes over the Akkadian activations (CPU). One task per method;
# each does both variants x {r8,r40} x {year,geo} x {last,mean}. Run after the
# matching WA1* extraction landed. Randoms cleanup their npz after probing.
#   0 qwen3_1b7  1 qwen3_8b  2 qwen3_32b  3 gpt_oss_120b  4 random
#   5 llama2_7b  6 llama2_13b  7 llama2_70b
#   8 llama2_7b_random*  9 llama2_13b_random*  10 llama2_70b_random*
#   sbatch v_1/src/world_models/akkadian/sbatch/WA2_probe.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b random llama2_7b llama2_13b \
         llama2_70b llama2_7b_random llama2_13b_random llama2_70b_random)
CLEAN=(0 0 0 0 1 0 0 0 1 1 1)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"; C="${CLEAN[$SLURM_ARRAY_TASK_ID]}"
echo "=== WA2 probe ${M} (cleanup=${C}) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

EXTRA=""; [ "$C" = "1" ] && EXTRA="--cleanup"
python -u v_1/src/world_models/akkadian/probe_akk.py --method "$M" $EXTRA \
    || { echo "FAILED probe ${M}"; exit 1; }

commit_push "WA2: akk probes ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/probes/${M}
echo "=== Done ${M}. $(date) ==="
