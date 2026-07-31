#!/bin/bash
#SBATCH --job-name=WB4_entity_pls
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WB4_entity_pls_%A_%a.out
#SBATCH --array=0-14
# ===========================================================================
# WB4 — best-layer PLS-k sweep (k=1..64) for the CELL-B entity surfaces, so the PLS
# figure is comparable across entity(salient) / entity(obscure) / fragment(eng_tier0) /
# fragment(akk_maximal). Reuses the committed best layer; CPU, no re-extraction.
#   sbatch v_1/src/world_models/akkadian/sbatch/WB4_entity_pls.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b random \
         llama2_7b llama2_13b llama2_70b \
         llama2_7b_random llama2_13b_random llama2_70b_random \
         thalesian_akk300m thalesian_cunei400m umt5_base tfidf)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WB4 entity PLS ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u v_1/src/world_models/akkadian/probe_entity_pls.py --method "$M" \
    || { echo "FAILED entity_pls ${M}"; exit 1; }
commit_push "WB4: entity-level PLS-k sweep ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/probes_entity_pls
echo "=== Done ${M}. $(date) ==="
