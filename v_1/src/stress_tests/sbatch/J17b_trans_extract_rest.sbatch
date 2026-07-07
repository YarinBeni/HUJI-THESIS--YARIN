#!/bin/bash
#SBATCH --job-name=J17b_trans_extract_rest
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=v_1/src/stress_tests/logs/J17b_trans_extract_rest_%A_%a.out
# ===========================================================================
# J17b — embed the English translations with the REMAINING ladder models
# (J17 covered qwen3_1b7 / qwen3_8b / random). Translations are already
# committed (translation/translations.parquet); this is extraction only.
#   0 qwen3_32b   1 thalesian_akk300m   2 thalesian_cunei400m   3 umt5_base
# gpt-oss = J17c (gpu:8). Note: the Thalesian encoders were trained on Akkadian
# input, so embedding English with them is out-of-domain — informative anyway
# (the translator embedding its own output).
#   sbatch v_1/src/stress_tests/sbatch/J17b_trans_extract_rest.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_32b thalesian_akk300m thalesian_cunei400m umt5_base)
HFIDS=(Qwen/Qwen3-32B Thalesian/AKK_300m Thalesian/cuneiformBase-400m google/umt5-base)
ARCHS=(causal encoder encoder encoder)
i=${SLURM_ARRAY_TASK_ID:-0}
M=${METHODS[$i]}; H=${HFIDS[$i]}; A=${ARCHS[$i]}
echo "=== J17b trans-extract ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/stress_tests/translation/extract_translation_acts.py \
    --hfid "$H" --method "$M" --arch "$A" || { echo "FAIL extract ${M}"; exit 1; }
echo "=== Done ${M}. $(date) ==="
