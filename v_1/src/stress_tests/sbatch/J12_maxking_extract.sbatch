#!/bin/bash
#SBATCH --job-name=J12_maxking_extract
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J12_maxking_extract_%A_%a.out
#SBATCH --array=0-5
# ===========================================================================
# J12 — activation extraction for the "maximal-with-kings" config. Array: one
# GPU-light model per task (gpt-oss = J12b gpu:4, random = J12c). Writes all THREE
# pooling sites on the maximal_keepking cleaning (mean/king_last/king_mean share
# the same maximal text) under the "_maxking_*" activation dirs.
#   0 qwen3_1b7 (causal)   1 qwen3_8b (causal)   2 qwen3_32b (causal)
#   3 thalesian_akk300m (encoder)  4 thalesian_cunei400m (encoder)  5 umt5_base (encoder)
# *.npz are gitignored/cluster-local; committed = king_coverage.json + metadata.json.
# Probe is J13 (CPU).
#   sbatch v_1/src/stress_tests/sbatch/J12_maxking_extract.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b thalesian_akk300m thalesian_cunei400m umt5_base)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B Thalesian/AKK_300m Thalesian/cuneiformBase-400m google/umt5-base)
ARCHS=(causal causal causal encoder encoder encoder)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"; H="${HFIDS[$SLURM_ARRAY_TASK_ID]}"; A="${ARCHS[$SLURM_ARRAY_TASK_ID]}"

echo "=== J12 maxking-extract ${M} (${H}, ${A}) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/stress_tests/p1_gurnee_tegmark/extract_maxking_acts.py \
    --hfid "$H" --method "$M" --arch "$A" --max-tokens 512 \
    || { echo "FAILED extract ${M}"; exit 1; }

ACTS=v_1/src/linear_probing/results/orcc__embed/activations
commit_push "J12: maxking acts ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    "${ACTS}/${M}_maxking_mean/metadata.json" "${ACTS}/${M}_maxking_mean/king_coverage.json" \
    "${ACTS}/${M}_maxking_kinglast/metadata.json" "${ACTS}/${M}_maxking_kinglast/king_coverage.json" \
    "${ACTS}/${M}_maxking_kingmean/metadata.json" "${ACTS}/${M}_maxking_kingmean/king_coverage.json"
echo "=== Done ${M}. $(date) ==="
