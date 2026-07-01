#!/bin/bash
#SBATCH --job-name=J12b_maxking_extract_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J12b_maxking_extract_gptoss_%j.out
# ===========================================================================
# J12b — maximal-with-kings extraction for gpt-oss-120B (needs gpu:4; device_map
# =auto in extract_lib). Same three sites as J12.
#   sbatch v_1/src/stress_tests/sbatch/J12b_maxking_extract_gptoss.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b
echo "=== J12b maxking-extract ${M} === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/stress_tests/p1_gurnee_tegmark/extract_maxking_acts.py \
    --hfid openai/gpt-oss-120b --method "$M" --arch causal --max-tokens 512 \
    || { echo "FAILED extract ${M}"; exit 1; }

ACTS=v_1/src/linear_probing/results/orcc__embed/activations
commit_push "J12b: maxking acts ${M} (job ${SLURM_JOB_ID})" \
    "${ACTS}/${M}_maxking_mean/metadata.json" "${ACTS}/${M}_maxking_mean/king_coverage.json" \
    "${ACTS}/${M}_maxking_kinglast/metadata.json" "${ACTS}/${M}_maxking_kinglast/king_coverage.json" \
    "${ACTS}/${M}_maxking_kingmean/metadata.json" "${ACTS}/${M}_maxking_kingmean/king_coverage.json"
echo "=== Done ${M}. $(date) ==="
