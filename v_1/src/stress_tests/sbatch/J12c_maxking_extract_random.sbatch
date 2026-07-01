#!/bin/bash
#SBATCH --job-name=J12c_maxking_extract_random
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J12c_maxking_extract_random_%j.out
# ===========================================================================
# J12c — maximal-with-kings extraction for the RANDOM-weights control
# (Qwen3-8B from_config, seed 42). THE decisive baseline: if random matches the
# trained models on ruler_clf here, the site reads name-token identity, not
# learned structure.
#   sbatch v_1/src/stress_tests/sbatch/J12c_maxking_extract_random.sbatch
# ===========================================================================
set -uo pipefail
M=random
echo "=== J12c maxking-extract ${M} === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/stress_tests/p1_gurnee_tegmark/extract_maxking_acts.py \
    --hfid Qwen/Qwen3-8B --method "$M" --arch causal --random --max-tokens 512 \
    || { echo "FAILED extract ${M}"; exit 1; }

ACTS=v_1/src/linear_probing/results/orcc__embed/activations
commit_push "J12c: maxking acts ${M} (job ${SLURM_JOB_ID})" \
    "${ACTS}/${M}_maxking_mean/metadata.json" "${ACTS}/${M}_maxking_mean/king_coverage.json" \
    "${ACTS}/${M}_maxking_kinglast/metadata.json" "${ACTS}/${M}_maxking_kinglast/king_coverage.json" \
    "${ACTS}/${M}_maxking_kingmean/metadata.json" "${ACTS}/${M}_maxking_kingmean/king_coverage.json"
echo "=== Done ${M}. $(date) ==="
