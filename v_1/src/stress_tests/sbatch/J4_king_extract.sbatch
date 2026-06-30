#!/bin/bash
#SBATCH --job-name=J4_king
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J4_king_%A_%a.out
#SBATCH --array=0-5
# ===========================================================================
# J4 — king-token activation extraction (tier0). Array: one task per GPU-light
# model. gpt-oss-120B is a separate job (J4b, needs gpu:4).
#   0 qwen3_1b7 (causal)   1 qwen3_8b (causal)   2 qwen3_32b (causal)
#   3 thalesian_akk300m (encoder)  4 thalesian_cunei400m (encoder)  5 umt5_base (encoder)
# Writes king_last/king_mean acts (gitignored *.npz, cluster-local) + committed
# king_coverage.json + metadata.json. Probing is J6 (CPU).
#   sbatch v_1/src/stress_tests/sbatch/J4_king_extract.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b thalesian_akk300m thalesian_cunei400m umt5_base)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B Thalesian/AKK_300m Thalesian/cuneiformBase-400m google/umt5-base)
ARCHS=(causal causal causal encoder encoder encoder)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"; H="${HFIDS[$SLURM_ARRAY_TASK_ID]}"; A="${ARCHS[$SLURM_ARRAY_TASK_ID]}"

echo "=== J4 king-extract ${M} (${H}, ${A}) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/stress_tests/p1_gurnee_tegmark/extract_king_acts.py \
    --hfid "$H" --method "$M" --arch "$A" --max-tokens 512 \
    || { echo "FAILED extract ${M}"; exit 1; }

ACTS=v_1/src/linear_probing/results/orcc__embed/activations
git add "${ACTS}/${M}_tier0_kinglast/metadata.json" "${ACTS}/${M}_tier0_kinglast/king_coverage.json" \
        "${ACTS}/${M}_tier0_kingmean/metadata.json" "${ACTS}/${M}_tier0_kingmean/king_coverage.json" 2>/dev/null || true
git commit -m "J4: king-token acts ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" || echo "nothing to commit"
for _i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 8 + 5)); done
echo "=== Done ${M}. $(date) ==="
