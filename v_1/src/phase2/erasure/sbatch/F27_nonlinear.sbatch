#!/bin/bash
#SBATCH --job-name=F27_nl
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=v_1/src/phase2/erasure/logs/F27_nl_%A_%a.out
# ===========================================================================
# F27 — nonlinear probes (kernel-RBF + MLP), the E4 leftover. Retires the
# standing caveat "a nonlinear document-time code would not be caught".
# GroupKFold-by-ruler OOF; pairs scored only where BOTH rulers are held out.
# Pre-registered rules in the script docstring.
# ===========================================================================
set -uo pipefail
ARMS=(olmo2_7b qwen3_8b olmo2_7b_random tfidf_char)
VARS=(eng_tier0 akk_maximal)
METHOD=${ARMS[$((SLURM_ARRAY_TASK_ID / 2))]}
VARIANT=${VARS[$((SLURM_ARRAY_TASK_ID % 2))]}
echo "=== F27 nonlinear ${METHOD}/${VARIANT} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/erasure/logs

python -u v_1/src/phase2/erasure/e4_nonlinear.py \
    --method "${METHOD}" --variant "${VARIANT}" \
    || { echo "FAILED nonlinear ${METHOD}/${VARIANT}"; exit 1; }

commit_push "F27: nonlinear probes ${METHOD}/${VARIANT} (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/erasure/results
echo "=== Done. $(date) ==="
