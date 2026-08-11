#!/bin/bash
#SBATCH --job-name=F28_ladder
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-31
#SBATCH --output=v_1/src/phase2/erasure/logs/F28_ladder_%A_%a.out
# ===========================================================================
# F28 — the single-concept erasure ladder (reviewer request): LEACE one
# variable at a time from the document representations and re-run the full
# E1 read-out. Rungs: ruler (era ladder's top, ICC=1 caveat in the script
# docstring), provenance (find-spot), length, year10 (coarse era bins ~
# period — the POSITIVE CONTROL: erasing it must crush any real chronology).
# Each rung reports raw vs erased pairwise macro + grouped ridge rho + a
# concept-probe manipulation check.
# ===========================================================================
set -uo pipefail
CONCEPTS=(ruler provenance length year10)
METHODS=(olmo2_7b qwen3_8b olmo2_7b_random tfidf_char)
VARS=(eng_tier0 akk_maximal)
CONCEPT=${CONCEPTS[$((SLURM_ARRAY_TASK_ID / 8))]}
METHOD=${METHODS[$(((SLURM_ARRAY_TASK_ID % 8) / 2))]}
VARIANT=${VARS[$((SLURM_ARRAY_TASK_ID % 2))]}
echo "=== F28 ${CONCEPT}/${METHOD}/${VARIANT} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/erasure/logs

python -u v_1/src/phase2/erasure/e4_confounders.py \
    --method "${METHOD}" --variant "${VARIANT}" --concept "${CONCEPT}" \
    || { echo "FAILED ladder ${CONCEPT}/${METHOD}/${VARIANT}"; exit 1; }

commit_push "F28: erasure ladder ${CONCEPT}/${METHOD}/${VARIANT} (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/erasure/results
echo "=== Done. $(date) ==="
