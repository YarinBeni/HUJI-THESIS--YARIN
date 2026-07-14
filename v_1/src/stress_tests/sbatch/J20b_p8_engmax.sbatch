#!/bin/bash
#SBATCH --job-name=J20b_p8_engmax
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J20b_p8_engmax_%j.out
# ===========================================================================
# J20b — P8 lambda-probe on the engmaximal cleaning (translations of the
# name-stripped maximal texts; acts = {method}_engmaximal_mean from J17b/c).
# Deck moved the English column from engtier0 (name-confounded: the translator
# re-inserts real king names) to engmaximal. run_acts now MERGES the new
# cleaning into the existing p8_lambda__{method}.json. Resumable per model.
#   sbatch v_1/src/stress_tests/sbatch/J20b_p8_engmax.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J20b P8 engmaximal === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

D=v_1/src/stress_tests/p8_lambda_probe
for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random; do
    F="$D/results/p8_lambda__${M}.json"
    if [ -f "$F" ] && python -c "import json,sys; c=json.load(open('$F'))['cleanings'].get('engmaximal'); sys.exit(0 if c and not c.get('missing') else 1)"; then
        echo "--- P8 engmaximal: $M already done, skipping ---"; continue
    fi
    echo "--- P8 lambda-probe engmaximal: $M ---"
    python -u $D/run_acts.py --method "$M" --cleanings engmaximal --n-jobs 32 \
        || echo "WARN p8 engmax $M failed"
    commit_push "J20b P8: lambda-probe ${M} engmaximal (job ${SLURM_JOB_ID})" \
        "$D/results/p8_lambda__${M}.json"
done
echo "=== Done. $(date) ==="
