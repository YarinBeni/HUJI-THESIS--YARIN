#!/bin/bash
#SBATCH --job-name=J20_p8_lambda
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J20_p8_lambda_%j.out
# ===========================================================================
# J20 — P8 supervision-dial spectral probe (lambda-probe) on stored mean acts.
# lambda=1 pure manifold geometry (Laplacian eigenmaps), lambda=0 pure
# supervised HSIC dependence; grid 0..1; 200 balanced draws x GroupKFold-by-
# ruler. CPU only; reads existing acts (J1/J3/J4-era mean dirs). See
# p8_lambda_probe/MATH_NOTES.md. TF-IDF row runs locally (run_tfidf_local.py).
#   sbatch v_1/src/stress_tests/sbatch/J20_p8_lambda.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J20 P8 lambda-probe === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

D=v_1/src/stress_tests/p8_lambda_probe
for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random mlm; do
    if [ -f "$D/results/p8_lambda__${M}.json" ]; then
        echo "--- P8: $M already done, skipping ---"; continue
    fi
    echo "--- P8 lambda-probe: $M ---"
    python -u $D/run_acts.py --method "$M" --n-jobs 32 \
        || echo "WARN p8 $M failed"
    commit_push "J20 P8: lambda-probe ${M} (job ${SLURM_JOB_ID})" \
        "$D/results/p8_lambda__${M}.json"
done
echo "=== Done. $(date) ==="
