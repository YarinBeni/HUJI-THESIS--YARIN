#!/bin/bash
#SBATCH --job-name=J11_aggregate
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=v_1/src/stress_tests/logs/J11_aggregate_%j.out
# ===========================================================================
# J11 — aggregate balanced-MC result JSONs into the write-up tables
# (results/RESULTS_stress_tests.md). Run last, after J6_p1_mc + J7. Trivial CPU;
# also runnable locally: python v_1/src/stress_tests/aggregate_tables.py
#   sbatch v_1/src/stress_tests/sbatch/J11_aggregate.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J11 aggregate === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs

python -u v_1/src/stress_tests/aggregate_tables.py || echo "WARN aggregate failed"

git add v_1/src/stress_tests/results/RESULTS_stress_tests.md 2>/dev/null || true
git commit -m "J11: aggregate balanced-MC tables (job ${SLURM_JOB_ID})" || echo "nothing to commit"
push_main
echo "=== Done. $(date) ==="
