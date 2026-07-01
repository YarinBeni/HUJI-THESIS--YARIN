#!/bin/bash
#SBATCH --job-name=J7_p2_geography
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J7_p2_geography_%j.out
# ===========================================================================
# J7 — P2 Godey geography (CPU). Runs NOW on existing mean activations + the J1
# gazetteer; no GPU, no re-extraction. The positive control / explicit-carrier
# contrast. Loops all methods with on-disk mean acts.
#   sbatch v_1/src/stress_tests/sbatch/J7_p2_geography.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J7 P2 geography === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random; do
    python -u v_1/src/stress_tests/p2_godey_geography/probe_p2.py --method "$M" || echo "WARN p2 $M failed"
done
git add v_1/src/stress_tests/p2_godey_geography/results/*.json 2>/dev/null || true
git commit -m "J7: P2 geography probe (job ${SLURM_JOB_ID})" || echo "nothing to commit"
push_main
echo "=== Done. $(date) ==="
