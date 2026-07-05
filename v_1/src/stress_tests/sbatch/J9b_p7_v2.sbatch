#!/bin/bash
#SBATCH --job-name=J9b_p7_v2
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=v_1/src/stress_tests/logs/J9b_p7_v2_%A_%a.out
# ===========================================================================
# J9b — P7 k-sparse v2 (CPU). One model per array task; per model runs BOTH probe
# families (classification + year-regression) on all three cleanings
# (tier0 / maximal / maxking) from on-disk mean acts. Task 0 additionally builds
# the 3-panel per-k curve figures at the end (after its own runs; the figures are
# rebuilt by J11-style reruns anyway once all tasks land).
#   sbatch v_1/src/stress_tests/sbatch/J9b_p7_v2.sbatch
# ===========================================================================
set -uo pipefail
MODELS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random)
M=${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "=== J9b P7 v2 [$M] === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

for C in tier0 maximal maxking; do
    python -u v_1/src/stress_tests/p7_ksparse/probe_p7_v2.py --method "$M" --cleaning "$C" \
        || echo "WARN p7_v2 $M $C failed"
done
commit_push "J9b: P7 v2 (cls+reg x 3 cleanings) $M (job ${SLURM_JOB_ID})" \
    v_1/src/stress_tests/p7_ksparse/results/v2/p7_v2__${M}__*.json
echo "=== Done [$M]. $(date) ==="
