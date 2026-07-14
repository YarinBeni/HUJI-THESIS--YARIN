#!/bin/bash
#SBATCH --job-name=J22e_t12_rescore_1b7eng
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/stress_tests/logs/J22e_t12_rescore_1b7eng_%j.out
# ===========================================================================
# J22e — rescore qwen3_1b7 x engtier0 x pv0-2 (the three T12 F1 cells that
# were clobbered during the stale-rebase incident; raw JSONLs are on disk).
# CPU, minutes.
#   sbatch v_1/src/stress_tests/sbatch/J22e_t12_rescore_1b7eng.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J22e T12 rescore 1b7 eng === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs

D=v_1/src/stress_tests/t12_forced_dating
for PV in pv0 pv1 pv2; do
    python -u $D/score_forced.py --model qwen3_1b7 --cleaning engtier0 --variant "$PV" \
        || { echo "FAIL rescore $PV"; exit 1; }
done
commit_push "J22e: T12 rescore qwen3_1b7 engtier0 pv0-2 (ruler macro-F1; job ${SLURM_JOB_ID})" \
    "$D/results/t12f__qwen3_1b7__engtier0__pv0.json" \
    "$D/results/t12f__qwen3_1b7__engtier0__pv1.json" \
    "$D/results/t12f__qwen3_1b7__engtier0__pv2.json"
echo "=== Done. $(date) ==="
