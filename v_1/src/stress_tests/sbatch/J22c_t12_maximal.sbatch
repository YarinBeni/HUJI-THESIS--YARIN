#!/bin/bash
#SBATCH --job-name=J22c_t12_maximal
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=v_1/src/stress_tests/logs/J22c_t12_maximal_%A_%a.out
# ===========================================================================
# J22c — T12 forced dating on the MAXIMAL cleaning (names stripped — the
# unconfounded Akkadian test), qwen3 ladder x pv0-pv3. Same forced templates
# as J22; scoring includes ruler macro-F1. task = model: 0 qwen3_1b7,
# 1 qwen3_8b, 2 qwen3_32b. Resumable (skips scored cells, JSONL-append).
#   sbatch v_1/src/stress_tests/sbatch/J22c_t12_maximal.sbatch
# ===========================================================================
set -uo pipefail
MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B)
BATCH=(16 8 4)
T=${SLURM_ARRAY_TASK_ID:-0}
M=${MODELS[$T]}; H=${HFIDS[$T]}; B=${BATCH[$T]}
echo "=== J22c T12 maximal ${M} === job ${SLURM_ARRAY_JOB_ID}_${T} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs

D=v_1/src/stress_tests/t12_forced_dating
for PV in pv0 pv1 pv2 pv3; do
    if [ -f "$D/results/t12f__${M}__maximal__${PV}.json" ]; then
        echo "--- $M x maximal x $PV already scored, skipping ---"; continue
    fi
    echo "--- T12 forced: $M x maximal x $PV ---"
    python -u $D/generate_forced.py --model "$M" --model_path "$H" \
        --cleaning maximal --variant "$PV" --batch-size "$B" \
        || { echo "WARN generate $M maximal $PV failed"; continue; }
    python -u $D/score_forced.py --model "$M" --cleaning maximal --variant "$PV" \
        || { echo "WARN score $M maximal $PV failed"; continue; }
    commit_push "J22c T12: forced dating ${M} x maximal x ${PV} (job ${SLURM_ARRAY_JOB_ID}_${T})" \
        "$D/results/t12f__${M}__maximal__${PV}.json"
done
echo "=== Done ${M}. $(date) ==="
