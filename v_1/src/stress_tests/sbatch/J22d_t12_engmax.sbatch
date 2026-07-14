#!/bin/bash
#SBATCH --job-name=J22d_t12_engmax
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=v_1/src/stress_tests/logs/J22d_t12_engmax_%A_%a.out
# ===========================================================================
# J22d — T12 forced dating on ENGMAXIMAL (English translations of the name-
# stripped maximal texts): the unconfounded English behavioral test. Deck
# moved the T12 English columns from engtier0 (translator re-inserts real
# king names) to engmaximal. Same forced templates; scoring includes ruler
# macro-F1. task = model: 0 qwen3_1b7, 1 qwen3_8b, 2 qwen3_32b. Resumable.
#   sbatch v_1/src/stress_tests/sbatch/J22d_t12_engmax.sbatch
# ===========================================================================
set -uo pipefail
MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B)
BATCH=(16 8 4)
T=${SLURM_ARRAY_TASK_ID:-0}
M=${MODELS[$T]}; H=${HFIDS[$T]}; B=${BATCH[$T]}
echo "=== J22d T12 engmaximal ${M} === job ${SLURM_ARRAY_JOB_ID}_${T} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs

D=v_1/src/stress_tests/t12_forced_dating
for PV in pv0 pv1 pv2 pv3; do
    if [ -f "$D/results/t12f__${M}__engmaximal__${PV}.json" ]; then
        echo "--- $M x engmaximal x $PV already scored, skipping ---"; continue
    fi
    echo "--- T12 forced: $M x engmaximal x $PV ---"
    python -u $D/generate_forced.py --model "$M" --model_path "$H" \
        --cleaning engmaximal --variant "$PV" --batch-size "$B" \
        || { echo "WARN generate $M engmaximal $PV failed"; continue; }
    python -u $D/score_forced.py --model "$M" --cleaning engmaximal --variant "$PV" \
        || { echo "WARN score $M engmaximal $PV failed"; continue; }
    commit_push "J22d T12: forced dating ${M} x engmaximal x ${PV} (job ${SLURM_ARRAY_JOB_ID}_${T})" \
        "$D/results/t12f__${M}__engmaximal__${PV}.json"
done
echo "=== Done ${M}. $(date) ==="
