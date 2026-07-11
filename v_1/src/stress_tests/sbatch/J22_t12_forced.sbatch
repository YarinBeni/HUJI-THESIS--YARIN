#!/bin/bash
#SBATCH --job-name=J22_t12_forced
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=v_1/src/stress_tests/logs/J22_t12_forced_%A_%a.out
# ===========================================================================
# J22 — T12 FORCED-answer dating, qwen3 ladder x {tier0, engtier0} x pv0-pv3.
# No refusal escape: every prompt variant demands a committed best guess
# (ruler + year_bce). Templates = the T10/phase-1b pv0-pv3 specs (identical
# few-shot pool, seed 42). task = m*2 + c; m: 0 qwen3_1b7, 1 qwen3_8b,
# 2 qwen3_32b; c: 0 tier0, 1 engtier0. Variants loop inside the task;
# generation resumes per-JSONL, scoring per variant commits results.
#   sbatch v_1/src/stress_tests/sbatch/J22_t12_forced.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J22 T12 forced === job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs

MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B)
BATCH=(16 8 4)
CLEANINGS=(tier0 engtier0)
T=${SLURM_ARRAY_TASK_ID}
M=${MODELS[$((T / 2))]}; H=${HFIDS[$((T / 2))]}; B=${BATCH[$((T / 2))]}
C=${CLEANINGS[$((T % 2))]}

D=v_1/src/stress_tests/t12_forced_dating
for PV in pv0 pv1 pv2 pv3; do
    if [ -f "$D/results/t12f__${M}__${C}__${PV}.json" ]; then
        echo "--- $M x $C x $PV already scored, skipping ---"; continue
    fi
    echo "--- T12 forced: $M x $C x $PV ---"
    python -u $D/generate_forced.py --model "$M" --model_path "$H" \
        --cleaning "$C" --variant "$PV" --batch-size "$B" \
        || { echo "WARN generate $M $C $PV failed"; continue; }
    python -u $D/score_forced.py --model "$M" --cleaning "$C" --variant "$PV" \
        || { echo "WARN score $M $C $PV failed"; continue; }
    commit_push "J22 T12: forced dating ${M} x ${C} x ${PV} (job ${SLURM_JOB_ID})" \
        "$D/results/t12f__${M}__${C}__${PV}.json"
done
echo "=== Done. $(date) ==="
