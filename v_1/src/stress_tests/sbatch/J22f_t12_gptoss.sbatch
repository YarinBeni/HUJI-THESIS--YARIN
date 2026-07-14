#!/bin/bash
#SBATCH --job-name=J22f_t12_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=v_1/src/stress_tests/logs/J22f_t12_gptoss_%A_%a.out
# ===========================================================================
# J22f — T12 forced dating for gpt-oss-120B (the roster's missing answer row
# on the T12 + tier0-baseline slides). Same forced templates + scorer as the
# qwen3 ladder. task = cleaning: 0 tier0, 1 maximal, 2 engtier0; pv0-pv3 each.
# Resumable (skips scored cells, JSONL-append). ~J18b resources (gpu:8).
#   sbatch v_1/src/stress_tests/sbatch/J22f_t12_gptoss.sbatch
# ===========================================================================
set -uo pipefail
CLEANINGS=(tier0 maximal engtier0)
T=${SLURM_ARRAY_TASK_ID:-0}
C=${CLEANINGS[$T]}
M=gpt_oss_120b
H=openai/gpt-oss-120b
echo "=== J22f T12 gpt-oss ${C} === job ${SLURM_ARRAY_JOB_ID}_${T} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs

D=v_1/src/stress_tests/t12_forced_dating
for PV in pv0 pv1 pv2 pv3; do
    if [ -f "$D/results/t12f__${M}__${C}__${PV}.json" ]; then
        echo "--- $M x $C x $PV already scored, skipping ---"; continue
    fi
    echo "--- T12 forced: $M x $C x $PV ---"
    python -u $D/generate_forced.py --model "$M" --model_path "$H" \
        --cleaning "$C" --variant "$PV" --batch-size 4 \
        || { echo "WARN generate $M $C $PV failed"; continue; }
    python -u $D/score_forced.py --model "$M" --cleaning "$C" --variant "$PV" \
        || { echo "WARN score $M $C $PV failed"; continue; }
    commit_push "J22f T12: forced dating ${M} x ${C} x ${PV} (job ${SLURM_ARRAY_JOB_ID}_${T})" \
        "$D/results/t12f__${M}__${C}__${PV}.json"
done
echo "=== Done ${C}. $(date) ==="
