#!/bin/bash
#SBATCH --job-name=J18b_gen_dating_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=v_1/src/stress_tests/logs/J18b_gen_dating_gptoss_%A_%a.out
# ===========================================================================
# J18b — T11 generated-answer dating, gpt-oss-120B. gpu:8 (weights need it,
# proven by J12b/J3b) + --max-frag-words 300 (same input cap as every T11 run;
# also required here: no sdpa -> eager attention OOMs on long fragments).
# Array task = cleaning: 0 tier0, 1 maximal, 2 maxking, 3 engtier0.
# Raw generations gitignored; scored summary JSON committed. Resumable.
#   sbatch v_1/src/stress_tests/sbatch/J18b_gen_dating_gptoss.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b; H=openai/gpt-oss-120b
CLEANINGS=(tier0 maximal maxking engtier0)
C=${CLEANINGS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "=== J18b T11 ${M} x ${C} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

D=v_1/src/stress_tests/t11_gen_dating
python -u $D/score_gen_dating.py --selftest || { echo "FAIL selftest"; exit 1; }
python -u $D/generate_dates.py --model "$M" --model_path "$H" --cleaning "$C" \
    --batch-size 4 --max-frag-words 300 --max-new-tokens 768 \
    || { echo "FAIL generate $M $C"; exit 1; }
python -u $D/score_gen_dating.py --model "$M" --cleaning "$C" \
    || { echo "FAIL score $M $C"; exit 1; }

commit_push "J18b: T11 generated-answer dating ${M} x ${C} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    "$D/results/t11_gen__${M}__${C}.json"
echo "=== Done ${M} x ${C}. $(date) ==="
