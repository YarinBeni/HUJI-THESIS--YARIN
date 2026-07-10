#!/bin/bash
#SBATCH --job-name=J18a_gen_dating_qwen3
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=v_1/src/stress_tests/logs/J18a_gen_dating_qwen3_%A_%a.out
# ===========================================================================
# J18a — T11 generated-answer dating, Qwen3 set. The behavioral counterpart of
# T10: instead of probing activations under a dating prompt, ask the model for
# the year and score the parsed answer with the SAME balanced-MC Spearman.
# Array task = model x cleaning (12 = 3 models x 4 cleanings):
#   task = m*4 + c;  m: 0 qwen3_1b7, 1 qwen3_8b, 2 qwen3_32b
#                    c: 0 tier0, 1 maximal, 2 maxking, 3 engtier0
# Raw generations -> t11_gen_dating/raw/ (gitignored); only the scored summary
# JSON is committed. Generation resumes from an existing JSONL if re-queued.
#   sbatch v_1/src/stress_tests/sbatch/J18a_gen_dating_qwen3.sbatch
# ===========================================================================
set -uo pipefail
MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B)
BATCH=(16 16 8)
CLEANINGS=(tier0 maximal maxking engtier0)
T=${SLURM_ARRAY_TASK_ID:-0}
M=${MODELS[$((T / 4))]}; H=${HFIDS[$((T / 4))]}; B=${BATCH[$((T / 4))]}
C=${CLEANINGS[$((T % 4))]}
echo "=== J18a T11 ${M} x ${C} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
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
    --batch-size "$B" --max-frag-words 300 --max-new-tokens 512 \
    || { echo "FAIL generate $M $C"; exit 1; }
python -u $D/score_gen_dating.py --model "$M" --cleaning "$C" \
    || { echo "FAIL score $M $C"; exit 1; }

commit_push "J18a: T11 generated-answer dating ${M} x ${C} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    "$D/results/t11_gen__${M}__${C}.json"
echo "=== Done ${M} x ${C}. $(date) ==="
