#!/bin/bash
#SBATCH --job-name=J3b_t10_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=v_1/src/stress_tests/logs/J3b_t10_gptoss_%A_%a.out
# ===========================================================================
# J3b — T10 prompted-activation extraction for gpt-oss-120B. gpu:8: the gpu:4 run
# OOM'd during weight placement (120B ~240GB bf16; gpu:8 proven by J12b). Array
# task = one cleaning (0 tier0, 1 maximal, 2 maxking), each extracts pv0-pv3 into
# acts_gpt_oss_120b/prompted_king[_<cleaning>]/. MC reprobe = J3r (CPU).
#   sbatch v_1/src/stress_tests/sbatch/J3b_t10_gptoss.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b; H=openai/gpt-oss-120b
CLEANINGS=(tier0 maximal maxking)
C=${CLEANINGS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "=== J3b T10 ${M} x ${C} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EX=v_1/src/stress_tests/redo_t10_prompt/extract_prompted_king_acts.py
OUT=v_1/src/stress_tests/redo_t10_prompt/acts_${M}
for V in pv0 pv1 pv2 pv3; do
    python -u $EX --variant "$V" --cleaning "$C" --model_path "$H" --out_dir "$OUT" --layers all --max-frag-words 300 \
        || { echo "FAIL extract $V $M $C"; exit 1; }
done
echo "=== Done ${M} x ${C}. $(date) ==="
