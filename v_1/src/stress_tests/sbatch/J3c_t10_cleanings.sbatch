#!/bin/bash
#SBATCH --job-name=J3c_t10_cleanings
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --output=v_1/src/stress_tests/logs/J3c_t10_cleanings_%A_%a.out
# ===========================================================================
# J3c — T10 prompted-activation extraction for the NEW cleanings (maximal +
# maxking) on the qwen3 ladder. Array task -> (model, cleaning):
#   0 1b7/maximal  1 8b/maximal  2 32b/maximal
#   3 1b7/maxking  4 8b/maxking  5 32b/maxking
# Extract pv0-pv3 into acts_<model>/prompted_king_<cleaning>/ (npz, cluster-only).
# The MC reprobe is J3r (CPU, array over model x cleaning).
#   sbatch v_1/src/stress_tests/sbatch/J3c_t10_cleanings.sbatch
# ===========================================================================
set -uo pipefail
MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B)
CLEANINGS=(maximal maxking)
i=${SLURM_ARRAY_TASK_ID}
M=${MODELS[$((i % 3))]}; H=${HFIDS[$((i % 3))]}; C=${CLEANINGS[$((i / 3))]}
echo "=== J3c T10 ${M} x ${C} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EX=v_1/src/stress_tests/redo_t10_prompt/extract_prompted_king_acts.py
OUT=v_1/src/stress_tests/redo_t10_prompt/acts_${M}
for V in pv0 pv1 pv2 pv3; do
    python -u $EX --variant "$V" --cleaning "$C" --model_path "$H" --out_dir "$OUT" --layers all \
        || { echo "FAIL extract $V $M $C"; exit 1; }
done
echo "=== Done ${M} x ${C}. $(date) ==="
