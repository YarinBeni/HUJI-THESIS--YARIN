#!/bin/bash
#SBATCH --job-name=J3a_t10_qwen3
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J3a_t10_qwen3_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# J3a — redo T10 prompt-reprobe on the qwen3 set, pooling mean + king_last +
# king_mean under pv0-pv3. Array: one model per task. gpt-oss = J3b (gpu:4).
#   0 qwen3_1b7   1 qwen3_8b   2 qwen3_32b
# Per task: extract (pv0..pv3) -> reprobe (year PLS+ridge per pool/layer).
# Activations are *.npz (gitignored, cluster-local); only the summary JSON is committed.
#   sbatch v_1/src/stress_tests/sbatch/J3a_t10_qwen3.sbatch
# ===========================================================================
set -uo pipefail
MODELS=(qwen3_1b7 qwen3_8b qwen3_32b)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B)
M="${MODELS[$SLURM_ARRAY_TASK_ID]}"; H="${HFIDS[$SLURM_ARRAY_TASK_ID]}"
echo "=== J3a T10 ${M} (${H}) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EX=v_1/src/stress_tests/redo_t10_prompt/extract_prompted_king_acts.py
RP=v_1/src/stress_tests/redo_t10_prompt/reprobe_king_pv.py
OUT=v_1/src/stress_tests/redo_t10_prompt/acts_${M}
for V in pv0 pv1 pv2 pv3; do
    python -u $EX --variant "$V" --model_path "$H" --out_dir "$OUT" --layers all \
        || { echo "FAIL extract $V $M"; exit 1; }
done
python -u $RP --acts-dir "$OUT/prompted_king" --model "$M" \
    --out v_1/src/stress_tests/redo_t10_prompt/results || { echo "FAIL reprobe $M"; exit 1; }

git add v_1/src/stress_tests/redo_t10_prompt/results/${M}__t10_king_summary.json 2>/dev/null || true
git commit -m "J3a: T10 prompt-reprobe (mean+king) ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" || echo "nothing to commit"
push_main
echo "=== Done ${M}. $(date) ==="
