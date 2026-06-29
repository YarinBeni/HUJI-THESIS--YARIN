#!/bin/bash
#SBATCH --job-name=J3b_t10_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J3b_t10_gptoss_%j.out
# ===========================================================================
# J3b — T10 prompt-reprobe (mean+king_last+king_mean, pv0-pv3) on gpt-oss-120B.
# Needs gpu:4. Same extract->reprobe chain as J3a.
#   sbatch v_1/src/stress_tests/sbatch/J3b_t10_gptoss.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b; H=openai/gpt-oss-120b
echo "=== J3b T10 ${M} (${H}) === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin claude/stress-test-timeline-analysis-9sh2vs || echo "WARN pull failed"
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
git commit -m "J3b: T10 prompt-reprobe (mean+king) gpt_oss_120b (job ${SLURM_JOB_ID})" || echo "nothing to commit"
git pull --rebase origin claude/stress-test-timeline-analysis-9sh2vs || true
git push origin claude/stress-test-timeline-analysis-9sh2vs || echo "WARN push failed"
echo "=== Done ${M}. $(date) ==="
