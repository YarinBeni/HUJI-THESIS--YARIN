#!/bin/bash
#SBATCH --job-name=J19c_e5_shuffle_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=16:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J19c_e5_shuffle_gptoss_%j.out
# ===========================================================================
# J19c — E5 word-shuffle control, gpt-oss-120B (gpu:8: no sdpa -> eager
# attention needs the full node, same as J18b/T10). Same design as J19/J19b.
#   sbatch v_1/src/stress_tests/sbatch/J19c_e5_shuffle_gptoss.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b; H=openai/gpt-oss-120b
echo "=== J19c E5 ${M} === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

D=v_1/src/stress_tests/e5_shuffle
if [ -f "$D/results/e5_mc__${M}.json" ]; then
    echo "--- E5 $M already done, skipping ---"; exit 0
fi
python -u $D/extract_shuffled_acts.py --hfid "$H" --method "$M" --arch causal \
    --max-frag-words 300 --max-tokens 2048 \
    || { echo "FAIL extract $M"; exit 1; }
python -u $D/probe_e5_mc.py --method "$M" --n-jobs 8 \
    || { echo "FAIL probe $M"; exit 1; }

commit_push "J19c: E5 word-shuffle control ${M} (job ${SLURM_JOB_ID})" \
    "$D/results/e5_mc__${M}.json"
echo "=== Done ${M}. $(date) ==="
