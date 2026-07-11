#!/bin/bash
#SBATCH --job-name=J19b_e5_shuffle_rest
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=v_1/src/stress_tests/logs/J19b_e5_shuffle_rest_%A_%a.out
# ===========================================================================
# J19b — E5 word-shuffle control, the REST of the ladder (J19 covered qwen3_8b
# + thalesian_cunei400m). Same design: word-cap FIRST then shuffle (seed 42),
# shuf/unshuf twins extracted identically, standard balanced MC per cleaning.
# Array task = model: 0 qwen3_1b7, 1 qwen3_32b, 2 thalesian_akk300m,
# 3 umt5_base, 4 random (Qwen3-8B from_config seed 42, --random).
# gpt-oss = J19c (gpu:8). MLM (37M custom arch) has no E5 extractor — parked.
# Word caps: causal 300w/2048t; encoders 120w/512t (uMT5 window).
#   sbatch v_1/src/stress_tests/sbatch/J19b_e5_shuffle_rest.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_32b thalesian_akk300m umt5_base random)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-32B Thalesian/AKK_300m google/umt5-base Qwen/Qwen3-8B)
ARCHS=(causal causal encoder encoder causal)
WORDS=(300 300 120 120 300)
TOKENS=(2048 2048 512 512 2048)
EXTRA=("" "" "" "" "--random")
T=${SLURM_ARRAY_TASK_ID:-0}
M=${METHODS[$T]}; H=${HFIDS[$T]}; A=${ARCHS[$T]}; W=${WORDS[$T]}; K=${TOKENS[$T]}; X=${EXTRA[$T]}
echo "=== J19b E5 ${M} (${H}, ${A}) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
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
python -u $D/extract_shuffled_acts.py --hfid "$H" --method "$M" --arch "$A" \
    --max-frag-words "$W" --max-tokens "$K" $X \
    || { echo "FAIL extract $M"; exit 1; }
python -u $D/probe_e5_mc.py --method "$M" --n-jobs 8 \
    || { echo "FAIL probe $M"; exit 1; }

commit_push "J19b: E5 word-shuffle control ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    "$D/results/e5_mc__${M}.json"
echo "=== Done ${M}. $(date) ==="
