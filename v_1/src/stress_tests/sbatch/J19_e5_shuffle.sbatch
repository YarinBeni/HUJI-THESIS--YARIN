#!/bin/bash
#SBATCH --job-name=J19_e5_shuffle
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=v_1/src/stress_tests/logs/J19_e5_shuffle_%A_%a.out
# ===========================================================================
# J19 — E5 word-shuffle control. Does word ORDER matter to the embedding year
# signal? Each fragment is word-capped FIRST (both variants keep the same
# words) then shuffled (seed 42); shuf + unshuf twins are extracted with
# identical settings and probed with the standard balanced MC. delta ~ 0 =>
# the probe reads bag-of-tokens (TF-IDF-like), not composition.
# Array task = model: 0 qwen3_8b (causal), 1 thalesian_cunei400m (encoder).
# Word caps: qwen 300 words / 2048 tokens; thalesian 120 words / 512 tokens
# (its umT5 window; cap chosen so 120 transliteration words fit ~unclipped).
# Cleanings: tier0, maximal, maxking, engtier0. Acts npz are cluster-local
# (gitignored); only e5_shuffle/results/e5_mc__<model>.json is committed.
#   sbatch v_1/src/stress_tests/sbatch/J19_e5_shuffle.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_8b thalesian_cunei400m)
HFIDS=(Qwen/Qwen3-8B Thalesian/cuneiformBase-400m)
ARCHS=(causal encoder)
WORDS=(300 120)
TOKENS=(2048 512)
T=${SLURM_ARRAY_TASK_ID:-0}
M=${METHODS[$T]}; H=${HFIDS[$T]}; A=${ARCHS[$T]}; W=${WORDS[$T]}; K=${TOKENS[$T]}
echo "=== J19 E5 ${M} (${H}, ${A}) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
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
python -u $D/extract_shuffled_acts.py --hfid "$H" --method "$M" --arch "$A" \
    --max-frag-words "$W" --max-tokens "$K" \
    || { echo "FAIL extract $M"; exit 1; }
python -u $D/probe_e5_mc.py --method "$M" --n-jobs 8 \
    || { echo "FAIL probe $M"; exit 1; }

commit_push "J19: E5 word-shuffle control ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    "$D/results/e5_mc__${M}.json"
echo "=== Done ${M}. $(date) ==="
