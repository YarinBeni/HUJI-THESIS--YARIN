#!/bin/bash
#SBATCH --job-name=J3r_t10_reprobe_mc
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=v_1/src/stress_tests/logs/J3r_t10_reprobe_mc_%A_%a.out
# ===========================================================================
# J3r (MC) — T10 reprobe under Monte-Carlo BALANCED draws on the prompted
# activations from J3a (acts_<model>/prompted_king/, gitignored, on cluster NFS).
# CPU only, no GPU, no re-extraction.
#
# WHY AN ARRAY: the old single job looped all 4 models serially and TIMED OUT at
# 3h (the per-model PLS k-sweep + Ridge over every pv variant x layer is heavy;
# only qwen3_1b7 finished). Each array task now does ONE model in its own wall
# clock, in parallel. Task->model:  0 qwen3_1b7 | 1 qwen3_8b | 2 qwen3_32b |
# 3 gpt_oss_120b. A model with no prompted acts on disk exits cleanly (run J3a
# first to produce 8b/32b/gpt-oss prompted acts if missing).
#   sbatch v_1/src/stress_tests/sbatch/J3r_t10_reprobe_mc.sbatch
# ===========================================================================
set -uo pipefail
MODELS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b)
M=${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "=== J3r MC T10 reprobe [$M] === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

RP=v_1/src/stress_tests/redo_t10_prompt/reprobe_king_mc.py
AD=v_1/src/stress_tests/redo_t10_prompt/acts_${M}/prompted_king
if [ -d "$AD" ] && ls "$AD"/pv*/L*.npz >/dev/null 2>&1; then
    echo "--- T10 MC reprobe: $M ---"
    python -u $RP --acts-dir "$AD" --model "$M" || { echo "FAIL t10_mc $M"; exit 1; }
    commit_push "J3r MC: T10 balanced-MC reprobe $M (job ${SLURM_JOB_ID})" \
        "v_1/src/stress_tests/redo_t10_prompt/results/${M}__t10_mc_summary.json"
else
    echo "skip $M (no prompted acts on disk — run J3a to extract them first)"
fi
echo "=== Done [$M]. $(date) ==="
