#!/bin/bash
#SBATCH --job-name=J3r_t10_reprobe_mc
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=v_1/src/stress_tests/logs/J3r_t10_reprobe_mc_%A_%a.out
# ===========================================================================
# J3r (MC) — T10 reprobe under balanced MC draws, one (model x cleaning) per array
# task. CPU only; needs the prompted acts from J3a/J3b/J3c on cluster NFS.
#   task = cleaning*4 + model:
#     models    0 qwen3_1b7 | 1 qwen3_8b | 2 qwen3_32b | 3 gpt_oss_120b
#     cleanings 0 tier0     | 1 maximal  | 2 maxking
#   (tier0 qwen tasks 0-2 already have summaries; submit --array=3-11 to skip, or
#    the task skips itself if the summary exists.)
#   sbatch --array=3-11 v_1/src/stress_tests/sbatch/J3r_t10_reprobe_mc.sbatch
# ===========================================================================
set -uo pipefail
MODELS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b)
CLEANINGS=(tier0 maximal maxking)
i=${SLURM_ARRAY_TASK_ID:-0}
M=${MODELS[$((i % 4))]}; C=${CLEANINGS[$((i / 4))]}
echo "=== J3r MC T10 reprobe [$M x $C] === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

RP=v_1/src/stress_tests/redo_t10_prompt/reprobe_king_mc.py
RES=v_1/src/stress_tests/redo_t10_prompt/results
if [ "$C" = "tier0" ]; then
    AD=v_1/src/stress_tests/redo_t10_prompt/acts_${M}/prompted_king
    SUMMARY=${RES}/${M}__t10_mc_summary.json
    TAG=""
else
    AD=v_1/src/stress_tests/redo_t10_prompt/acts_${M}/prompted_king_${C}
    SUMMARY=${RES}/${M}__t10_mc_summary__${C}.json
    TAG="$C"
fi
if [ -f "$SUMMARY" ]; then
    echo "skip $M x $C (summary already exists)"
elif [ -d "$AD" ] && ls "$AD"/pv*/L*.npz >/dev/null 2>&1; then
    echo "--- T10 MC reprobe: $M x $C ---"
    python -u $RP --acts-dir "$AD" --model "$M" ${TAG:+--tag "$TAG"} \
        || { echo "FAIL t10_mc $M $C"; exit 1; }
    commit_push "J3r MC: T10 reprobe $M x $C (job ${SLURM_JOB_ID})" "$SUMMARY"
else
    echo "skip $M x $C (no prompted acts on disk — run J3a/J3b/J3c first)"
fi
echo "=== Done [$M x $C]. $(date) ==="
