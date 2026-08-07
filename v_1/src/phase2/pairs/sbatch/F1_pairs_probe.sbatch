#!/bin/bash
#SBATCH --job-name=F1_pairs
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/phase2/pairs/logs/F1_pairs_%A_%a.out
#SBATCH --array=1-13
# ===========================================================================
# F1 — E1 representation-side pairwise probe ("which fragment is earlier?").
#
# One array task per arm; each runs the akk_maximal and eng_tier0 variants at
# site=mean. CPU only: it walks the akkadian npz already on this filesystem
# (the same store probe_akk reads), so it needs the extractions to exist but
# never touches a GPU.
#
#   0 tfidf_char (already run locally; re-enable with --array=0-13)
#                       1 olmo2_7b            2 olmo2_7b_random
#   3 llama2_7b         4 llama2_7b_random    5 llama2_13b
#   6 llama2_13b_random 7 llama2_70b          8 llama2_70b_random
#   9 qwen3_1b7        10 qwen3_8b           11 qwen3_32b
#  12 gpt_oss_120b     13 random
#
#   sbatch v_1/src/phase2/pairs/sbatch/F1_pairs_probe.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(tfidf_char olmo2_7b olmo2_7b_random llama2_7b llama2_7b_random
         llama2_13b llama2_13b_random llama2_70b llama2_70b_random
         qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F1 pairs ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/pairs/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

FAILED=0
for V in akk_maximal eng_tier0; do
    if [ "$M" != "tfidf_char" ] && \
       [ ! -d "v_1/src/world_models/akkadian/activations/${M}/${V}" ]; then
        echo "[skip] ${M}/${V}: no extraction on disk"
        continue
    fi
    python -u v_1/src/phase2/pairs/probe_pairs.py --method "$M" --variant "$V" \
        || { echo "FAILED ${M}/${V}"; FAILED=1; }
done
[ "$FAILED" -eq 0 ] || exit 1

N=$(ls v_1/src/phase2/pairs/results/probes/${M}.*.json 2>/dev/null | wc -l)
echo "[check] ${M}: ${N} result files"
[ "$N" -gt 0 ] || { echo "FAILED: ${M} produced no results"; exit 1; }

commit_push "F1: pairwise probe ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/pairs/results/probes \
    v_1/src/phase2/pairs/results/directions
echo "=== Done ${M}. $(date) ==="
