#!/bin/bash
#SBATCH --job-name=F2_behav
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/phase2/pairs/logs/F2_behav_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# F2 — E1 behavioural pairwise test: ask the model itself "was Text A composed
# earlier than Text B?", both presentation orders, single-token Yes/No read-out
# (protocol of El-Shangiti et al., NAACL 2025, adapted to whole inscriptions).
#
# Instruct-capable arms only — base Llama-2/OLMo have no Yes/No calibration and
# would only measure prompt-format noise (the probe in F1 covers their
# representations):
#   0 qwen3_1b7    1 qwen3_8b    2 qwen3_32b
# (gpt_oss_120b needs the 8-GPU node; submit it separately with --gres=gpu:8
#  and --array=3-3 after editing METHODS if wanted.)
#
# Each task runs eng_tier0 (the trusted gloss) and akk_maximal (raw Akkadian).
#
#   sbatch v_1/src/phase2/pairs/sbatch/F2_pairs_behavioral.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== F2 behavioral ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/pairs/logs

FAILED=0
for V in eng_tier0 akk_maximal; do
    python -u v_1/src/phase2/pairs/behavioral_pairs.py --method "$M" --variant "$V" \
        || { echo "FAILED ${M}/${V}"; FAILED=1; }
done
[ "$FAILED" -eq 0 ] || exit 1

N=$(ls v_1/src/phase2/pairs/results/behavioral/${M}.*.json 2>/dev/null | wc -l)
echo "[check] ${M}: ${N} result files"
[ "$N" -gt 0 ] || { echo "FAILED: ${M} produced no results"; exit 1; }

commit_push "F2: behavioral pairs ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/phase2/pairs/results/behavioral
echo "=== Done ${M}. $(date) ==="
