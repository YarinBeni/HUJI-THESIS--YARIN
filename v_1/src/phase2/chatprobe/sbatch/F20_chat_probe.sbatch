#!/bin/bash
#SBATCH --job-name=F20_chat
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=v_1/src/phase2/chatprobe/logs/F20_chat_%j.out
# ===========================================================================
# F20 — E-chat: does a chat model represent documents differently when spoken
# to the way it was trained? Extracts qwen3_8b activations with the fragments
# wrapped in the chat template + a dating question (new arm: qwen3_8b_chat),
# then runs the standard E1 pairwise probe on both sites.
#
# Decision rule, pre-registered: qwen3_8b_chat beating qwen3_8b by > 1 sd on a
# variant means the bare-text choice UNDERSTATED the chat models in the deck's
# probing tables, and the affected arms should be flagged/rerun; within 1 sd
# means the deck stands (base-model arms were never affected either way).
# ===========================================================================
set -uo pipefail
echo "=== F20 chat probe === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/chatprobe/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

FAILED=0
for V in eng_tier0 akk_maximal; do
    python -u v_1/src/phase2/chatprobe/extract_chat_acts.py \
        --method qwen3_8b --variant "$V" \
        || { echo "FAILED extract ${V}"; FAILED=1; continue; }
    for S in mean last; do
        python -u v_1/src/phase2/pairs/probe_pairs.py \
            --method qwen3_8b_chat --variant "$V" --site "$S" \
            || { echo "FAILED probe ${V}/${S}"; FAILED=1; }
    done
done
[ "$FAILED" -eq 0 ] || exit 1

commit_push "F20: chat-mode probe qwen3_8b_chat (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/pairs/results/probes \
    v_1/src/phase2/pairs/results/directions \
    v_1/src/world_models/akkadian/activations/qwen3_8b_chat/*/metadata.json
echo "=== Done. $(date) ==="
