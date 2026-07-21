#!/bin/bash
#SBATCH --job-name=W0_build_random_llama70b
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/world_models/logs/W0_build_random_llama70b_%j.out
# ===========================================================================
# W0 — materialize the random-init Llama-2-70B (seed 42) ONCE on a CPU node
# (from_config needs a ~137GB RAM spike; extraction then loads the saved shards
# with device_map=auto). Writes to $WM_MODELS_DIR/llama2_70b_random
# (cluster-local, ~130GB). Prereq for W1d task 1. Idempotent (skips if present).
#   sbatch v_1/src/world_models/sbatch/W0_build_random_llama70b.sbatch
# ===========================================================================
set -uo pipefail
echo "=== W0 build random llama70b === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
# Llama-2's slow tokenizer needs sentencepiece (fast conversion broken in transformers>=5)
python -c "import sentencepiece" 2>/dev/null || pip install -q sentencepiece
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export WM_MODELS_DIR="${WM_MODELS_DIR:-$HOME/projects/wm_models}"

python -u v_1/src/world_models/build_random_llama.py --size 70b \
    || { echo "FAILED W0"; exit 1; }
echo "=== Done. $(date) ==="
