#!/bin/bash
#SBATCH --job-name=WA1t_tier0
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WA1t_tier0_%A_%a.out
#SBATCH --array=0-7
# ===========================================================================
# WA1t — Akkadian extraction of the NEW eng_tier0 variant (faithful literal gloss)
# for the gpu:1 decoder arms. One array task per method; variant fixed = eng_tier0.
# eng_tier0 is long (~1.9k chars) so we extract at 512 tokens (vs 256 for the short
# variants). gpt-oss/llama-70b handled elsewhere; encoders excluded by design.
#   0 qwen3_1b7  1 qwen3_8b  2 qwen3_32b  3 random  4 llama2_7b  5 llama2_13b
#   6 llama2_7b_random  7 llama2_13b_random
#   sbatch v_1/src/world_models/akkadian/sbatch/WA1t_extract_tier0.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b random llama2_7b llama2_13b \
         llama2_7b_random llama2_13b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"; V="eng_tier0"
echo "=== WA1t extract ${M}/${V} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
python -c "import sentencepiece" 2>/dev/null || pip install -q sentencepiece
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WM_MODELS_DIR="${WM_MODELS_DIR:-$HOME/projects/wm_models}"
python -u v_1/src/world_models/akkadian/extract_akk.py --method "$M" --variant "$V" \
    --max-tokens 512 \
    || { echo "FAILED extract ${M}/${V}"; exit 1; }
commit_push "WA1t: akk acts ${M}/eng_tier0 (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/activations/${M}/${V}/metadata.json
echo "=== Done ${M}/${V}. $(date) ==="
