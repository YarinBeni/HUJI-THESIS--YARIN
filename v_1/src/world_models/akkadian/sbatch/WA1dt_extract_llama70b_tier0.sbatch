#!/bin/bash
#SBATCH --job-name=WA1dt_tier0_70b
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=18:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WA1dt_tier0_70b_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# WA1dt — Llama-2-70B (trained + random) Akkadian extraction of eng_tier0, gpu:4.
# One task per method; variant fixed = eng_tier0, 512 tokens.
#   0 llama2_70b/eng_tier0   1 llama2_70b_random/eng_tier0
#   sbatch v_1/src/world_models/akkadian/sbatch/WA1dt_extract_llama70b_tier0.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(llama2_70b llama2_70b_random); V="eng_tier0"
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WA1dt extract ${M}/${V} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
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
    --batch-size 8 --max-tokens 512 \
    || { echo "FAILED extract ${M}/${V}"; exit 1; }
commit_push "WA1dt: akk acts ${M}/eng_tier0 (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/activations/${M}/${V}/metadata.json
echo "=== Done ${M}/${V}. $(date) ==="
