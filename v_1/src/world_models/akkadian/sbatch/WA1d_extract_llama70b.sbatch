#!/bin/bash
#SBATCH --job-name=WA1d_extract_llama70b
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=18:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WA1d_extract_llama70b_%A_%a.out
#SBATCH --array=0-3
# ===========================================================================
# WA1d — Llama-2-70B (trained + random) Akkadian extraction, gpu:4. One task per
# (method, variant). llama2_70b_random reuses the checkpoint the world-models W0
# already built at $WM_MODELS_DIR/llama2_70b_random.
#   0 llama2_70b/akk  1 llama2_70b/eng  2 llama2_70b_random/akk  3 .../eng
#   sbatch v_1/src/world_models/akkadian/sbatch/WA1d_extract_llama70b.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(llama2_70b llama2_70b_random); VARIANTS=(akk_maximal eng_maximal)
M="${METHODS[$((SLURM_ARRAY_TASK_ID/2))]}"; V="${VARIANTS[$((SLURM_ARRAY_TASK_ID%2))]}"
echo "=== WA1d extract ${M}/${V} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
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

python -u v_1/src/world_models/akkadian/extract_akk.py --method "$M" --variant "$V" --batch-size 8 \
    || { echo "FAILED extract ${M}/${V}"; exit 1; }

commit_push "WA1d: akk acts ${M}/${V} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/activations/${M}/${V}/metadata.json
echo "=== Done ${M}/${V}. $(date) ==="
