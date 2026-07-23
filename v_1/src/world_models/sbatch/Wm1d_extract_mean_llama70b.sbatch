#!/bin/bash
#SBATCH --job-name=Wm1d_mean_70b
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/world_models/logs/Wm1d_mean_70b_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# Wm1d — add `mean` pooling to Llama-2-70B (trained + random). gpu:4 each.
#   0 llama2_70b   1 llama2_70b_random
#   sbatch v_1/src/world_models/sbatch/Wm1d_extract_mean_llama70b.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(llama2_70b llama2_70b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Wm1d extract ${M} (last,mean) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
python -c "import sentencepiece" 2>/dev/null || pip install -q sentencepiece
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WM_MODELS_DIR="${WM_MODELS_DIR:-$HOME/projects/wm_models}"
python -u v_1/src/world_models/extract_acts.py --method "$M" --sites last,mean --batch-size 32 \
    || { echo "FAILED extract ${M}"; exit 1; }
commit_push "Wm1d: world-models acts ${M} +mean (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/activations/${M}/*/metadata.json
echo "=== Done ${M}. $(date) ==="
