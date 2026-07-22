#!/bin/bash
#SBATCH --job-name=WA1_extract
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WA1_extract_%A_%a.out
#SBATCH --array=0-15
# ===========================================================================
# WA1 — Akkadian G&T-mimic extraction, gpu:1 decoder arms. One array task per
# (method, text-variant). 8 methods x 2 variants = 16 tasks.
#   methods: qwen3_1b7 qwen3_8b qwen3_32b random llama2_7b llama2_13b
#            llama2_7b_random llama2_13b_random
#   variants: akk_maximal eng_maximal
# gpt-oss = WA1b (gpu:8), llama-70b = WA1d (gpu:4). Encoders excluded by design.
#   sbatch v_1/src/world_models/akkadian/sbatch/WA1_extract.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(qwen3_1b7 qwen3_8b qwen3_32b random llama2_7b llama2_13b llama2_7b_random llama2_13b_random)
VARIANTS=(akk_maximal eng_maximal)
M="${METHODS[$((SLURM_ARRAY_TASK_ID/2))]}"; V="${VARIANTS[$((SLURM_ARRAY_TASK_ID%2))]}"

echo "=== WA1 extract ${M}/${V} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
python -c "import sentencepiece" 2>/dev/null || pip install -q sentencepiece
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/world_models/akkadian/extract_akk.py --method "$M" --variant "$V" \
    || { echo "FAILED extract ${M}/${V}"; exit 1; }

commit_push "WA1: akk acts ${M}/${V} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/activations/${M}/${V}/metadata.json
echo "=== Done ${M}/${V}. $(date) ==="
