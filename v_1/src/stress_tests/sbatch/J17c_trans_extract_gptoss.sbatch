#!/bin/bash
#SBATCH --job-name=J17c_trans_extract_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J17c_trans_extract_gptoss_%j.out
# ===========================================================================
# J17c — embed the English translations with gpt-oss-120B (gpu:8; eager
# attention is safe here because extract_translation_acts truncates inputs at
# 512 tokens).
#   sbatch v_1/src/stress_tests/sbatch/J17c_trans_extract_gptoss.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J17c trans-extract gpt_oss_120b === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/stress_tests/translation/extract_translation_acts.py \
    --hfid openai/gpt-oss-120b --method gpt_oss_120b --arch causal \
    || { echo "FAIL extract gpt_oss_120b"; exit 1; }
echo "=== Done. $(date) ==="
