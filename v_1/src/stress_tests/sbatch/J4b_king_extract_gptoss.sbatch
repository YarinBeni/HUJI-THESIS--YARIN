#!/bin/bash
#SBATCH --job-name=J4b_king_gptoss
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J4b_king_gptoss_%j.out
# ===========================================================================
# J4b — king-token extraction for gpt-oss-120B (needs gpu:4; device_map=auto
# shards across the 4 GPUs). Same script/output contract as J4.
#   sbatch v_1/src/stress_tests/sbatch/J4b_king_extract_gptoss.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J4b king-extract gpt_oss_120b === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/stress_tests/p1_gurnee_tegmark/extract_king_acts.py \
    --hfid openai/gpt-oss-120b --method gpt_oss_120b --arch causal --max-tokens 512 \
    || { echo "FAILED extract gpt_oss_120b"; exit 1; }

ACTS=v_1/src/linear_probing/results/orcc__embed/activations
git add "${ACTS}/gpt_oss_120b_tier0_kinglast/metadata.json" "${ACTS}/gpt_oss_120b_tier0_kinglast/king_coverage.json" \
        "${ACTS}/gpt_oss_120b_tier0_kingmean/metadata.json" "${ACTS}/gpt_oss_120b_tier0_kingmean/king_coverage.json" 2>/dev/null || true
git commit -m "J4b: king-token acts gpt_oss_120b (job ${SLURM_JOB_ID})" || echo "nothing to commit"
for _i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 8 + 5)); done
echo "=== Done gpt_oss_120b. $(date) ==="
