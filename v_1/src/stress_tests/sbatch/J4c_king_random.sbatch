#!/bin/bash
#SBATCH --job-name=J4c_king_random
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J4c_king_random_%j.out
# ===========================================================================
# J4c — king-token extraction for the RANDOM-weight model (Qwen3-8B arch,
# from_config, seed=42). THE critical control: if random king_last ~ 0 while
# pretrained ~ 0.66, the "date-at-the-king-name" result is airtight.
# After this: re-run J6_p1_mc to fill random's king_last/king_mean.
#   sbatch v_1/src/stress_tests/sbatch/J4c_king_random.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J4c king-extract random === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
git fetch origin main && git merge --ff-only FETCH_HEAD || echo "WARN sync failed"
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/stress_tests/p1_gurnee_tegmark/extract_king_acts.py \
    --hfid Qwen/Qwen3-8B --method random --arch causal --random --max-tokens 512 \
    || { echo "FAILED extract random king"; exit 1; }

ACTS=v_1/src/linear_probing/results/orcc__embed/activations
git add "${ACTS}/random_tier0_kinglast/metadata.json" "${ACTS}/random_tier0_kinglast/king_coverage.json" \
        "${ACTS}/random_tier0_kingmean/metadata.json" "${ACTS}/random_tier0_kingmean/king_coverage.json" 2>/dev/null || true
git commit -m "J4c: king-token acts random (control) (job ${SLURM_JOB_ID})" || echo "nothing to commit"
for _i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 8 + 5)); done
echo "=== Done random king. $(date) ==="
