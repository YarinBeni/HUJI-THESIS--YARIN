#!/bin/bash
#SBATCH --job-name=J4d_king_mlm
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J4d_king_mlm_%j.out
# ===========================================================================
# J4d — MLM (AeneasForMLM) activation extraction onto the balanced-MC ladder.
# Produces mlm_tier0_mean / mlm_maximal_mean / mlm_tier0_kinglast /
# mlm_tier0_kingmean under orcc__embed/activations, named so probe_p1_mc finds
# them with method="mlm" (mean tier0+maximal, king_last/king_mean tier0-only,
# sign-level king location). After this, re-run J6_p1_mc to add the mlm row.
#   sbatch v_1/src/stress_tests/sbatch/J4d_king_mlm.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J4d king-extract MLM === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/stress_tests/p1_gurnee_tegmark/extract_mlm_king_acts.py --max-tokens 512 \
    || { echo "FAILED extract MLM king"; exit 1; }

ACTS=v_1/src/linear_probing/results/orcc__embed/activations
git add "${ACTS}/mlm_tier0_kinglast/metadata.json"   "${ACTS}/mlm_tier0_kinglast/king_coverage.json" \
        "${ACTS}/mlm_tier0_kingmean/metadata.json"   "${ACTS}/mlm_tier0_kingmean/king_coverage.json" \
        "${ACTS}/mlm_tier0_mean/metadata.json"       "${ACTS}/mlm_maximal_mean/metadata.json" 2>/dev/null || true
git commit -m "J4d: MLM king+mean acts on balanced-MC setup (job ${SLURM_JOB_ID})" || echo "nothing to commit"
for _i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 8 + 5)); done
echo "=== Done MLM king. $(date) ==="
