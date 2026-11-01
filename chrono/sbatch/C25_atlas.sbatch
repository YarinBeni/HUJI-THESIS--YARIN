#!/bin/bash
#SBATCH --job-name=C25_atlas
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=chrono/sbatch/logs/C25_atlas_%j.out
# ===========================================================================
# C25 / P6 — the alignment atlas (figures in the style of arXiv:2610.01821):
# layer x layer manifold alignment, akk<->eng language alignment per layer,
# and the depth-of-time curves with entity overlays. CPU only; tier0 store.
# ===========================================================================
set -uo pipefail
echo "=== C25 atlas === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
python -c "import umap, matplotlib" 2>/dev/null || pip install -q umap-learn matplotlib 2>&1 | tail -1
python -u chrono/scripts/alignment_atlas.py || { echo "FAILED atlas"; exit 1; }
commit_push_sandbox "C25: alignment atlas figures (job ${SLURM_JOB_ID})" chrono/reports/tier0/atlas
echo "=== Done. $(date) ==="
