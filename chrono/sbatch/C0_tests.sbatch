#!/bin/bash
#SBATCH --job-name=C0_tests
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=chrono/sbatch/logs/C0_tests_%j.out
# ===========================================================================
# C0 — the full chrono test suite on the cluster: THE gate before any GPU
# job. Nothing under C1..C3 may be submitted until this job exits 0 (SLA
# section 8 / plan P0.4 discipline). CPU only, no artifacts produced, so
# nothing to commit — the log is the deliverable.
# ===========================================================================
set -uo pipefail
echo "=== C0 tests === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run "\
    "stale code and push results from it"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs

# pytest is a dev-only dependency and the cluster `thesis` env does not
# ship it (job 32287 died here). Install it into the active env once,
# quietly; everything else chrono needs (torch, pandas, pyarrow, sklearn,
# scipy) is already there because phase 2 ran on this env.
python -c "import pytest" 2>/dev/null || {
    echo "[C0] installing pytest into \$CONDA_DEFAULT_ENV"
    python -m pip install --quiet pytest \
        || { echo "FAILED: cannot install pytest"; exit 1; }
}
python -c "import pandas, numpy, scipy, sklearn, torch, pyarrow" \
    || { echo "FAILED: the thesis env is missing a core dependency"; \
         exit 1; }

python -m pytest chrono/tests -q \
    || { echo "FAILED: chrono tests — do NOT submit C1/C2/C3"; exit 1; }

echo "=== C0 green: GPU jobs may be submitted. $(date) ==="
