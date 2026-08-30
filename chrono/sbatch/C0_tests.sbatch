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
sync_sandbox
mkdir -p chrono/sbatch/logs

python -m pytest chrono/tests -q \
    || { echo "FAILED: chrono tests — do NOT submit C1/C2/C3"; exit 1; }

echo "=== C0 green: GPU jobs may be submitted. $(date) ==="
