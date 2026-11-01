#!/bin/bash
#SBATCH --job-name=AGENT_runner
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=chrono/sbatch/logs/AGENT_runner_%j.out
# ===========================================================================
# AGENT — the git-driven command runner (chrono/agent/runner.sh) as a
# ONE-DAY, 2-CPU job. It polls the sandbox branch, runs pushed inbox
# scripts (including `sbatch ...` for the real GPU jobs), pushes their
# output. Deliberately expires after 24h so a forgotten runner cannot
# sit there for a week; it is meant to be started per working session
# and STOPPED at its end (push chrono/agent/STOP, or `scancel`). See
# chrono/agent/README.md. Nothing else in this file: the runner re-syncs
# the branch itself, so this script never goes stale.
# ===========================================================================
set -uo pipefail
echo "=== AGENT runner === job ${SLURM_JOB_ID} $(date)"
cd ~/projects/HUJI-THESIS--YARIN
mkdir -p chrono/sbatch/logs
exec bash chrono/agent/runner.sh
