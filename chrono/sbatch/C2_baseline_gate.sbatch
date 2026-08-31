#!/bin/bash
#SBATCH --job-name=C2_gate
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=chrono/sbatch/logs/C2_gate_%j.out
# ===========================================================================
# C2 — P0.4 baseline reproduction gate (CPU: probes over cached features;
# needs C1's EmbStore). Ridge + PLS over the full thalesian layer grid,
# both sites, both langs, read out through the A5 protocol. The verdict
# stays UNPINNED until the reference rho is re-derived from
# v_1/src/phase2/pairs/RESULTS.md (plan addendum: the legacy 0.41 used the
# 1,202/41 frame) — then re-run with GATE_RHO=<value> in the environment.
# chrono/artifacts is gitignored: the small results.parquet + gate reports
# are copied into tracked chrono/reports/ before commit_push_sandbox.
# ===========================================================================
set -uo pipefail
echo "=== C2 baseline gate === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox
mkdir -p chrono/sbatch/logs

ART=chrono/artifacts
GATE_ARGS=()
if [ -n "${GATE_RHO:-}" ]; then
    GATE_ARGS+=(--gate-rho "${GATE_RHO}" --gate-tol "${GATE_TOL:-0.02}")
fi

for LANG_ in akk eng; do
    python -u chrono/scripts/run_baseline_gate.py \
        --model Thalesian/AKK_300m --layers 0-8 --sites mean last \
        --lang "${LANG_}" --probes ridge pls \
        --report-out "${ART}/baseline_gate_report_${LANG_}.txt" \
        ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} \
        || { echo "FAILED baseline gate (${LANG_})"; exit 1; }
done

mkdir -p chrono/reports
cp "${ART}/results.parquet" chrono/reports/results.parquet
cp "${ART}"/baseline_gate_report_*.txt chrono/reports/
commit_push_sandbox "C2: P0.4 baseline gate reports (job ${SLURM_JOB_ID})" \
    chrono/reports
echo "=== Done. $(date) ==="
