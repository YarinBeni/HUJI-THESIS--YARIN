#!/bin/bash
# One-shot submission of the current phase-2 wave with SLURM dependencies.
# Safe to re-run: each job overwrites its own results and commits them.
#
#   bash v_1/src/phase2/submit_all.sh
#
# Dependency graph:
#   F6 logit-lens      independent (CPU)
#   F7 FVU gate        independent (CPU, downloads the Scope SAE)
#   F8 feature hunt    afterok F7  (needs the gate verdict + layer offset)
#   F9 steering        independent (GPU; regenerates directions if missing)
set -euo pipefail
cd "$(dirname "$0")/../../.."      # repo root

F6=$(sbatch --parsable v_1/src/phase2/traces/sbatch/F6_logit_lens.sbatch)
F7=$(sbatch --parsable v_1/src/phase2/sae/sbatch/F7_fvu_gate.sbatch)
F8=$(sbatch --parsable --dependency=afterok:${F7} \
     v_1/src/phase2/sae/sbatch/F8_feature_hunt.sbatch)
F9=$(sbatch --parsable v_1/src/phase2/steering/sbatch/F9_steering.sbatch)

echo "submitted:"
echo "  F6 logit-lens    ${F6}   (3 CPU tasks)"
echo "  F7 fvu gate      ${F7}   (1 CPU task)"
echo "  F8 feature hunt  ${F8}   (afterok:${F7})"
echo "  F9 steering      ${F9}   (3 GPU tasks)"
squeue -u "$USER"
