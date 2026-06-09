#!/bin/bash
# Submit M1, then M2 chained on M1 success (afterok). Run from repo root:
#   bash v_1/src/geodesic/maximal_figs/sbatch/submit_all.sh
set -euo pipefail
SB=v_1/src/geodesic/maximal_figs/sbatch

JID=$(sbatch --parsable "$SB/M1_supervised_maximal.sbatch")
echo "M1 (supervised probes + fig1/2/4) = $JID"

M2=$(sbatch --parsable --dependency=afterok:"$JID" "$SB/M2_mae_maximal.sbatch")
echo "M2 (per-ruler MAE, waits for M1 ok) = $M2"
echo "watch: squeue -j $JID,$M2"
