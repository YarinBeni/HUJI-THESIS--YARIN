#!/bin/bash
# C9 (Round-3) — Phase D centroid-spline for the new flagship qwen3_1b7.
# Config: tier0 / mean / L1 (matches the C8 LORO flagship config; best layer
# from geodesic_best_layers.json qwen3_1b7__tier0__mean=1). Reuses the
# parametrized phase_d_job.sh, which commits its own phase_d/ outputs.
#
# Run from repo root on the cluster:
#   bash v_1/src/geodesic/phase_d/sbatch/submit_phase_d_qwen3_1b7.sh
# Independent of C1 — submit immediately.

SCRIPT=v_1/src/geodesic/phase_d/sbatch/phase_d_job.sh

JID=$(sbatch --parsable \
    --export=METHOD=qwen3_1b7,CLEANING=tier0,POOL=mean,LAYER=1 \
    "$SCRIPT")
echo "Submitted Phase D qwen3_1b7 tier0/mean/L1 → job $JID"
echo "Monitor: squeue -u \$USER"
