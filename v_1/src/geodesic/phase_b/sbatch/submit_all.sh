#!/bin/bash
# Submit all Phase B geodesic scan jobs.
# Run from repo root: bash v_1/src/geodesic/phase_b/sbatch/submit_all.sh
# All jobs run in parallel (CPU-only, no GPU needed).

SCRIPT=v_1/src/geodesic/phase_b/sbatch/scan_job.sh
JOB_IDS=()

submit() {
    local METHOD=$1 CLEANING=$2 POOL=$3
    JID=$(sbatch --parsable --export=METHOD=$METHOD,CLEANING=$CLEANING,POOL=$POOL $SCRIPT)
    echo "Submitted $METHOD/$CLEANING/$POOL → job $JID"
    JOB_IDS+=($JID)
}

# ── Thalesian models (4 combos each) ────────────────────────────────────────
for METHOD in thalesian_cunei400m thalesian_akk300m; do
    for CLEANING in tier0 maximal; do
        for POOL in mean last; do
            submit $METHOD $CLEANING $POOL
        done
    done
done

# ── Qwen2.5-7B (4 combos) ───────────────────────────────────────────────────
for CLEANING in tier0 maximal; do
    for POOL in mean last; do
        submit qwen $CLEANING $POOL
    done
done

# ── Single-config methods ────────────────────────────────────────────────────
submit random_qwen tier0 mean
submit mlm_aeneas  tier0 mean

# ── Qwen3 scale models (4 combos each: tier0+maximal × mean+last) ────────────
for SIZE in 1b7 8b 32b; do
    METHOD=qwen3_${SIZE}
    for CLEANING in tier0 maximal; do
        for POOL in mean last; do
            submit $METHOD $CLEANING $POOL
        done
    done
done

echo ""
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Aggregate when all done:"
echo "  git pull --rebase origin main"
echo "  python v_1/src/geodesic/phase_b/aggregate.py"
