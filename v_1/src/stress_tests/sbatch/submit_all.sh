#!/bin/bash
# Submit the whole stress-test wave with correct dependencies, in one shot.
#   bash v_1/src/stress_tests/sbatch/submit_all.sh
# J6 (P1 probe) waits for J4 + J4b (king activations). J8 (P3 timeline) waits for
# J5 (anchors). Everything else runs immediately and in parallel.
set -uo pipefail
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARN pull failed"
S=v_1/src/stress_tests/sbatch

# --- independent jobs (run now) ---
sbatch --parsable $S/J2a_t9_qwen3.sbatch          | tee /dev/stderr >/dev/null
sbatch --parsable $S/J2b_t9_gptoss.sbatch         >/dev/null
sbatch --parsable $S/J3a_t10_qwen3.sbatch         >/dev/null
sbatch --parsable $S/J3b_t10_gptoss.sbatch        >/dev/null
sbatch --parsable $S/J7_p2_geography.sbatch       >/dev/null
sbatch --parsable $S/J9_p7_ksparse.sbatch         >/dev/null

# --- jobs that others depend on (capture their ids) ---
J4=$(sbatch --parsable $S/J4_king_extract.sbatch)
J4b=$(sbatch --parsable $S/J4b_king_extract_gptoss.sbatch)
J5=$(sbatch --parsable $S/J5_p3_anchors.sbatch)
echo "J4=$J4  J4b=$J4b  J5=$J5"

# --- dependent jobs (auto-start after their inputs finish OK) ---
J6=$(sbatch --parsable --dependency=afterok:${J4}:${J4b} $S/J6_p1_probe.sbatch)
J8=$(sbatch --parsable --dependency=afterok:${J5}        $S/J8_p3_timeline.sbatch)
echo "J6=$J6 (after J4,J4b)   J8=$J8 (after J5)"

echo "All submitted. Watch: squeue -u \$USER"
