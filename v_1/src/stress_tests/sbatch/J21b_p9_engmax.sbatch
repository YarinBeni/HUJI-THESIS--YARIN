#!/bin/bash
#SBATCH --job-name=J21b_p9_engmax
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J21b_p9_engmax_%j.out
# ===========================================================================
# J21b — P9 G-KPLS on the engmaximal cleaning (translations of the name-
# stripped maximal texts; acts = {method}_engmaximal_mean from J17b/c). Deck
# moved the English column from engtier0 (name-confounded) to engmaximal.
# run_acts now MERGES the new cleaning into p9_gkpls__{method}.json.
#   sbatch v_1/src/stress_tests/sbatch/J21b_p9_engmax.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J21b P9 engmaximal === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

D=v_1/src/stress_tests/p9_gkpls
for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random; do
    F="$D/results/p9_gkpls__${M}.json"
    if [ -f "$F" ] && python -c "import json,sys; c=json.load(open('$F'))['cleanings'].get('engmaximal'); sys.exit(0 if c and not c.get('missing') else 1)"; then
        echo "--- P9 engmaximal: $M already done, skipping ---"; continue
    fi
    echo "--- P9 G-KPLS engmaximal: $M ---"
    python -u $D/run_acts.py --method "$M" --cleanings engmaximal --n-jobs 32 \
        || echo "WARN p9 engmax $M failed"
    commit_push "J21b P9: G-KPLS ${M} engmaximal (job ${SLURM_JOB_ID})" \
        "$D/results/p9_gkpls__${M}.json"
done
echo "=== Done. $(date) ==="
