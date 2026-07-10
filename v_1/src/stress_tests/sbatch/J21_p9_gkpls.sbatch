#!/bin/bash
#SBATCH --job-name=J21_p9_gkpls
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J21_p9_gkpls_%j.out
# ===========================================================================
# J21 — P9 Geodesic Kernel PLS (working note §2) on stored mean acts, all 4
# cleanings (tier0/maximal/maxking/engtier0). Kernel PLS on K_G = -1/2 H D_G H
# (Isomap = kPCA on K_G), Nystrom out-of-sample through the FIXED train graph;
# baselines: RBF-KPLS (geodesic vs kernel) + KRR on K_G (PLS vs kernel).
# 200 balanced draws x GroupKFold-by-ruler. CPU only; resumable (skips done
# JSONs). See p9_gkpls/gkpls.py docstring + p8_lambda_probe/MATH_NOTES.md.
#   sbatch v_1/src/stress_tests/sbatch/J21_p9_gkpls.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J21 P9 G-KPLS === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

D=v_1/src/stress_tests/p9_gkpls
for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random mlm; do
    if [ -f "$D/results/p9_gkpls__${M}.json" ]; then
        echo "--- P9: $M already done, skipping ---"; continue
    fi
    echo "--- P9 G-KPLS: $M ---"
    python -u $D/run_acts.py --method "$M" --n-jobs 32 \
        || echo "WARN p9 $M failed"
    commit_push "J21 P9: G-KPLS ${M} (job ${SLURM_JOB_ID})" \
        "$D/results/p9_gkpls__${M}.json"
done
echo "=== Done. $(date) ==="
