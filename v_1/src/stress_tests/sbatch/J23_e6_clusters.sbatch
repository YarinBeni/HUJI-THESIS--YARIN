#!/bin/bash
#SBATCH --job-name=J23_e6_clusters
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J23_e6_clusters_%j.out
# ===========================================================================
# J23 — E6 cluster structure: KMeans k-sweep (is k=8 optimal?), cluster-vs-
# metadata alignment (ARI/AMI vs ruler8/period/sub_genre/provenance/year),
# label silhouettes, all under BOTH the full corpus and the 200 balanced
# draws. All table-1 winner models + controls, maximal + engtier0, best year
# layer. CPU; resumable per model x cleaning.
#   sbatch v_1/src/stress_tests/sbatch/J23_e6_clusters.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J23 E6 clusters === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

D=v_1/src/stress_tests/e6_clusters
for M in thalesian_cunei400m qwen3_32b qwen3_8b qwen3_1b7 gpt_oss_120b \
         thalesian_akk300m umt5_base mlm random tfidf; do
    echo "--- E6: $M ---"
    python -u $D/run_cluster_metrics.py --method "$M" \
        || echo "WARN e6 $M failed"
    commit_push "J23 E6: cluster metrics ${M} (job ${SLURM_JOB_ID})" \
        "$D/results/e6_clusters__${M}.json"
done
echo "=== Done. $(date) ==="
