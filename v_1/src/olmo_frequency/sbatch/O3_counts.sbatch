#!/bin/bash
#SBATCH --job-name=O3_counts
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/world_models/logs/O3_counts_%j.out
# ===========================================================================
# O3 — count how often each entity appears in OLMo's training data.
#
# No GPU, no model, barely any CPU: this is ~7.5k HTTPS calls to the free
# infini-gram API, one per held-out entity, plus the 34 Assyrian rulers.
#
# TWO THINGS THIS JOB NEEDS THAT THE OTHERS DID NOT:
#
#   1. OUTBOUND HTTPS. Compute nodes on some clusters have none. If the job dies
#      immediately with "no usable infini-gram index", run the same python line on
#      the LOGIN node instead — the script is resumable, so a partial run from
#      either place is picked up by the next.
#
#   2. THE BEST-LAYER PROJECTIONS. O2's commit_push listed probes/, eng_pls/ and
#      probes_entity/ but not results/projections/, so OLMo's per-entity
#      predictions are sitting on this filesystem uncommitted. They are what the
#      frequency correlation is computed against, so this job commits them first.
#
# Counting is the slow half and is rate-limited, not compute-bound; --sleep is the
# knob if the API starts returning 429.
#
#   sbatch v_1/src/olmo_frequency/sbatch/O3_counts.sbatch
# ===========================================================================
set -uo pipefail

echo "=== O3 counts === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs

echo "--- the per-entity predictions O2 left behind ---"
commit_push "O3: OLMo best-layer projections (the per-entity predictions O2 omitted)" \
    v_1/src/world_models/results/projections/olmo2_7b \
    v_1/src/world_models/results/projections/olmo2_7b_random

echo "--- which infini-gram indexes are served here ---"
python -u v_1/src/olmo_frequency/count_frequencies.py --list-indexes

echo "--- counting ---"
python -u v_1/src/olmo_frequency/count_frequencies.py \
    || { echo "FAILED counting — rerun to resume, nothing counted is lost"; exit 1; }

commit_push "O3: entity training-data counts from OLMo's corpus (job ${SLURM_JOB_ID})" \
    v_1/src/olmo_frequency/results/entity_counts.csv
echo "=== Done. $(date) ==="
