#!/bin/bash
#SBATCH --job-name=WB0_build
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/world_models/logs/WB0_build_%j.out
# ===========================================================================
# WB0 — build the CELL-B entity-level datasets (CPU, seconds). Run this FIRST;
# every other WB job reads the two CSVs it commits.
#   assyrian_ruler.csv      34 rulers  x 6 templates = 204 rows  (-> year)
#   mesopotamian_place.csv  25 places  x 6 templates = 150 rows  (-> lon/lat)
# Also runs the TF-IDF floor on the same splits, since it needs no GPU.
#   sbatch v_1/src/world_models/akkadian/sbatch/WB0_build_datasets.sbatch
# ===========================================================================
set -uo pipefail
echo "=== WB0 build entity datasets === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs

python -u v_1/src/world_models/akkadian/build_entity_datasets.py \
    || { echo "FAILED build"; exit 1; }
python -u v_1/src/world_models/akkadian/probe_entity.py --tfidf \
    || { echo "FAILED tfidf floor"; exit 1; }

commit_push "WB0: cell-B entity datasets + tfidf floor (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/data/entity_datasets/assyrian_ruler.csv \
    v_1/src/world_models/data/entity_datasets/mesopotamian_place.csv \
    v_1/src/world_models/akkadian/results/probes_entity/tfidf
echo "=== Done. $(date) ==="
