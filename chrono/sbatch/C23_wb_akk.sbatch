#!/bin/bash
#SBATCH --job-name=C23_wb_akk
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=chrono/sbatch/logs/C23_wb_akk_%A_%a.out
# ===========================================================================
# C23 / WB-akk — the missing cell of the entity matrix: the same rulers, the
# same templates and fold-matched split as CELL-B, but the entity string is
# the ruler's ATTESTED TRANSLITERATED NAME harvested from his own
# inscriptions (text_akk vs the [PN]-masked tier). Separates the "language
# cliff" from the "span/aggregation cliff" at the name level.
# ===========================================================================
set -uo pipefail
METHODS=(llama2_7b qwen3_8b thalesian_cunei400m olmo2_7b)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "=== C23 WB-akk ${M} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
WB=v_1/src/world_models/akkadian
# dataset build is idempotent and cheap; the flock keeps array tasks off each other
( flock -w 600 9
  [ -f v_1/src/world_models/data/entity_datasets/assyrian_ruler_akk.csv ] \
      || python -u ${WB}/build_entity_akk.py || exit 1 ) 9>"${GIT_LOCK}.wbakk" || exit 1
python -u ${WB}/extract_entity.py --method "${M}" --entity-type assyrian_ruler_akk \
    || { echo "FAILED extract ${M}"; exit 1; }
python -u ${WB}/probe_entity.py --method "${M}" --entity-type assyrian_ruler_akk \
    || { echo "FAILED probe ${M}"; exit 1; }
python -u ${WB}/aggregate_entity.py || echo "WARN aggregate failed"
commit_push_sandbox "C23: WB-akk ${M} (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/akkadian/results v_1/src/world_models/data/entity_datasets
echo "=== Done. $(date) ==="
