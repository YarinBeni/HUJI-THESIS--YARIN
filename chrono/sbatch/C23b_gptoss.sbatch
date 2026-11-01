#!/bin/bash
#SBATCH --job-name=C23b_gptoss
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00

#SBATCH --output=chrono/sbatch/logs/C23b_gptoss_%j.out
# ===========================================================================
# C23 / WB-akk — the missing cell of the entity matrix: the same rulers, the
# same templates and fold-matched split as CELL-B, but the entity string is
# the ruler's ATTESTED TRANSLITERATED NAME harvested from his own
# inscriptions (text_akk vs the [PN]-masked tier). Separates the "language
# cliff" from the "span/aggregation cliff" at the name level.
# ===========================================================================
set -uo pipefail
METHODS=(gpt_oss_120b)
M=${METHODS[0]}
echo "=== C23 WB-akk ${M} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
# WHY gpu:4 not gpu:8: the 8-GPU request sat PENDING for ~2 days — co-locating
# 8 free GPUs on one node is the bottleneck, not compute. gpt-oss-120b loads
# bf16 via device_map=auto at ~240 GB of weights, which fits 4x80GB with room
# for activations (batches here are small probe sentences).
nvidia-smi --query-gpu=name,memory.total --format=csv || true
# periodic log push: every 15 min commit the tail of this job's stdout, so a
# stall is diagnosable from the repo while the job still runs. The subshell
# dies with the job; the EXIT trap still pushes the final log.
( OUT="chrono/sbatch/logs/C23b_gptoss_${SLURM_JOB_ID}.out"
  while sleep 900; do
      dest="chrono/reports/logs/C23b_gptoss_${SLURM_JOB_ID}.progress.log"
      { echo "# progress $(date -u)"; tail -n 300 "$OUT" 2>/dev/null; } > "$dest"
      commit_push_sandbox "logs: C23b ${SLURM_JOB_ID} progress" chrono/reports/logs || true
  done ) &
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
