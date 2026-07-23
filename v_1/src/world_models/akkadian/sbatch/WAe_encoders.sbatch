#!/bin/bash
#SBATCH --job-name=WAe_encoders
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WAe_encoders_%A_%a.out
#SBATCH --array=0-2
# ===========================================================================
# WAe — bring the 3 thesis ENCODERS into the Akkadian layer/PLS/geo plots. Extract
# akk_maximal + eng_tier0 (mean pooling is the meaningful one for encoders), then run
# the year/geo MC, by-site geo, and layer+PLS probes for that arm. gpu:1, small models.
#   0 thalesian_akk300m  1 thalesian_cunei400m  2 umt5_base
#   sbatch v_1/src/world_models/akkadian/sbatch/WAe_encoders.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(thalesian_akk300m thalesian_cunei400m umt5_base)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WAe ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
python -c "import sentencepiece" 2>/dev/null || pip install -q sentencepiece
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
D=v_1/src/world_models/akkadian
for V in akk_maximal eng_tier0; do
  python -u $D/extract_akk.py --method "$M" --variant "$V" --max-tokens 512 \
      || { echo "FAILED extract ${M}/${V}"; exit 1; }
done
# probes (CPU) over the freshly extracted encoder acts
python -u $D/probe_akk.py        --method "$M" || echo "warn: probe_akk ${M}"
python -u $D/probe_geo_site.py   --method "$M" || echo "warn: geo_site ${M}"
python -u $D/probe_layers_pls.py --method "$M" || echo "warn: layers_pls ${M}"
commit_push "WAe: encoder ${M} Akkadian acts+probes (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    $D/results/probes/${M} $D/results/probes_geosite/${M} $D/results/layers_pls/${M}
echo "=== Done ${M}. $(date) ==="
