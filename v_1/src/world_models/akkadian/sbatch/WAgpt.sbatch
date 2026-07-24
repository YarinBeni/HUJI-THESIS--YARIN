#!/bin/bash
#SBATCH --job-name=WAgpt
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WAgpt_%j.out
# ===========================================================================
# WAgpt — close the gpt-oss-120B gap on the AKKADIAN side: extract akk_maximal +
# eng_tier0 (512 tok), then year/geo MC, by-site geo R², and layer+PLS probes. gpu:8.
#   sbatch v_1/src/world_models/akkadian/sbatch/WAgpt.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b
echo "=== WAgpt ${M} === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=v_1/src/world_models/akkadian
for V in akk_maximal eng_tier0; do
  python -u $D/extract_akk.py --method "$M" --variant "$V" --max-tokens 512 --batch-size 8 \
      || { echo "FAILED extract ${M}/${V}"; exit 1; }
done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u $D/probe_akk.py        --method "$M" || echo "warn: probe_akk ${M}"
python -u $D/probe_geo_site.py   --method "$M" || echo "warn: geo_site ${M}"
python -u $D/probe_layers_pls.py --method "$M" || echo "warn: layers_pls ${M}"
commit_push "WAgpt: gpt-oss-120B Akkadian acts+probes (job ${SLURM_JOB_ID})" \
    $D/results/probes/${M} $D/results/probes_geosite/${M} $D/results/layers_pls/${M}
echo "=== Done ${M}. $(date) ==="
