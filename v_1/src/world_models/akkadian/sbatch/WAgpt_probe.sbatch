#!/bin/bash
#SBATCH --job-name=WAgpt_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WAgpt_probe_%j.out
# ===========================================================================
# WAgpt_probe — re-probe gpt-oss-120B Akkadian on CPU only. The WAgpt extraction
# succeeded; only the probes died on fp16-overflow inf (now clamped by
# probing.sanitize), so there is no need to repeat the gpu:8 extraction.
#   sbatch v_1/src/world_models/akkadian/sbatch/WAgpt_probe.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b
echo "=== WAgpt_probe ${M} === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
D=v_1/src/world_models/akkadian
python -u $D/probe_akk.py        --method "$M" || echo "warn: probe_akk ${M}"
python -u $D/probe_geo_site.py   --method "$M" || echo "warn: geo_site ${M}"
python -u $D/probe_layers_pls.py --method "$M" || echo "warn: layers_pls ${M}"
commit_push "WAgpt_probe: gpt-oss-120B Akkadian probes (job ${SLURM_JOB_ID})" \
    $D/results/probes/${M} $D/results/probes_geosite/${M} $D/results/layers_pls/${M}
echo "=== Done ${M}. $(date) ==="
