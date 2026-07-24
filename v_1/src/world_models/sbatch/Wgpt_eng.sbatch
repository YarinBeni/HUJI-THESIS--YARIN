#!/bin/bash
#SBATCH --job-name=Wgpt_eng
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=v_1/src/world_models/logs/Wgpt_eng_%j.out
# ===========================================================================
# Wgpt_eng — close the gpt-oss-120B gap on the ENGLISH side: extract (last+mean) over
# the 6 G&T datasets, then ridge layer probes + best-layer PLS-k. gpu:8.
#   sbatch v_1/src/world_models/sbatch/Wgpt_eng.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b
echo "=== Wgpt_eng ${M} === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u v_1/src/world_models/extract_acts.py --method "$M" --sites last,mean --batch-size 32 \
    || { echo "FAILED extract ${M}"; exit 1; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u v_1/src/world_models/probe_wm.py       --method "$M" || echo "warn: probe_wm ${M}"
python -u v_1/src/world_models/probe_eng_pls.py  --method "$M" || echo "warn: eng_pls ${M}"
commit_push "Wgpt_eng: gpt-oss-120B English acts+probes (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/results/probes/${M} \
    v_1/src/world_models/results/eng_pls/${M}
echo "=== Done ${M}. $(date) ==="
