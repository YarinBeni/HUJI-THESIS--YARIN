#!/bin/bash
#SBATCH --job-name=mc_sanity
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/linear_probing/logs/mc_sanity_%j.out
# Validate the joblib layer-parallelism: run ONE draw of qwen3_1b7_cls_numeric
# both sequentially (--n-jobs 1) and in parallel (--n-jobs 8) to scratch dirs,
# then assert the per-config spearman/mae/r2 match. Parallelism must not change
# numerics. Does NOT touch the real probes/ dir and does NOT commit.

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARN: git pull failed"

BASE=v_1/src/linear_probing/results/orcc_round2_phase0
SEQ=$BASE/_sanity_seq
PAR=$BASE/_sanity_par
rm -rf "$SEQ" "$PAR"; mkdir -p "$SEQ" "$PAR"

COMMON=(--draws-matrix   $BASE/balanced_subset/draws_matrix.npy
        --fragment-order $BASE/balanced_subset/corpus_fragment_order.json
        --probes         qwen3_1b7_cls_numeric
        --layers         all
        --draws-range    0-0)

echo "--- sequential (n-jobs=1) ---"
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    "${COMMON[@]}" --output-dir "$SEQ" --n-jobs 1

echo "--- parallel (n-jobs=8) ---"
python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    "${COMMON[@]}" --output-dir "$PAR" --n-jobs 8

echo "--- compare ---"
python - "$SEQ" "$PAR" <<'PY'
import json, sys, glob, math
seq_dir, par_dir = sys.argv[1], sys.argv[2]
def load(d):
    f = glob.glob(f"{d}/qwen3_1b7_cls_numeric__mc_balanced__draw000.json")[0]
    return json.load(open(f))["results"]
a, b = load(seq_dir), load(par_dir)
assert set(a) == set(b), f"config keys differ: {set(a)^set(b)}"
bad = 0
for k in a:
    for m in ("spearman_mean", "mae_mean", "r2_mean"):
        va, vb = a[k].get(m), b[k].get(m)
        if va is None and vb is None: continue
        if va is None or vb is None or (isinstance(va,float) and math.isnan(va)) != (isinstance(vb,float) and math.isnan(vb)):
            print(f"  MISMATCH {k}.{m}: {va} vs {vb}"); bad += 1; continue
        if isinstance(va,float) and math.isnan(va): continue
        if abs(va - vb) > 1e-9:
            print(f"  MISMATCH {k}.{m}: {va} vs {vb}"); bad += 1
print(f"configs={len(a)}  mismatches={bad}")
sys.exit(1 if bad else 0)
PY
echo "=== SANITY PASS: parallel == sequential ==="
rm -rf "$SEQ" "$PAR"
