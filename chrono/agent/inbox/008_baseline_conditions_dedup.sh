# TIMEOUT=3600
# rerun 007 with training rows deduplicated by text (see baseline_conditions.py
# BUG FIX note); three passes: all languages (what the head sees), akk, eng.
rm -f chrono/reports/scores/baseline_*.parquet
python -u chrono/scripts/baseline_conditions.py --probes ridge pls
python -u chrono/scripts/baseline_conditions.py --probes ridge pls --langs akk
python -u chrono/scripts/baseline_conditions.py --probes ridge pls --langs eng
for r in baseline_ridge_L8mean baseline_pls_L8mean baseline_ridge_L8mean_akk baseline_pls_L8mean_akk baseline_ridge_L8mean_eng; do
  python -u chrono/scripts/aggregate_emin.py --run $r --out chrono/reports/emin_$r.md
done
git add -A chrono/reports/scores chrono/reports/emin_*.md
