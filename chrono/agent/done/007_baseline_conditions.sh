# TIMEOUT=3600
# the other half of the E-MIN comparison: ridge / PLS fit on orig views,
# read out on every masked/cropped condition, same folds and read-out as
# the head. Two passes: all languages (what the head sees) and akk-only.
python -u chrono/scripts/baseline_conditions.py --probes ridge pls
python -u chrono/scripts/baseline_conditions.py --probes ridge pls --langs akk
python -u chrono/scripts/aggregate_emin.py --run baseline_ridge_L8mean     --out chrono/reports/emin_baseline_ridge.md
python -u chrono/scripts/aggregate_emin.py --run baseline_pls_L8mean       --out chrono/reports/emin_baseline_pls.md
python -u chrono/scripts/aggregate_emin.py --run baseline_ridge_L8mean_akk --out chrono/reports/emin_baseline_ridge_akk.md
git add chrono/reports/scores chrono/reports/emin_baseline_*.md   # picked up by the runner's commit
