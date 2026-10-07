# TIMEOUT=3600
# the plan's "ridge +/- mask views" arm: ridge fitted on ALL training views
# (augmentation as data augmentation, no invariance loss). If this matches the
# head, the Barlow objective adds nothing beyond seeing the views.
python -u chrono/scripts/baseline_conditions.py --probes ridge --train-views all
python -u chrono/scripts/aggregate_emin.py --run baseline_ridge_L8mean_allviews --out chrono/reports/emin_baseline_ridge_L8mean_allviews.md
git add -A chrono/reports/scores chrono/reports/emin_*.md
