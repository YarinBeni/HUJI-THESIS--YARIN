# TIMEOUT=14400
# waits for the 30 seed-0 HSIC heads (6 configs x 5 folds), then runs the
# nonlinear-recovery probe on RAW features for each, and aggregates dating rho.
for i in $(seq 1 230); do n=$(ls chrono/reports/tier0/hsic/heads/*-s0-f*.pt 2>/dev/null | wc -l); [ "$n" -ge 30 ] && break; sleep 60; done
echo "heads present: $(ls chrono/reports/tier0/hsic/heads/*.pt 2>/dev/null | wc -l)"
for E in cunei400m llama2_7b qwen3_8b; do for L in 1 10; do
  C=emin2_${E}_t0_akk_hsic${L}_provenance
  python -u chrono/scripts/probe_head_hidden.py --config chrono/configs/${C}.yaml --no-erase \
      --heads-dir chrono/reports/tier0/hsic/heads --out chrono/reports/tier0/hsic/nonlinear_recovery_${C}.md || echo "WARN probe failed $C"
  python -u chrono/scripts/aggregate_emin.py --run ${C} --scores-dir chrono/reports/tier0/hsic/scores \
      --corpus chrono/artifacts_tier0/corpus_chrono.parquet --splits-dir chrono/artifacts_tier0/splits \
      --out chrono/reports/tier0/hsic/summary_${C}.md || echo "WARN aggregate failed $C"
done; done
git add chrono/reports/tier0/hsic/*.md || true
