# TIMEOUT=10800
# waits for the 15 saved heads from C5b, then asks linear + MLP probes whether
# provenance is recoverable from the head's hidden layer after LEACE erasure
for i in $(seq 1 150); do
  n=$(ls chrono/reports/tier0/heads/*.pt 2>/dev/null | wc -l)
  [ "$n" -ge 15 ] && break; sleep 60
done
echo "heads present: $(ls chrono/reports/tier0/heads/*.pt 2>/dev/null | wc -l)"
for E in cunei400m llama2_7b qwen3_8b; do
  python -u chrono/scripts/probe_head_hidden.py --config chrono/configs/emin2_${E}_t0_akk_erase_provenance.yaml || echo "WARN probe failed for $E"
done
git add chrono/reports/tier0/ladder/nonlinear_recovery_*.md
