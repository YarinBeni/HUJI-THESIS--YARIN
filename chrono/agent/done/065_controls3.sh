# TIMEOUT=7200
python3 chrono/scripts/atlas_controls3.py || echo "FAILED controls3"
source chrono/sbatch/_sandbox.sh
commit_push_sandbox "controls3: mask-only alignment, order-test diagnostics" chrono/reports/tier0/atlas
