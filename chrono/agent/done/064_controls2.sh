# TIMEOUT=7200
python3 chrono/scripts/atlas_controls2.py || echo "FAILED controls2"
source chrono/sbatch/_sandbox.sh
commit_push_sandbox "controls2: length/names-by-construction, categories-vs-order, target sensitivity" chrono/reports/tier0/atlas
