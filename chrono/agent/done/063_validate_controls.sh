# TIMEOUT=5400
# Advisor review: (1) prove the name harvester with measurements, not trust —
# histograms, coverage, ruler-identification, leakage (E1–E4); (2) price the
# Fig B alignment against chance and length-only clustering. Both CPU.
python3 v_1/src/world_models/akkadian/validate_entity_akk.py || echo "FAILED validate"
python3 chrono/scripts/atlas_controls.py || echo "FAILED controls"
source chrono/sbatch/_sandbox.sh
commit_push_sandbox "validation: harvester E1-E4 + Fig B chance/length controls" \
    v_1/src/world_models/akkadian/results chrono/reports/tier0/atlas
