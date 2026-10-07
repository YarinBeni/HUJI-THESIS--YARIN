# TIMEOUT=600
# The first harvest let royal ENEMIES through (Teumman for Ashurbanipal,
# Merodach-baladan for Sennacherib): dominance alone prefers the most
# distinctive name, and in annals that is the enemy. The fix harvests only
# each document's FIRST personal name (the titulary). Cancel the queued C23
# (it would have built the contaminated CSV), drop any CSV already built,
# print the new report, resubmit.
scancel 53143 2>/dev/null || true
rm -f v_1/src/world_models/data/entity_datasets/assyrian_ruler_akk.csv
python3 v_1/src/world_models/akkadian/build_entity_akk.py --report || { echo "harvest FAILED"; exit 1; }
J=$(sbatch --parsable chrono/sbatch/C23_wb_akk.sbatch); echo "C23 resubmitted: $J"
