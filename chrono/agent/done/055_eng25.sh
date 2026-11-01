# TIMEOUT=3600
# Fairness cell: the English-name probe restricted to exactly the rulers
# whose Akkadian spelling survived, so the eng<->akk entity comparison is on
# identical entity sets. CPU only — activations already exist.
for M in llama2_7b qwen3_8b thalesian_cunei400m; do
  python3 v_1/src/world_models/akkadian/probe_entity.py --method "$M" \
      --entity-type assyrian_ruler_eng25 || echo "FAILED $M"
done
python3 v_1/src/world_models/akkadian/aggregate_entity.py || echo "WARN aggregate"
