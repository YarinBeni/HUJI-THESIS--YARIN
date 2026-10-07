# TIMEOUT=3600
WB=v_1/src/world_models/akkadian
python3 ${WB}/probe_entity.py --method gpt_oss_120b --entity-type assyrian_ruler_eng25 || echo "FAILED probe eng25"
python3 ${WB}/aggregate_entity.py || echo "WARN aggregate failed"
source chrono/sbatch/_sandbox.sh
commit_push_sandbox "C23: gpt-oss eng25 fairness cell" v_1/src/world_models/akkadian/results
