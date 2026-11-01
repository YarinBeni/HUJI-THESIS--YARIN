# TIMEOUT=600
# First the harvest report alone, so the chosen spellings are on the branch
# for eyeballing even before the GPU work; then the two experiment waves.
python3 v_1/src/world_models/akkadian/build_entity_akk.py --report || { echo "harvest FAILED"; exit 1; }
J=$(sbatch --parsable chrono/sbatch/C23_wb_akk.sbatch); echo "C23 WB-akk: $J"
K=$(sbatch --parsable chrono/sbatch/C24_nlmcd.sbatch); echo "C24 NLMCD: $K"
