# TIMEOUT=300
J=$(sbatch --parsable chrono/sbatch/C23c_olmo_eng.sbatch); echo "C23c: $J"
