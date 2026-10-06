# TIMEOUT=300
J=$(sbatch --parsable chrono/sbatch/C27_lenctl.sbatch); echo "C27 lenctl v2 (store-root fix): $J"
