# TIMEOUT=300
J=$(sbatch --parsable chrono/sbatch/C18_emin_ssl.sbatch); echo "C18: $J"
