# TIMEOUT=600
J=$(sbatch --parsable chrono/sbatch/C18_emin_ssl.sbatch); echo "C18 with family summary: $J"
