# TIMEOUT=300
J=$(sbatch --parsable chrono/sbatch/C17_transfer.sbatch); echo "C17: $J"
