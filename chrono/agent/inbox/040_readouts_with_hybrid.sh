# TIMEOUT=600
# The hybrid family is trained and probed; fold it into the two tables that
# matter by re-running the readouts over the store as it now stands.
J=$(sbatch --parsable chrono/sbatch/C18_emin_ssl.sbatch); echo "C18: $J"
K=$(sbatch --parsable chrono/sbatch/C17_transfer.sbatch); echo "C17: $K"
