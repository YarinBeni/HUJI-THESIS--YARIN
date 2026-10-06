# TIMEOUT=300
J=$(sbatch --parsable chrono/sbatch/C25_atlas.sbatch); echo "C25 v3: $J"
