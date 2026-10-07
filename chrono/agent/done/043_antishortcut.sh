# TIMEOUT=300
J=$(sbatch --parsable chrono/sbatch/C19_antishortcut.sbatch); echo "C19: $J"
K=$(sbatch --parsable --dependency=afterany:"${J%%;*}" chrono/sbatch/C18_emin_ssl.sbatch); echo "C18 refresh: $K"
