# TIMEOUT=300
# LEOPARD-style density-matching arm (advisor request). Under the new 2x4
# mapping the leopard cells are indices 3 (barlow) and 7 (jepa); the queued
# arrays keep their own snapshots and are unaffected.
J=$(sbatch --parsable --array=3,7 chrono/sbatch/C19_antishortcut.sbatch); echo "C19 leopard: $J"
K=$(sbatch --parsable --dependency=afterany:"${J%%;*}" chrono/sbatch/C18_emin_ssl.sbatch); echo "C18 refresh: $K"
