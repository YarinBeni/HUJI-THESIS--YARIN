# TIMEOUT=300
# The queued C19 array (34176) was snapshotted with the old 2x2 mapping, so it
# still runs {barlow,jepa} x {leace,adv} correctly. The new 2x3 mapping puts
# the combined leace+adv cells at indices 2 and 5 -- submit just those.
J=$(sbatch --parsable --array=2,5 chrono/sbatch/C19_antishortcut.sbatch); echo "C19 both-arm: $J"
K=$(sbatch --parsable --dependency=afterany:"${J%%;*}" chrono/sbatch/C18_emin_ssl.sbatch); echo "C18 refresh: $K"
