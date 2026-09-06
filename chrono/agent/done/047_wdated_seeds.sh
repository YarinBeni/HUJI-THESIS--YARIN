# TIMEOUT=300
# The first two wdated arms beat or match the frozen baseline on one seed;
# a single seed is not a claim. Two more seeds for the two best.
for S in 1 2; do
  for OBJ in barlow byol; do
    J=$(sbatch --parsable --export=ALL,WD_OBJ=$OBJ,WD_SEED=$S chrono/sbatch/C20b_wdated_seed.sbatch)
    echo "C20b $OBJ s$S: $J"; LAST=$J
  done
done
K=$(sbatch --parsable --dependency=afterany:"${LAST%%;*}" chrono/sbatch/C18_emin_ssl.sbatch); echo "C18: $K"
