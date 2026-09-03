# re-run every probe cell with the filename fix (C10/C12 probes died on '::' paths)
J=$(sbatch --parsable chrono/sbatch/C14_reprobe.sbatch); echo "C14 array: $J"
sleep 5; squeue -u "$USER" | head -20
