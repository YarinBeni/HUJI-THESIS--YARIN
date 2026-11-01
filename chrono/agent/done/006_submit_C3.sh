# C3 again: emin_thalesian.yaml asked the store for L11, which this encoder
# does not have (same 12-block assumption as the C1 bug). Now L8.
sbatch chrono/sbatch/C3_emin.sbatch
sleep 5; squeue -u "$USER"
