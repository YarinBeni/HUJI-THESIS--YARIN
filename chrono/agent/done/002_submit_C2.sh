# resubmit the P0.4 baseline gate now that constant cells are skipped
sbatch chrono/sbatch/C2_baseline_gate.sbatch
sleep 5
squeue -u "$USER"
