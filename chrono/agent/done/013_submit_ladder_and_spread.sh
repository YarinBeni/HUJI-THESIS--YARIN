# P1 erasure ladder (6 CPU tasks) + baseline seed spread (6 CPU tasks)
sbatch chrono/sbatch/C4_ladder.sbatch
sbatch chrono/sbatch/C3v2c_baseline_spread.sbatch
sleep 5; squeue -u "$USER" | head -20
