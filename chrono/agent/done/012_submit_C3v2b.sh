# ridge trained on all views, per encoder x arm -- the fair comparator for the head
sbatch chrono/sbatch/C3v2b_baselines.sbatch
sleep 5; squeue -u "$USER" | head -15
