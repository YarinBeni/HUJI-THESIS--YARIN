# re-run erase_provenance heads with saving on (3 GPU tasks) for the nonlinear check
sbatch chrono/sbatch/C5b_save_heads.sbatch
sleep 5; squeue -u "$USER" | head -12
