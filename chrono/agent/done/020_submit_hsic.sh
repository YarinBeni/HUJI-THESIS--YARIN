# P2 first step: HSIC-deconfounded heads (18 GPU tasks)
sbatch chrono/sbatch/C6_hsic.sbatch
sleep 5; squeue -u "$USER" | head -12
