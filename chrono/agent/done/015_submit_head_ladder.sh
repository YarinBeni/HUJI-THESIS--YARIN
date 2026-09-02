# P1 head ladder: 3 encoders x {provenance, period, subgenre, length} x 3 seeds (GPU)
sbatch chrono/sbatch/C5_head_ladder.sbatch
sleep 5; squeue -u "$USER" | head -12
