# S2: embed all SSL views (20 GPU tasks, sharded) then the 16-run SSL grid + probes
J=$(sbatch --parsable chrono/sbatch/C9_ssl_views_extract.sbatch); echo "C9 array: $J"
K=$(sbatch --parsable --dependency=afterok:${J%%;*} chrono/sbatch/C10_ssl_train.sbatch); echo "C10 array: $K"
sleep 5; squeue -u "$USER" | head -30
