# S1: embed the SSL corpus with 4 encoders (GPU), then the representation probes (CPU)
J=$(sbatch --parsable chrono/sbatch/C7_ssl_extract.sbatch); echo "C7 array: $J"
K=$(sbatch --parsable --dependency=afterok:${J%%;*} chrono/sbatch/C8_ssl_probes.sbatch); echo "C8 array: $K"
sleep 5; squeue -u "$USER" | head -14
