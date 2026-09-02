# S2 from-scratch family: 8 x 5-hour GPU runs (S/M/L/XL x barlow/jepa), probes chained
J=$(sbatch --parsable chrono/sbatch/C11_ssl_e2e.sbatch); echo "C11 array: $J"
K=$(sbatch --parsable --dependency=afterok:${J%%;*} chrono/sbatch/C12_e2e_probes.sbatch); echo "C12 array: $K"
sleep 5; squeue -u "$USER" | head -40
