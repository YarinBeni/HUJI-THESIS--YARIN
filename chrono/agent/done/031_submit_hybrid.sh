# hybrid family: frozen token states -> fresh Transformer (8 x 5h GPU) then its probes
J=$(sbatch --parsable chrono/sbatch/C15_hybrid.sbatch); echo "C15 array: $J"
K=$(sbatch --parsable --dependency=afterany:${J%%;*} chrono/sbatch/C16_probe_hyb.sbatch); echo "C16 array: $K"
sleep 5; squeue -u "$USER" | head -20
