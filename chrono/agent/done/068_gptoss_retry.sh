# TIMEOUT=300
scancel 53183 && echo "cancelled 53183" || echo "53183 already gone"
J=$(sbatch --parsable chrono/sbatch/C23b_gptoss.sbatch); echo "C23b v2 (gpu:4): $J"
squeue -u "$USER" -o "%F %j %T %R" | head -10
