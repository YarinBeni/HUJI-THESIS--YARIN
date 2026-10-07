# S3 fine-tune sweep, chained after the C10 adapter grid (whose heads it initialises from)
C10=$(squeue -u "$USER" -h --name=C10_ssl -o %A | head -1)
if [ -n "$C10" ]; then J=$(sbatch --parsable --dependency=afterany:${C10} chrono/sbatch/C13_finetune.sbatch); echo "C13 array: $J (afterany:$C10)"
else J=$(sbatch --parsable chrono/sbatch/C13_finetune.sbatch); echo "C13 array: $J (C10 not in queue -> no dependency)"; fi
sleep 5; squeue -u "$USER" | head -12
