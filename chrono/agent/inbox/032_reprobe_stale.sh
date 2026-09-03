# TIMEOUT=300
# Re-run only the C14 cells whose S1 table is missing or still carries the
# empty HELD-OUT section (written before the 2026-09-03 probe fix). Waits for
# any C14 array still in the queue so we never run the same cell twice.
LIST=$(python chrono/scripts/c14_stale_cells.py)
LIVE=$(squeue -u "$USER" -h -n C14_reprobe -o %A | sort -u | head -1)
echo "stale cells: ${LIST:-none}   live C14 array: ${LIVE:-none}"
[ -z "$LIST" ] && { echo "nothing to do"; exit 0; }
if [ -n "$LIVE" ]; then
    J=$(sbatch --parsable --dependency=afterany:"$LIVE" --array="$LIST" chrono/sbatch/C14_reprobe.sbatch)
else
    J=$(sbatch --parsable --array="$LIST" chrono/sbatch/C14_reprobe.sbatch)
fi
echo "C14 re-run: $J"
