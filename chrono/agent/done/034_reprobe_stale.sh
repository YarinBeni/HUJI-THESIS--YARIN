# TIMEOUT=300
# The backlog is already pushed (033 reported "unpushed before: 0" before it
# collided with a job's git lock and exited 1). All that is left is to run the
# probe cells that are missing or were written before the held-out-source fix.
# NO git operations here on purpose: 033 died on '.git/HEAD.lock' held by a
# job pushing at the same moment, and this script has nothing to commit.
LIST=$(python3 chrono/scripts/c14_stale_cells.py) || { echo "stale-cell check FAILED"; exit 1; }
ARRAYS=$(squeue -u "$USER" -h -n C14_reprobe -o %F | sed 's/_.*//' | sort -u)
LIVE=$(echo "$ARRAYS" | head -1)
echo "stale cells: ${LIST:-none}   live C14 array: ${LIVE:-none}"
[ -z "$LIST" ] && { echo "nothing stale"; exit 0; }
[ "$(echo "$ARRAYS" | grep -c .)" -ge 2 ] && { echo "a re-probe array is already queued"; exit 0; }
if [ -n "$LIVE" ]; then
    sbatch --parsable --dependency=afterany:"$LIVE" --array="$LIST" chrono/sbatch/C14_reprobe.sbatch
else
    sbatch --parsable --array="$LIST" chrono/sbatch/C14_reprobe.sbatch
fi
