# TIMEOUT=900
# Second push outage of the day: jobs that were already running when the
# .gitignore fix landed committed the leftover e2e checkpoints again, so the
# 18 local commits carry hundreds of MB and pack-objects is OOM-killed.
# Squash them without files > 20 MB (the checkpoints stay on disk under
# artifacts_ssl), push, then queue the missing/pre-fix probe cells.
git config pack.windowMemory 32m; git config pack.threads 1
git fetch -q origin yarin-sandbox
echo "unpushed before: $(git rev-list --count FETCH_HEAD..HEAD)"
git reset -q --soft FETCH_HEAD
big=$(git diff --cached --name-only --diff-filter=AM | while read -r f; do
        [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 20000000 ] && echo "$f"; done)
echo "left out: ${big:-none}"
[ -n "$big" ] && git reset -q -- $big
git diff --cached --quiet || git commit -qm "cluster: C13/C14 results flushed (second push outage)"
git push origin HEAD:yarin-sandbox 2>&1 | tail -2
LIST=$(python chrono/scripts/c14_stale_cells.py)
LIVE=$(squeue -u "$USER" -h -n C14_reprobe -o %A | sort -u | head -1)
echo "stale cells: ${LIST:-none}   live C14: ${LIVE:-none}"
[ -z "$LIST" ] && exit 0
if [ -n "$LIVE" ]; then
    sbatch --parsable --dependency=afterany:"$LIVE" --array="$LIST" chrono/sbatch/C14_reprobe.sbatch
else
    sbatch --parsable --array="$LIST" chrono/sbatch/C14_reprobe.sbatch
fi
