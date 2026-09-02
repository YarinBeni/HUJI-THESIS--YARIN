# re-run the ladder once the first pass is done, with the fixed readability
# diagnostic (within-train split, frequent classes, before/after). rho columns
# are unchanged; this only makes the erasure check trustworthy.
sbatch --dependency=afterany:33433 chrono/sbatch/C4_ladder.sbatch
sleep 3; squeue -u "$USER" | head -12
