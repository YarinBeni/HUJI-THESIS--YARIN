# The first 7 C14 cells (array 0-6) ran before the held-out-source fix in
# probe_representations.py (empty HELD-OUT table). Re-run just those; the
# pending cells 7-31 pick the fixed script up on their own via sync_sandbox.
J=$(sbatch --parsable --array=0-6 chrono/sbatch/C14_reprobe.sbatch); echo "C14 re-run 0-6: $J"
sleep 5; timeout 60 squeue -u "$USER" -h | wc -l
