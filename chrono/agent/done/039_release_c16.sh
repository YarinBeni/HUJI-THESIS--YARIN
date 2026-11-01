# TIMEOUT=300
# C15 finished all 8 hybrid runs at 02:13 UTC; C16 (their probes) was
# submitted with --dependency=afterany on the C15 array and has not started
# since. Report why, and if the dependency can never be satisfied — the state
# that killed C8 earlier in this project — cancel and resubmit it clean.
squeue -u "$USER" -h -n C16_probe_hyb -o "%F %T %R" | sort -u
STUCK=$(squeue -u "$USER" -h -n C16_probe_hyb -o "%F|%R" | grep -i 'never' | cut -d'|' -f1 | sort -u)
LIVE=$(squeue -u "$USER" -h -n C16_probe_hyb -o %F | sort -u | grep -c .)
if [ -n "$STUCK" ]; then
    echo "cancelling unsatisfiable: $STUCK"; scancel $STUCK
    J=$(sbatch --parsable chrono/sbatch/C16_probe_hyb.sbatch); echo "C16 resubmitted: $J"
elif [ "$LIVE" -eq 0 ]; then
    echo "no C16 in the queue at all; submitting"
    J=$(sbatch --parsable chrono/sbatch/C16_probe_hyb.sbatch); echo "C16 submitted: $J"
else
    echo "C16 is queued and its dependency is still satisfiable; leaving it alone"
fi
