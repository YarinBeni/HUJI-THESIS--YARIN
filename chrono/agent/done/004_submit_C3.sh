# C2 reproduced the M.Sc. signal (clean nulls, ridge in band) -> C3 unblocked.
# E-MIN: 5 seeds x 5 folds, one array task per seed. First contact with real
# data for train_cjb; if it dies the tasks push their logs and get fixed.
sbatch chrono/sbatch/C3_emin.sbatch
sleep 5; squeue -u "$USER"
