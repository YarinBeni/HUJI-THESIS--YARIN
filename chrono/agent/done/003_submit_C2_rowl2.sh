# C2 again with the M.Sc. probe convention (row-wise L2 before the probe),
# so the a-priori PLS cell is compared like-for-like. Reports get _rowl2.
ROW_L2=1 sbatch chrono/sbatch/C2_baseline_gate.sbatch
sleep 5; squeue -u "$USER"
