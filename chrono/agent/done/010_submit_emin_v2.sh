# E-MIN v2 (review 2026-09-02): tier0 Akkadian, three encoders, three language
# arms. C1v2 extracts (3 GPU tasks); C3v2 (45 tasks) waits for the whole array.
J=$(sbatch --parsable chrono/sbatch/C1v2_extract.sbatch)
echo "C1v2 array: $J"
K=$(sbatch --parsable --dependency=afterok:${J%%;*} chrono/sbatch/C3v2_emin.sbatch)
echo "C3v2 array: $K (afterok:${J%%;*})"
sleep 5; squeue -u "$USER" | head -20
