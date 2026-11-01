# C7 task died in the census (to_markdown needs 'tabulate'); fixed without the
# dependency. Cancel the orphaned C8 chain, rerun C7 (the store resumes cached
# shards, so completed encoders cost nothing) and re-chain C8.
scancel --name=C8_ssl_probe 2>/dev/null || true
J=$(sbatch --parsable chrono/sbatch/C7_ssl_extract.sbatch); echo "C7 array: $J"
K=$(sbatch --parsable --dependency=afterok:${J%%;*} chrono/sbatch/C8_ssl_probes.sbatch); echo "C8 array: $K"
sleep 5; squeue -u "$USER" | head -14
