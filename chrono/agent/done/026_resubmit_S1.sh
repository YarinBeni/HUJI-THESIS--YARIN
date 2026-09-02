# fragment_ids collide across sources (letters vs unified) -> store ids are now
# 'ssl::<source>::<fragment_id>' and letters only add texts absent from unified.
# Rebuild the corpus on the cluster, cancel orphaned chains, resubmit C7 -> C8.
scancel --name=C8_ssl_probe 2>/dev/null || true; scancel --name=C7_ssl_ext 2>/dev/null || true
rm -f chrono/artifacts_ssl/corpus_all.parquet chrono/artifacts_ssl/CENSUS.md
J=$(sbatch --parsable chrono/sbatch/C7_ssl_extract.sbatch); echo "C7 array: $J"
K=$(sbatch --parsable --dependency=afterok:${J%%;*} chrono/sbatch/C8_ssl_probes.sbatch); echo "C8 array: $K"
sleep 5; squeue -u "$USER" | head -14
