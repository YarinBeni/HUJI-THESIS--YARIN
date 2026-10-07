# C1v2 33341 died on the shared-manifest race (all three tasks); the store is
# now locked + atomic. The tier0 store holds half-written legacy shards from
# that run -> wipe it and re-extract cleanly. C3v2 33342 can never satisfy
# its dependency any more -> cancel and re-chain.
scancel 33342 2>/dev/null || true
rm -rf chrono/artifacts_tier0/emb_store
J=$(sbatch --parsable chrono/sbatch/C1v2_extract.sbatch)
echo "C1v2 array: $J"
K=$(sbatch --parsable --dependency=afterok:${J%%;*} chrono/sbatch/C3v2_emin.sbatch)
echo "C3v2 array: $K (afterok:${J%%;*})"
sleep 5; squeue -u "$USER" | head -12
