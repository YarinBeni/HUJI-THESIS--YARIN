# TIMEOUT=300
# Six hybrid runs trained their 5 h and were killed by the wall limit before
# they could write embeddings, so C16 found nothing for them. They resume from
# ckpt.pt: resubmit with a 6-minute training budget so the job goes straight
# to the embedding pass, which is all that is missing.
J=$(sbatch --parsable --array=1,2,3,4,6,7 --export=ALL,HYB_HOURS=0.1 chrono/sbatch/C15_hybrid.sbatch)
echo "C15 embed-only: $J"
K=$(sbatch --parsable --dependency=afterany:"${J%%;*}" chrono/sbatch/C16_probe_hyb.sbatch)
echo "C16 probes: $K"
L=$(sbatch --parsable --dependency=afterany:"${K%%;*}" chrono/sbatch/C18_emin_ssl.sbatch)
echo "C18 refresh: $L"
