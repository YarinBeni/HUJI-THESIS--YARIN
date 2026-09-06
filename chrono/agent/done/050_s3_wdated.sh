# TIMEOUT=300
# S3 on the winning representations: supervised head initialised from the
# wdated SSL checkpoints (the advisor's two-stage pipeline, now with the SSL
# stage that actually moved), vs the init-none control already in s3_scores.
J=$(sbatch --parsable chrono/sbatch/C22_s3_wdated.sbatch); echo "C22: $J"
