# TIMEOUT=300
# atlas v2 ran while two C26 arms were still extracting (the chain released
# early); rerun now that all deep layers are in the store.
J=$(sbatch --parsable chrono/sbatch/C25_atlas.sbatch); echo "C25 rerun: $J"
