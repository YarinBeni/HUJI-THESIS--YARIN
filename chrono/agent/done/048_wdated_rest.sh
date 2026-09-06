# TIMEOUT=300
# "מה עם שאר השיטות" — complete the wdated grid where it can still teach us
# something: the fourth objective (infonce), source-erasure on the two arms
# that gained the most (barlow, byol), and one from-scratch run (barlow S,
# the best of its family) with the dated text included.
for OBJ in infonce; do
  J=$(sbatch --parsable --export=ALL,WD_OBJ=$OBJ,WD_SEED=0 chrono/sbatch/C20b_wdated_seed.sbatch)
  echo "C20b $OBJ s0: $J"; LAST=$J
done
for OBJ in barlow byol; do
  J=$(sbatch --parsable --export=ALL,WD_OBJ=$OBJ,WD_SEED=0,WD_ANTI=leace chrono/sbatch/C20b_wdated_seed.sbatch)
  echo "C20b $OBJ+leace s0: $J"; LAST=$J
done
J=$(sbatch --parsable chrono/sbatch/C21_e2e_wdated.sbatch); echo "C21 e2e barlow S wdated: $J"; LAST=$J
K=$(sbatch --parsable --dependency=afterany:"${LAST%%;*}" chrono/sbatch/C18_emin_ssl.sbatch); echo "C18: $K"
