# TIMEOUT=600
# The rerun atlas computed with all 16/12/11 layers but its outputs never got
# committed (the job's rebase autostash log shows them left as working-tree
# changes). Show what's sitting there and commit it.
git status --short chrono/reports/tier0/atlas | head -20
source chrono/sbatch/_sandbox.sh
commit_push_sandbox "C25: alignment atlas v2 outputs (rescued from the working tree)" \
    chrono/reports/tier0/atlas
