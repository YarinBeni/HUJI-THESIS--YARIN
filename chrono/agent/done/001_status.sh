# first contact: prove the channel works and show the cluster state
echo "## whoami/host"; whoami; hostname
echo "## squeue"; squeue -u "$USER"
echo "## recent jobs"
sacct -u "$USER" --starttime now-3days \
  --format=JobID,JobName%16,State,Elapsed,ExitCode,End | grep -v "\.batch\|\.extern"
echo "## git"; git status -sb | head -5; git log --oneline -3
echo "## emb_store"
du -sh chrono/artifacts/emb_store 2>/dev/null
ls chrono/artifacts/emb_store 2>/dev/null | head
echo "## extract meta"
cat chrono/artifacts/emb_store/extract_meta__thalesian_akk300m.json 2>/dev/null
echo "## can sbatch run from here?"; which sbatch squeue sacct
