echo "## squeue"; squeue -u "$USER"
echo "## recent (6h)"; sacct -u "$USER" --starttime now-6hours --format=JobID,JobName%16,State,Elapsed,ExitCode | grep -v "\.batch\|\.extern" | tail -30
echo "## ladder dir"; ls -la chrono/reports/tier0/ladder/ | head -20
echo "## heads"; ls chrono/reports/tier0/heads 2>/dev/null | head
