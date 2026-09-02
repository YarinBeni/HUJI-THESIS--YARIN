# outbox/*.out were gitignored (*.out rule) so 001-004 outputs never landed;
# .gitignore now un-ignores the dir. This pass's commit picks them all up.
ls -la chrono/agent/outbox/
echo "## squeue"; squeue -u "$USER"
echo "## recent"; sacct -u "$USER" --starttime now-6hours --format=JobID,JobName%16,State,Elapsed,ExitCode | grep -v "\.batch\|\.extern"
