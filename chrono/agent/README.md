# chrono/agent — the git-driven cluster runner

**What it is.** A small, CPU-only slurm job (`chrono/sbatch/AGENT_runner.sbatch`
→ `chrono/agent/runner.sh`) that turns the `yarin-sandbox` branch into a
command channel, so that whoever works on this branch from the outside can
submit jobs and read logs **without anyone copy-pasting into the browser
terminal**.

```
outside            GitHub (yarin-sandbox)           cluster runner (every 60 s)
───────            ──────────────────────           ───────────────────────────
push inbox/NNN.sh ──►  chrono/agent/inbox/  ──pull──►  bash -e NNN.sh   (timeout)
read outbox/      ◄──  chrono/agent/outbox/ ◄──push──  stdout+stderr → NNN.out
                       chrono/agent/done/   ◄──push──  the script, moved here
                       chrono/agent/heartbeat ◄─push─  "alive …" once an hour
```

An inbox script may itself run `sbatch chrono/sbatch/C2_baseline_gate.sbatch`
— so the runner is the one that submits the real GPU jobs. Those jobs push
their own logs (`chrono/reports/logs/`), independently of the runner.

## Rules of the runner (agreed 2026-09-01)

1. **Lives one day, not one week.** `--time=1-00:00:00`. A forgotten
   runner dies by itself within 24 h.
2. **Started per working session, stopped at its end.** Start:
   `sbatch chrono/sbatch/AGENT_runner.sbatch` (one line, from the repo
   root). Stop: either push an empty file `chrono/agent/STOP` to the branch
   (the runner sees it on its next poll, commits "agent: stopped", exits),
   or `scancel <jobid>`.
3. **Whoever drives it from the outside must remind Yarin to stop it at the
   end of the session, and must stop it when told.** This includes the
   AI assistant working on this branch: end-of-session checklist = "is the
   runner still alive? → push STOP → confirm the `agent: stopped` commit".
4. **Anything committed to `inbox/` runs as the cluster user.** Only
   people with push access to this repository can do that. Do not widen
   push access while a runner is alive.
5. Scripts run with `bash -e` from the repo root, conda env `thesis`
   active, default timeout 30 min (override with a line `# TIMEOUT=7200`).
   One script at a time, in filename order; a script already present in
   `done/` is never re-run — use a new number.

## How to tell whether it is alive

- `chrono/agent/heartbeat` on the branch: time of last beat, host, job id,
  current `squeue`. Older than ~2 h ⇒ the runner is gone (expired,
  cancelled, or the node died) — re-submit.
- The runner also restarts itself when `runner.sh` changes on the branch,
  so fixes to it need no manual restart.
- **Alive but silent = pushes are failing.** If `squeue` shows the runner
  running while the branch gets no commits, the cluster cannot push
  (the repo is public, so `fetch` still works without credentials — only
  `push` needs the token). Nothing is lost: every job and the runner commit
  locally and their commits ride along with the next push that succeeds.
  Diagnose on the login node with
  `tail -5 chrono/sbatch/logs/AGENT_runner_<jobid>.out` (the runner logs
  git's reason after `push failed`) or
  `git push origin HEAD:yarin-sandbox 2>&1 | tail -4`. The usual cause is
  an **expired token** (2026-09-03: ~6 h of finished jobs went unpushed).
  Fix: Yarin creates a new fine-grained token (this repo only, Contents
  read/write, ≥ 90 days) and sets it himself on the cluster with
  `git remote set-url origin https://<TOKEN>@github.com/YarinBeni/HUJI-THESIS--YARIN.git`.
  The token is never pasted into the chat with the assistant.

## Flushing the cluster after a push outage

Symptom: `git push` from the cluster dies with `the remote end hung up
unexpectedly … while reading sideband packet` (2026-09-03, ~8 h). That is a
cut transfer, not an auth error; hours of finished jobs then sit as local
commits, and every later push carries them all, so every push dies. Do not
rewrite history — push the backlog in slices, oldest first (each slice is a
fast-forward of the branch), with the runner stopped:

```bash
cd ~/projects/HUJI-THESIS--YARIN
git config http.postBuffer 524288000 && git config http.version HTTP/1.1
git fetch -q origin yarin-sandbox && git rebase -q --autostash FETCH_HEAD
echo "unpushed: $(git rev-list --count FETCH_HEAD..HEAD)"
for c in $(git rev-list --reverse FETCH_HEAD..HEAD | awk 'NR%4==0'); do
    git push -q origin "$c:yarin-sandbox" || { echo "FAIL at $c"; break; }
done
git push origin HEAD:yarin-sandbox
```

Then `sbatch chrono/sbatch/AGENT_runner.sbatch` again. Keep result files
small: heads of ~10 MB (`chrono/reports/ssl/heads/`) are the largest thing a
job commits; checkpoints stay under `chrono/artifacts_ssl/` (ignored).

## Files

| path | tracked | purpose |
|---|---|---|
| `runner.sh` | yes | the poll → run → push loop |
| `../sbatch/AGENT_runner.sbatch` | yes | keeps the loop alive as a 1-day job |
| `inbox/*.sh` | yes | commands waiting to run |
| `outbox/*.out` | yes | their output |
| `done/*.sh` | yes | commands that ran (audit trail) |
| `heartbeat` | yes | liveness |
| `STOP` | transient | its presence stops the runner |
