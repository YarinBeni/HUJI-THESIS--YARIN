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
