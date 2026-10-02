# Handoff: how we run experiments on Yarin's Slurm cluster

You are an AI assistant working in a repository that needs to run experiments on
Yarin's GPU cluster. This document is the complete operating manual, distilled
from a month of running ~200 jobs on it from a sibling repo. Follow it and you
inherit all the lessons without repeating the failures.

## 1. The physical setup

- The cluster is reachable ONLY through a browser terminal (parallel.works).
  Yarin pastes one-liners there and sends you screenshots of the output. He will
  not copy long scripts by hand — everything else must travel through git.
- Scheduler: **Slurm**. Partition: **`voltagepark`**. GPU nodes we saw:
  g0375–g0381 (one GPU per job is the norm; ~15–20 concurrent GPU jobs fit).
- Repo checkout lives at `~/projects/<REPO-NAME>` on the cluster.
- Python: **miniconda at `~/miniconda3`**, environment name **`thesis`**
  (torch + transformers + sklearn + pandas/pyarrow installed; `tabulate` is
  NOT — hand-roll markdown tables). Every job starts with:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate thesis
  ```
  Interactive shells do NOT have the env active: use `python3` (not `python`)
  in any script a human might run by hand, and fail loudly if a step fails —
  an empty output from a missing interpreter once read as "nothing to do".

## 2. Job shapes that work (copy these)

| kind | header |
|---|---|
| GPU training | `--partition=voltagepark --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00` |
| big CPU (probes/aggregation) | `--partition=voltagepark --cpus-per-task=16 --mem=96G --time=06:00:00` |
| tiny daemon | `--partition=voltagepark --cpus-per-task=2 --mem=4G --time=1-00:00:00` |

Arrays: `#SBATCH --array=0-N`, map `SLURM_ARRAY_TASK_ID` to a bash array of
cells inside the script. **Cluster quirk:** every array TASK gets its own job
id; in `squeue`, `%A` prints per-task ids — use `%F` for the array's base id.
Chain stages with `--dependency=afterany:<jobid>`. A dependency can become
`DependencyNeverSatisfied` and sit forever — check for it, cancel, resubmit.

**Slurm snapshots the script at submit time.** Editing a .sbatch never changes
already-queued jobs; code the job `git pull`s at start is the only live part.

## 3. The git-driven runner (how jobs get submitted without a human)

Nobody will paste your sbatch commands. Instead a tiny 1-day Slurm job — "the
runner" — polls the work branch every 60 s and turns it into a command channel:

```
you (outside)              GitHub branch                cluster runner
push inbox/NNN_name.sh ──►  agent/inbox/   ──pull──►  bash -e NNN_name.sh (timeout)
read outbox/           ◄──  agent/outbox/  ◄──push──  stdout+stderr → NNN_name.out
                            agent/done/    ◄──push──  the script, moved here
                            agent/heartbeat ◄─push──  "alive…" hourly when idle
```

Mechanics to replicate (copy `chrono/agent/runner.sh` +
`chrono/sbatch/AGENT_runner.sbatch` from the `yarin-sandbox` branch of
`YarinBeni/HUJI-THESIS--YARIN` — they are battle-tested):
- Scripts run with `bash -e`, repo root, conda env active, default timeout
  30 min (`# TIMEOUT=7200` line overrides). Sorted order, one at a time; a
  name already in `done/` never re-runs — new work needs a new number.
- Inbox scripts usually just call `sbatch` for the real jobs and echo the ids.
  Make them **idempotent** — the runner may start late and a human may have run
  the script by hand meanwhile. Guard resubmissions (check `squeue` for a live
  array before submitting a duplicate).
- The runner re-execs itself when runner.sh changes on the branch; a pushed
  empty `agent/STOP` file makes it exit and commit "agent: stopped".

**Rules agreed with Yarin (keep them):** the runner lives **one day max**
(`--time=1-00:00:00`), is started per session (`sbatch .../AGENT_runner.sbatch`
— the ONE command Yarin runs), must be stopped at session end (push STOP or
`scancel`), and YOU must remind him. Anything pushed to inbox/ runs as the
cluster user — never widen push access to the repo while a runner is alive.

## 4. Git from inside jobs (the part that bites)

Every job syncs the branch at start and pushes its results at exit. Copy
`chrono/sbatch/_sandbox.sh` — it encodes all of this:
- `sync_sandbox`: fetch + `git rebase --autostash FETCH_HEAD`, retried 5×,
  under an `flock` file lock (array tasks share one working tree — unlocked
  concurrent git corrupts the index / races HEAD.lock). It must FAIL the job
  if it can't sync — running stale code silently is worse. Never use
  `checkout -B origin/...` (it silently discards unpushed local commits).
- `commit_push_sandbox "msg" paths...`: add/commit/fetch/rebase/push under the
  same lock, retried, returns nonzero if the push never landed.
- **The 20 MB guard (hard-earned):** before committing, unstage any file over
  20 MB. Twice, a job copied a 100–400 MB model checkpoint into a
  results dir; every later commit swept it in, `git pack-objects` was
  **OOM-killed on the login node**, and NO push from the cluster succeeded for
  8 hours while results piled up as local commits. Checkpoints live in a
  **gitignored** artifacts dir; only small text/parquet results are committed.
- `enable_log_push`: an EXIT trap that commits the job's own Slurm log (last
  4000 lines) to `reports/logs/` on success AND failure — otherwise the only
  logs you can read are from jobs that didn't need reading.
- Auth: the repo being public means `fetch` works anonymously, but `push`
  needs Yarin's token, which HE sets himself on the cluster
  (`git remote set-url origin https://<TOKEN>@github.com/...`). **Never ask
  for or accept the token in chat.** Symptom of an expired token: runner
  alive, zero commits arriving. Symptom of the OOM problem: `send-pack:
  unexpected disconnect` / `pack-objects died of signal 9`.
- Recovery when pushes are stuck behind a fat commit: `git reset --soft` to
  the remote tip, drop files >20 MB from the index, one squash commit, push.

## 5. Loading models from Hugging Face on the cluster

Compute nodes have internet; models download to the default HF cache in the
user's home on first use and are reused after. What we learned:

```python
import os, torch
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    model = AutoModelForCausalLM.from_pretrained(
        hfid, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True, attn_implementation="sdpa")
except Exception:
    model = AutoModelForCausalLM.from_pretrained(      # some stacks lack sdpa
        hfid, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True)
core = getattr(model, "model", model)                  # decoder sans LM head
tok = AutoTokenizer.from_pretrained(hfid)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
```

- **bf16 + `device_map="auto"`** — a 7B/8B model fits one GPU this way.
- **Gated repos:** `meta-llama/*` fails with "Cannot access gated repo" unless
  the HF account was granted access. Use the ungated **NousResearch** mirrors
  (`NousResearch/Llama-2-7b-hf`, identical weights) as the primary or fallback.
  Wrap `snapshot_download` in try/except with a `fallback_hfid`.
- **transformers ≥ 5 cannot load Llama-2's SentencePiece `tokenizer.model`**
  (tiktoken converter crashes on ANY path). Load the tokenizer from
  `hf-internal-testing/llama-tokenizer` (prebuilt tokenizer.json, same 32k
  vocab, ids match the weights) and the weights from the model repo.
- Tokenizer loading in general: try `use_fast=True` then `False`, and print
  every failure — silent tokenizer fallbacks poison results.
- Extraction conventions from the thesis: `output_hidden_states=True` gives
  n_layers+1 states (index 0 = embeddings); for causal models prepend BOS but
  **exclude BOS from mean-pooling**; ask the loaded model how many layers it
  has rather than hardcoding (a "layer 11" default once crashed a 9-layer
  model's run).
- Random-weights control (if you need one):
  `AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(hfid))` with a
  fixed `torch.manual_seed`.
- A full forward pass over ~30k short texts through a frozen 7B takes **over
  an hour** on one GPU. Budget for it (next section).

## 6. Long-run discipline (each rule = a real failure)

1. **Wall-time ≠ work budget.** A "train 5 h" job inside a 6:30 wall limit was
   killed during its final save/embedding phase — 6×5 GPU-hours produced
   nothing. Keep the work budget a parameter (`HOURS=${HOURS:-5}`), checkpoint
   every few minutes, make every run **resumable from ckpt**, and leave ≥1.5 h
   of wall headroom for the output phase.
2. **Smoke before burn.** Every new training script first runs ~60 steps
   (`--steps 60`) inside the same job and aborts if that fails.
3. **Filename hygiene.** Run ids with `::` or `/` crashed every output write
   AFTER hours of compute. Sanitize: `re.sub(r"[^0-9A-Za-z_.-]+", "_", name)`.
4. **Atomic shared files.** Anything several array tasks write (a manifest, a
   results parquet) needs flock + write-to-temp + `os.replace`, and should be
   rebuildable from its parts.
5. **Self-describing outputs.** Put ids/config inside every artifact so a
   corrupted index can be rebuilt.
6. **Slurm logs** go to a repo-path like `sbatch/logs/` (gitignored) and reach
   the branch only via the log-push trap.

## 7. Working with Yarin

- He pastes ONE-liners and sends screenshots; design every human step as a
  single copyable line that prints its own diagnosis.
- Useful one-liners to give him:
  `squeue -u $USER | head -25` · `sacct -X -S <HH:MM> --format=JobID,JobName%14,State,Elapsed | tail -15` ·
  `tail -5 <runner log>` · `sbatch <file>` · `scancel <id>`.
- Status reports: short, in simple Hebrew, with a shrinking task list; lead
  with what changed; honest negatives stated plainly.
- Keep a living status board file (what ran, what it found, what's queued) and
  a dated results .md per experiment, committed with the results — the repo is
  the lab notebook. Document every incident (what broke, root cause, fix).
