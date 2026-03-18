# Schmidt Sciences Compute Cluster Guide

## Cluster Resources

| Resource | Specs |
|---|---|
| GPUs | 64x NVIDIA H100 80GB HBM3 |
| GPU Nodes | 8 servers, 8 GPUs each |
| CPUs | 832 Intel Xeon Platinum 8470 |
| RAM | 8 TB DDR5 total |
| Storage | 500 TB shared NFS |
| CUDA | 12.8 |
| Job Scheduler | Slurm |
| Partition Name | `voltagepark` |
| Max Job Time | 7 days |

## Access

- **Web UI**: https://schmidtsciences.parallel.works/
  - Browser-based terminal (no local install needed)
  - File manager for drag-and-drop uploads
  - Requires 2FA (use any TOTP app: Google Authenticator, Authy, etc.)
- **CLI (optional)**: `pw ssh schmidt` via Parallel Works CLI
  - Docs: https://parallelworks.com/docs/cli
  - Not required — web UI terminal is sufficient
- **Support**: compute@schmidtsciences.org

## Filesystem Layout

```
/home/yarin.b/                    # Your private home dir (persistent, on NFS)
├── miniconda3/                   # Conda installation (shared with compute nodes)
│   └── envs/thesis/              # Python 3.11 environment with ML packages
├── projects/                     # Your work goes here
│   └── lititure-review/          # Clone of the repo
├── pw/                           # Parallel Works platform (ignore)
│   ├── jobs/
│   ├── storage/
│   └── workflows/
└── .cache/huggingface/           # Cached model weights (auto-managed)

/data/                            # Shared data volume (101 TB, visible to all users)
├── software/                     # Shared software (CUDA, GCC, Python, etc.)
└── <other_users>/                # Other researchers' data
```

- `/home/yarin.b/` is **private** (only you can access)
- `/data/` is **shared** across all cluster users
- Both are **persistent** — files survive across sessions
- Both are on **NFS** — visible from web terminal AND compute nodes

## Architecture: Web Terminal vs Compute Nodes

```
┌──────────────────────────────────┐
│  Web Terminal (Browser)          │  You are here when you log in
│  - Manage files, git, submit     │
│  - NO GPUs, NO heavy compute     │
│  - Has access to /home/ and /data│
└──────────────┬───────────────────┘
               │ sbatch job.sh
               ▼
┌──────────────────────────────────┐
│  Compute Nodes (g0374–g0381)     │  Your jobs run here
│  - H100 GPUs, CUDA, full power   │
│  - Same /home/ and /data/ via NFS│
│  - Software via conda (not module)│
└──────────────────────────────────┘
```

Key rule: **Never run heavy compute on the web terminal.** Always use `sbatch` to send work to compute nodes.

## Environment Setup (Already Done)

Conda and the `thesis` environment are installed on the shared NFS at `~/miniconda3/`. This was done via a Slurm job (not on the web terminal) so the compute nodes can see it.

### Installed Packages (thesis env)
- Python 3.11
- PyTorch 2.10.0 (CUDA 12.8)
- Transformers 5.3.0
- Accelerate, Datasets, Pandas, NumPy, Scikit-learn

### To add packages in the future
Submit a quick Slurm job:
```bash
cat << 'EOF' > install_pkg.sh
#!/bin/bash
#SBATCH --job-name=install
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=install_%j.out

source ~/miniconda3/bin/activate thesis
pip install <package_name>
EOF

sbatch install_pkg.sh
```

Or install from the web terminal if it can see the conda env:
```bash
~/miniconda3/envs/thesis/bin/pip install <package_name>
```

## Daily Workflow

### On your Mac (VS Code)
1. Write/edit code locally
2. `git push`

### On the cluster (web terminal)
3. `cd ~/projects/lititure-review && git pull`
4. `sbatch v_1/src/cluster/<job_script>.sh`
5. `squeue -u $USER` to check status
6. `cat <output_file>.out` to read results

## Slurm Cheat Sheet

### Job Script Template
```bash
#!/bin/bash
#SBATCH --job-name=my_job         # name shown in squeue
#SBATCH --gres=gpu:1              # number of GPUs (1-8 per node)
#SBATCH --cpus-per-task=8         # CPU cores
#SBATCH --mem=64G                 # RAM
#SBATCH --time=04:00:00           # max runtime (HH:MM:SS), max 7 days
#SBATCH --output=logs/%j.out      # output file (%j = job ID)

source ~/miniconda3/bin/activate thesis
python your_script.py
```

### Commands
| Command | What it does |
|---|---|
| `sbatch job.sh` | Submit a job to the queue |
| `squeue -u $USER` | See your running/pending jobs |
| `scancel <job_id>` | Cancel a specific job |
| `scancel -u $USER` | Cancel ALL your jobs |
| `sinfo` | See node availability |

### Job States
| State | Meaning |
|---|---|
| `PD` (PENDING) | Waiting for resources |
| `R` (RUNNING) | Executing on a node |
| (disappears) | Completed or failed — check .out file |

### GPU Requests
```bash
#SBATCH --gres=gpu:1    # 1 GPU  — enough for 7-9B models
#SBATCH --gres=gpu:4    # 4 GPUs — for 70B models
#SBATCH --gres=gpu:8    # 8 GPUs — full node, for 405B models
```

## What We Verified Works

| Test | Result |
|---|---|
| Slurm job submission | Working |
| GPU access (H100 80GB) | Working |
| Conda env on compute nodes | Working |
| PyTorch + CUDA | Working (PyTorch 2.10, CUDA 12.8) |
| HuggingFace model download | Working (no auth needed for Qwen) |
| Qwen2.5-7B-Instruct inference | Working |
| Hidden state extraction (28 layers x 3584 dims) | Working |

## Model Capacity per GPU

| Model Size | GPUs Needed | Examples |
|---|---|---|
| 7-9B | 1 GPU | Qwen-7B, Llama-8B, Gemma-9B (Track B/C targets) |
| 27-32B | 1 GPU | Qwen-32B, Gemma-27B |
| 70-72B | 2-4 GPUs | Llama-70B, Qwen-72B |
| 405B | 8 GPUs | Llama-3.1-405B |

## Parallel Execution

You can submit multiple jobs simultaneously — each gets its own GPU(s):
```bash
sbatch run_qwen.sh      # Job 1 on GPU
sbatch run_llama.sh      # Job 2 on another GPU
sbatch run_gemma.sh      # Job 3 on another GPU
# All 3 run in parallel
```

With 64 GPUs available, you can easily run all your Track A/B experiments simultaneously.

## Notes and Gotchas

1. **Web terminal is a container** — software installed there (e.g., conda) is NOT visible to compute nodes. Always install via Slurm jobs or use the NFS-installed conda at `~/miniconda3/`.
2. **Model weights are cached** in `~/.cache/huggingface/` after first download. Subsequent runs load instantly.
3. **Gemma models require license acceptance** on HuggingFace + HF token login. Qwen and Llama do not.
4. **`module` command** is not available. Use conda/pip for environment management.
5. **No `nano`/`vim`** on the web terminal. Edit files locally and `git pull`, or use `cat << 'EOF' >` to create files directly.
6. **Output files** from jobs are written to the working directory where you ran `sbatch`. Create a `logs/` folder to keep things organized.
7. **Max job time** is 7 days (`7-00:00:00`). Always set `--time` to slightly more than you expect.
