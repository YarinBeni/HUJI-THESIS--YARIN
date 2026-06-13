# Round 4 — ChronoRank-SAE: master plan & agent operating manual

> **Read this file first.** It is the orientation for both you (Yarin) and any
> implementation agent. Each *pillar* below has its own self-contained brief in
> this folder (`PILLAR_*.md`). The intended workflow is:
>
> 1. You open an **Opus implementation agent** and tell it:
>    *"Read `round4/PILLAR_X_*.md` and implement it."*
> 2. The agent writes the code **and** the `sbatch` script, commits, and tells you
>    the exact command to paste into the cluster web terminal.
> 3. You run it on the cluster, copy the `.out` log (or the committed result files)
>    back, and paste them to the agent.
> 4. The agent reads the results, checks the pillar's success criterion, and either
>    iterates or hands off.
>
> The agent never SSHes. You are the only one who touches the cluster. Every pillar
> brief is written so the agent produces a **ready-to-paste sbatch command**.

---

## 0. The research question (why we are doing this)

This round is the synthesis of the three competing plans in
`yarin/research_plan/planning/thesis_plan.md`. Verdict from that comparison:

> **Plan 3 for the research question, Plan 1 for the system, Plan 2 only for tools.**

The thesis spine:

> Scale does **not** solve Akkadian dating. NTP finetuning does **not** help (Round 3
> negative result, confirmed under maximal-balanced PLS). A translation-trained
> cuneiform encoder (Thalesian, a uMT5-base finetune) wins. **Why?** And can a small,
> confound-resistant, *ordinal* dating head expose that signal as calibrated dates +
> retrieved parallels + interpretable evidence?

Two thrusts:

- **Thrust A — the Thalesian autopsy (THE FOCUS — do this first).** "Why does the small
  translation-trained model win?" Thalesian is the only real winner from the mean-balanced-
  maximal PLS experiment, so understanding *what* makes it good is the highest-value work:
  it is interesting on its own (what do LLMs encode about historical time / temporal
  world-models?) **and** it is actionable — if we can name the cause (tokenizer? bidirectional
  encoder architecture? seq2seq/translation objective? the act of cuneiform finetuning?), we
  can fold that lesson into a *better* finetune of the big models, pick a better frozen
  backbone, or train a purpose-built bidirectional/translation model. → **Pillar 1**.
- **Thrust B — build the honest ordinal model** ("expose the signal without cheating"):
  ChronoRank on frozen embeddings, then anti-shortcut training, then the ChronoAtlas
  retrieval interface. → **Pillars 0, 2, 3, 6**.

What we are explicitly **not** doing in Round 4 (from the plan comparison + Yarin's calls):
another blind NTP finetune, generative diffusion, NODE/NCDE, English-translation-as-main-input,
**unlabeled-corpus seriation (Pillar 4 — PARKED, see below)**, and **SAE interpretability
(Pillar 5 — DEFERRED to the end, contingent on Pillar 1, see below)**.

---

## 1. Pillars and the dependency graph

| Pillar | Name | Thrust | GPU? | Status / depends on |
|---|---|---|---|---|
| **P1** | **Thalesian autopsy** (tokenizer · architecture · objective · finetune) | A | Mixed | **DO FIRST — the focus.** Uses on-disk activations + P0 eval metrics |
| **P0** | Shared harness: labels, runtime text-transforms, ordinal-eval metrics | B (root) | No (CPU) | Root of Thrust B |
| **P2** | Minimal ChronoRank (pairwise rank + interval loss on frozen embeddings) | B | No (CPU) | P0 |
| **P3** | Anti-shortcut training (masking consistency · adversary · cross-genre pairs) | B | No (CPU) | P0, P2 |
| **P6** | ChronoAtlas: retrieval + evidence reports | B | No (CPU) | P2 (P3 better) |
| **P4** | ~~Graph smoothing / seriation over unlabeled corpus~~ | — | — | **PARKED** — unlabeled 2M words have no labels; ≈ what frozen-model already did (poorly). Revisit on royal inscriptions later. |
| **P5** | ~~SAE / sparse feature archaeology~~ | — | — | **DEFERRED to end** — contingent on P1 results (and a possible second, lesson-informed finetune). |

```
   P1 (Thalesian autopsy) ───────────────────────►  THE FOCUS, runs first & in parallel
        │  finding (tokenizer? arch? objective? finetune?)
        ▼
   (optional Phase-2 follow-up: a better, lesson-informed finetune of the big models)
        ┊
   ────────────────────────────────────────────────────────────────────
   Thrust B (honest ordinal system), uses on-disk frozen embeddings now:

   P0 (harness) ──► P2 (min ChronoRank) ──► P3 (anti-shortcut) ──► P6 (ChronoAtlas)

   P4 PARKED · P5 DEFERRED (only after P1 + possible 2nd finetune)
```

**Recommended launch order (what to give agents, when):**

1. **Now, the priority:** **P1** (the autopsy — its cheapest sub-experiment, the
   vanilla-uMT5 control probe, needs no training and only on-disk activations). In parallel,
   P0 (harness) so Thrust B is unblocked.
2. **After P0 lands:** P2.
3. **After P2 lands:** P3, then P6.
4. **P1's training arm** (objective ablation) needs a cluster run — start its data-prep early
   (and its English-parallel-data question — see P1).
5. **P5 (SAE)** only once P1 has told us *what* to look for, and ideally after any second
   finetune. **P4 stays parked.**

Use git worktrees for parallel pillars to avoid collisions
(`Agent(..., isolation: "worktree")` or `git worktree add`).

---

## 2. How the cluster works (give this to every agent)

**Schmidt Sciences HPC** — full reference: `v_1/src/cluster/README.md`. Essentials:

- **Scheduler:** Slurm. **Partition:** `voltagepark`. **GPUs:** H100 80GB. **Max time:** 7 days.
- **You log into a web terminal** (https://schmidtsciences.parallel.works/). It has
  **no GPU** — never run compute there. You only `git pull`, `sbatch`, `squeue`, `cat`.
- **Repo path on cluster:** `~/projects/HUJI-THESIS--YARIN` (this is what the existing
  finetune sbatch scripts `cd` into — note the `cluster/README.md` still says the old
  `lititure-review` name; the sbatch convention is authoritative).
- **Env activation (canonical, from the FT sbatch scripts):**
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate thesis
  ```
  `thesis` = Python 3.11, PyTorch 2.10 (CUDA 12.8), Transformers 5.3, sklearn, pandas.
  To add a package: `~/miniconda3/envs/thesis/bin/pip install <pkg>` from a small Slurm
  job (see cluster README), **not** the web terminal.
- **Job logs** land where you `sbatch` from; pillars write to `v_1/src/<area>/logs/%j.out`.

**The standard sbatch preamble every pillar uses** (agents must emit this verbatim):

```bash
#!/bin/bash
#SBATCH --job-name=<short>
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1            # omit this line for CPU-only pillars (P0/P2/P3/P6)
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/<area>/logs/%j.out
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# ... pillar work ...
# commit small result artifacts + push at the end (see FT3 sbatch for the pattern)
```

**The agent↔cluster↔Yarin loop, concretely:**

- Agent finishes code → commits + pushes to `main` → prints you a single block:
  *"Paste this into the cluster terminal: `cd ~/projects/HUJI-THESIS--YARIN && git pull && sbatch <path>.sbatch`"*.
- You run it, then `squeue -u $USER` to watch, then when done paste back either the
  tail of the `.out` log or `git pull` the committed result JSON and tell the agent the path.
- CPU-only pillars (P0, P2, P3, P6) are small enough that the agent can also run a
  **tiny local sanity check** before handing you the full job — encourage that.

---

## 3. How the data & results are organized today (give this to every agent)

All paths relative to repo root. **This is the ground truth — do not invent paths.**

### Corpora (inputs)
| What | Path | Rows | Key columns |
|---|---|---|---|
| ORCC royal inscriptions (the dated set) | `v_1/data/evaluation/corpora/orcc_corpus.parquet` | 1,202 | `fragment_id, ruler, period, year, genre, provenance, word_count, text, text_tier0, text_maximal, text_tier0_masked, text_maximal_masked` |
| Letters (period-labeled eval set) | `v_1/data/evaluation/corpora/texts_for_evaluation.parquet` | 4,957 | `text/full_text, period, fragment_id` |
| Unified corpus (the unlabeled 2M-word pool) | `v_1/data/unified/unified_corpus.parquet` | 2.45M **word-rows** | `source, fragment_id, line_num, word_idx, value_clean, value_signs, lemma, language, ...` |
| NTP finetune splits | `v_1/data/finetune/ntp_{train,val}.parquet` + `metadata.json` | — | leakage-tracked vs ORCC probe |

**Critical labeling facts (P0/P2 must respect these):**
- `year` is a single integer per fragment = **years BCE** (larger = older). 41 distinct
  rulers; counts are very uneven (Ashurbanipal 268, Sennacherib 237, Esarhaddon 176;
  long tail with n=1). Esarhaddon is the only ruler spanning a real range (669–681);
  most rulers map to a single `year` value → **ruler reign = the natural interval**.
- **ORCC is ~entirely one genre** (`Royal Inscription`, 1193/1202). So *leave-one-genre-out*
  cannot be done inside ORCC — cross-genre means ORCC(royal) vs letters vs SEAL, and
  letters carry only coarse OB/NA/LB period labels, not year. Be honest about this in
  P3; cross-genre transfer is coarse, not a clean year-level test.

### Frozen activations (already extracted — **do not re-extract these**)
- Base dir: `v_1/src/linear_probing/results/orcc__embed/activations/`
- Layout: one subdir per `<method>_<cleaning>_<pool>/`, containing per-layer arrays.
- **Already on disk** for ORCC: `thalesian_akk300m`, `qwen3_1b7`, `qwen3_8b`, `qwen3_32b`,
  `gpt_oss_120b`, `random` — each at `cleaning ∈ {tier0, maximal}` × `pool ∈ {mean, last}`,
  all layers. Plus finetune checkpoints `qwen3_*_ft{00,09,19,25}_*`.
- **Canonical loader — use it, do not hand-roll `.npy` paths:**
  `v_1/src/geodesic/utils.py` → `find_acts_dir(method, cleaning, pool)`, `load_layer(dir, L)`,
  `available_layers(dir)`, `load_year_labels(parquet_path)`.

### The honest evaluation harness (the maximal-balanced protocol)
- **Balanced-subset draws:** `v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/`
  → `draws_matrix.npy` (Monte-Carlo balanced ruler draws), `corpus_fragment_order.json`.
- **MC probe runner:** `v_1/src/linear_probing/round2_phase0/run_mc_probes.py` — it
  **auto-registers any method whose activations exist on disk** (`_register_dynamic_probes`)
  and emits `*__summary.json`. This is the existing PLS/CLS scoreboard path.
- **Geodesic toolkit (graphs, Isomap, pairwise-order accuracy):** `v_1/src/geodesic/utils.py`
  — `build_knn_graph`, `geodesic_dist`, `isomap_1d`, `pairwise_order_acc`, `pls_pairwise_acc`.
- **PLS baseline machinery (GroupKFold-by-ruler, metrics):** `v_1/src/linear_probing/pls_utils.py`
  → `fit_pls_groupkfold`, `compute_metrics` (r2, spearman, mae, mase, mdape), `l2_normalize`.
- **Cleaning + splits constants:** `v_1/src/linear_probing/utils.py` (SEED=42, 70/15/15,
  PERIOD_MAP, the 11-filter maximal cleaning).

### Where Round 4 outputs go (new tree — agents create these)
```
v_1/src/chronorank/
├── labels.py            # P0: interval/pair label builder
├── transforms.py        # P0: RUNTIME text transforms (mask/formula/crop) — see principle below
├── eval_ordinal.py      # P0: ordinal-eval metrics (pairwise acc, coverage, calibration, NLL)
├── model.py             # P2: ChronoHead (linear / sparse / 1-MLP) + losses
├── train.py             # P2/P3: training loop, staged losses
├── atlas.py             # P6: retrieval + evidence report builder
├── autopsy/             # P1: tokenization audit, control-ladder probing, objective ablation
├── sbatch/              # all *.sbatch live here
├── results/             # *.json summaries, figures (committed); large dumps gitignored
└── README.md            # running log, updated by each pillar
# P4 (graph.py) PARKED · P5 extends v_1/src/sae/ only once DEFERRED work begins
```

### Engineering principle: text transforms happen at RUNTIME, not on disk (Yarin's call)
The augmentation/masking views (name-masked, formula-removed, ≤N-word crop, normalized) must be
applied **on the fly inside the data loader** — exactly like a PyTorch `Dataset.__getitem__`
reads an image then applies a transform before returning it — **not** pre-computed and stored as
extra parquet columns or in-memory arrays. Each text is loaded once; the transform is a callable
applied per-batch. This keeps memory flat (we never hold N copies of the corpus) and lets us add
new views without re-materializing data. So P0 ships `transforms.py` as **composable callables**
(`Compose([NameMask(), Crop(32)])`), and P2/P3's training loop applies them in `__getitem__`.
(The frozen *embeddings* are still pre-extracted and cached — only the *text→view* step is
runtime. For frozen-embedding pillars this means: if a view changes the text, its embedding must
be computed at load time too; for the cheap probing experiments we use the already-extracted
clean/maximal/masked embeddings on disk, and reserve runtime transforms for any pillar that
trains through the text.)

**Existing finetune scoreboard reference (the bar Round 4 must beat/match honestly):**
Thalesian maximal-balanced PLS **Spearman ≈ 0.41 (best layer ~L11)**; Qwen3-8B ≈ 0.36;
TF-IDF ≈ 0.29 (and collapses to 0.245 when truncated — pure length crutch); MLM ≈ 0.31
(≈ random). Success is **not** "beat 0.41" — see §4.

---

## 4. What "success" means this round (give to every pillar)

From the plan comparison, the headline contribution is **honest chronological modeling**,
not leaderboard chasing. A model with Spearman ≈ 0.43 that survives name/formula masking,
gives calibrated intervals, and returns historian-usable parallels **beats** a 0.48 model
that collapses under shortcut removal. Each pillar brief states its own concrete PASS
criterion; they all roll up to this.

---

## 5. Papers to read (per thrust — full list lives in each pillar)

**Thrust A (autopsy):** Thalesian model card (`Thalesian/cuneiformBase-400m`, a uMT5-base
finetune); Akkadian NMT / Babylonian Engine (Akkademia, BiLSTM, BLEU≈37); uMT5 paper.

**Thrust B (system):** CORAL/CORN rank-consistent ordinal regression; Rank-N-Contrast (RnC);
SimCSE (augmentation consistency); domain-adversarial training / Gradient Reversal (DANN);
TALM & TicTac (diachronic text dating); Ithaca & Aeneas (retrieval-augmented historian tools).
*Parked/deferred:* CLSS + Snorkel (P4, parked); Anthropic dictionary-learning / SAE (P5, deferred).

Each pillar lists the 2–4 papers that are load-bearing **for that pillar** so an agent
isn't told to read everything.
