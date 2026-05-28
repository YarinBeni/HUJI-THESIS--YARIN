# Test 10 — Prompted reprobe (pv0/pv1/pv2/pv3)

**What it is:** does *prompting* the model first (context, framing, few-shot)
shift the linear probes' headline? For each prompt variant we re-extract
activations under that prompt and re-run the standard year-PLS / ruler-CLS
probes. Variants:

- **pv0** — headline "context-only" probe (the Round-3 default; no system
  prompt; last-token-inside-fragment pooling).
- **pv1 / pv2 / pv3** — control variants (system prompt swaps, few-shot
  injection, format perturbations). See
  `v_1/src/linear_probing/results/orcc_round2_phase1b/prompts/APPROVED.md` for
  the locked text and hashes.

**Coverage gaps (read carefully):**

- **qwen3_1b7** — pv0 complete (5 layers x 2 pools x {cls,pls} = 20 files);
  pv1 only L00/last/{cls,pls} = 2 files; pv2 / pv3 absent.
- **qwen3_8b** — pv0 only; even within pv0, `mean` pooling only at L00.
  Total 12 files. No pv1-3.
- **qwen3_32b** — pv0-pv3 fully complete (5 layers x 2 pools x {cls,pls} x 4
  variants = 80 files) plus a `phase1b_summary.json` side file.

The **cross-model comparison should focus on pv0** (the only variant present
for all three). Treat pv1-3 as **32B-only sensitivity** runs.

**CSV `T10_prompt_reprobe.csv`** — one row per (model, variant, pool, layer,
task). For PLS year: best-k Spearman from `metrics_per_k`. For CLS ruler:
Macro-F1 from the file's top level.
