# Stress-tests — full briefing (for an advisor / fresh-context agent)

This document explains **every experiment** under `v_1/src/stress_tests/`: the paper it
mirrors, what it tests, the exact configuration, which models ran, **where the results
live**, the headline numbers, what we concluded, and the known caveats. It is meant to
be read cold — you should be able to understand and defend the whole suite from this file.

---

## 0. The thesis question
We stress-test the "LLMs learn a world-model / linear timeline" literature
(Gurnee–Tegmark; Godey geography; "A Matter of Time"; k-sparse "Finding Neurons in a
Haystack") on a **hard, low-resource, no-web-leakage** target: **dating Akkadian royal
inscriptions** (ORCC), i.e. recover the **year BCE** from a transliterated cuneiform text.

**Bottom line (what all the experiments together show):** the models *know* the dates
behaviorally (T9) and the probing pipeline is valid (P2 geography decodes), **but the
date is not encoded as recoverable structure over text.** Every probe that looks
positive is explained by **token identity** (mostly the king's name), and a
**random-initialized network matches or beats the trained ones** on the decisive sites.
The only genuine flicker of *learned* structure is the cuneiform-domain **Thalesian**
model, and only modestly.

---

## 1. Data & shared setup
- **Corpus:** `v_1/data/evaluation/corpora/orcc_corpus.parquet` — 1,202 fragments, 41
  ruler labels, 6 periods, 74 provenances. Built by
  `v_1/src/corpus/03_build_orcc_corpus.py` from
  `v_1/data/raw/chungrong/orcc_round1/royal_inscriptions.csv`.
- **The `year` label = `min(digits of sub_period)`** (e.g. "ca. 668–631" → 631), i.e.
  the **lower bound of the reign/sub-period**, with reign-end fallbacks for 4 rulers
  (Neb II 562, Nabonidus 539, Sargon II 705, Tiglath-pileser III 727). **Crucially `year`
  is one constant per ruler** (nunique=1 for all but Esarhaddon), so *decoding year ≈
  identifying the ruler.* See `results/eda/`.
- **Cleanings:** `tier0` (minimal markup strip — royal names, which are logographic like
  `m-MAN-GIN`, stay intact) and `maximal` (11 filters incl. remove-logograms/lowercase —
  destroys royal names). A third, **`maximal_keepking`** ("maximal-with-kings"), was added
  by us: full maximal context but the king-name span is frozen intact
  (`shared/cleaning.py`).
- **Pooling sites:** `mean` (whole-text masked mean), `king_last` (last token of the
  commissioning ruler's name span), `king_mean` (mean over the name span). king_* need the
  logographic name → tier0 or maximal_keepking only.
- **Balanced Monte-Carlo (MC):** to control ruler imbalance, `build_balanced_subset.py`
  draws 200 balanced sets of 8 rulers × 21 fragments (the 8 with ≥21 frags). The
  maxking variant uses 5 rulers × 9 king-found fragments. Draws live in
  `v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset{,_maxking}/`.
- **Probes:** PLS regression (n_components "k" swept over {1,2,3,5}, best-k reported) and a
  Ridge arm; Spearman(pred, year) is the headline metric; shuffled-label gives the null.
- **The two baselines for "is this learned?":** (i) a **shuffled-label null** (≈0 for
  regression, ≈chance for classification) and (ii) a **random-initialized model**
  (`random` = Qwen3-8B `from_config`, seed 42). The random *model* is the more
  informative control and is the one that matters below.
- **Models:** Qwen3-1.7B/8B/32B, gpt-oss-120B, Thalesian-AKK-300m & Thalesian-cunei-400m
  (cuneiform-domain), uMT5-base (encoder), an Aeneas **MLM**, a **TF-IDF** bag-of-signs
  baseline, and the **random** control. Not every model ran in every test (see each
  section + the config matrix in §11).

---

## 2. T9 — behavioral date knowledge  *(control: do the models even know?)*
- **Code:** `redo_t9_knowledge/` (uses `linear_probing/round2_phase1a/{run,parse,score}_kp.py`).
- **Results:** `redo_t9_knowledge/direct_kp_<model>/scores/kp{0,1,2}_metrics.json`;
  CSV `results/csv/t9_knowledge.csv`.
- **Three distinct probes (free-text generation, not activation-probing):**
  - **kp0 — date accuracy:** "when did ruler X reign?" → parse (start,end); CORRECT if a
    true reign year is within **±50 yr** of the predicted window.
  - **kp1 — period recall:** "list the rulers of period P" → recall of the target rulers,
    matched **case- & diacritic-insensitively** (`NFKD` strip + lowercase) but by
    **exact normalized string** (so `Nebuchadnezzar` w/o `II`, or a spelling variant, is
    a MISS — see caveat).
  - **kp2 — hallucination gate:** feed **fake** ruler names; giving any date = hallucination;
    gate passes if rate < 0.30.
- **Numbers:** kp0 acc — 1.7B 0.88, 8B 0.88, 32B 0.75, gpt-oss 1.00. kp2 — all pass except
  1.7B (0.75, fails). kp1 recall low/variable (0.00–0.62).
- **What we saw:** models **do know** the dates (kp0 high) and mostly don't hallucinate
  (kp2). → the probing nulls below are *not* "the info is absent."
- **Caveat:** kp1's exact-normalized match likely **undercounts** recall (naming variants);
  gpt-oss's kp1=0.00 is suspicious — inspect its raw list format. Models: 4 causal chat
  models only (generation task; encoders/MLM/TF-IDF/random can't "answer").

---

## 3. P2 — Godey geography  *(POSITIVE CONTROL — this one should, and does, pass)*
- **Code:** `p2_godey_geography/probe_p2.py`. **Results:**
  `p2_godey_geography/results/p2_geography__<model>.json`; CSV `results/csv/p2_geography.csv`.
- **Tests:** can a probe decode **where** a text is from (find-spot lat/lon) from the
  **mean** activation? If yes, the pipeline is valid and toponyms-in-text are readable.
- **Config:** cleaning **tier0 + maximal**; **GroupKFold-by-site** (imbalanced, not MC);
  **PLS(best-k) + Ridge**; **mean pool only**. lat and lon are predicted by two separate
  PLS fits. `gc_km` = great-circle error; **`skill_vs_centroid` = 1 − err/centroid_err**
  (centroid = predict the mean training location).
- **Numbers (tier0):** 172–204 km error, **skill +0.23–0.35**; best = Thalesian-cunei
  (+0.354, 172 km); **random +0.232**.
- **What we saw:** geography decodes well → **pipeline valid**. Training adds only
  ~+0.06–0.12 skill over random (find-spot has strong lexical cues). Thalesian-cunei best.
- **Coverage:** 8 models incl. both Thalesian + random. MLM/TF-IDF absent (no mean geo acts).

---

## 4. P1 — Gurnee–Tegmark year probe  *(THE core question, 3 protocols)*
Recover **year** from frozen activations. Contrast: recoverable at `king_*` (local, where
the name is) but not at `mean` (global) ⇒ "local-but-not-global"; matched by `random` ⇒
token identity, not learned.

### 4a. Single GroupKFold — `p1_gurnee_tegmark/probe_p1.py`
- **Results:** `results/p1_year__<model>.json`; CSV `results/csv/p1_year_gkf.csv`.
- **Config:** mean (tier0+maximal) + king (tier0); GroupKFold-by-ruler, all frags
  (imbalanced); PLS(best-k)+Ridge **+ a 1-hidden-layer MLP linearity check**; fold-std +
  shuffle null.
- **Numbers (best-layer Spearman):** mean_tier0 0.41–0.51 (random 0.413); king_last
  0.30–0.44. *(random king cells blank — those acts were extracted later; see 4b.)*

### 4b. Balanced-MC — `p1_gurnee_tegmark/probe_p1_mc.py`  *(thesis headline)*
- **Results:** `results/mc/p1_year_mc__<model>.json`; CSV `results/csv/p1_year_mc.csv`;
  formatted table `results/RESULTS_stress_tests.md`.
- **Config:** 200 balanced draws (8×21), GroupKFold-by-ruler within each; PLS(best-k)+Ridge;
  **±std over 200 draws** + shuffle null. Includes **MLM** and **TF-IDF (cited)**.
- **Numbers (PLS Spearman):** mean_tier0 ≈ 0.35–0.40 across all incl. **random 0.351** and
  TF-IDF 0.407; **king_last 0.48–0.70 — but random 0.643** (and MLM 0.704).
- **What we saw (decisive):** whole-text `mean` ≈ random ≈ TF-IDF, **flat across scale &
  objective** → no learned text-level chronology. **king_last high but random matches it**
  → **name-token identity, not a learned date.**

### 4c. maximal-with-kings — `p1_gurnee_tegmark/probe_maxking.py`  *(the fair rematch)*
- **Why:** old setup compared `mean`(maximal) vs `king_*`(tier0) — not apples-to-apples.
  This puts all 3 sites on **one** cleaning (`maximal_keepking`) and rebalances (5 rulers ×
  k=9, drawn from king-found only) so the random baseline is a genuine control.
- **Results:** `results/maxking/p1_maxking__<model>.json` + `RESULTS_maxking.md`; CSV
  `results/csv/p1_maxking.csv`.
- **Metrics:** **ruler_clf macro-F1** (StratifiedKFold — the control), **year_strat**
  (StratifiedKFold Spearman + **±10-yr accuracy**), and **year_group** (legacy
  GroupKFold-by-ruler — degenerate for a per-king-constant label, near 0/negative by
  construction; kept only for continuity). Best layer by ruler-F1; std over draws;
  chance=0.20, shuffle≈0.08.
- **Numbers (ruler macro-F1, chance 0.20):** mean — trained 0.66–0.90, **random 0.741**;
  king_last/king_mean ≈ 0.94–0.99 **for everyone incl. random**.
- **What we saw:** **random ≥ every trained model on `mean`**, and ≈ everyone on the king
  sites → even whole-text recoverability is token identity, not learned. **Only
  Thalesian-cunei beats random on mean (0.897)** — the one real positive.
- **Not run:** gpt-oss (OOM), MLM (needs its own extractor), TF-IDF (no per-token reps).

---

## 5. P3 — "A Matter of Time" timeline geometry
- **Code:** `p3_matter_of_time/{extract_anchor_acts,timeline_p3}.py`, prompts in
  `p3_matter_of_time/anchors/`. **Results:** `results/p3_timeline__<model>.json`; CSV
  `results/csv/p3_timeline.csv`.
- **Tests two things per layer:**
  - **3a — do the date-anchor prompts form an ordered line?** Fit a 1-D embedding of the
    153 **anchor-prompt** activations (PCA-1D linear + Isomap-1D nonlinear/cosine),
    Spearman(1-D coord, anchor year) (abs — 1-D sign is arbitrary).
  - **3b — do real ORCC texts land on that line?** For each text's **tier0 mean**
    embedding, nearest anchor by cosine → its year = prediction; Spearman(pred, true).
- **Config:** tier0, mean pool for texts; **unsupervised** PCA/Isomap + nearest-neighbor
  (no PLS/MC). No std; baseline = the random model.
- **Numbers:** **3a 0.21–0.57** (grows with scale; gpt-oss 0.567; **random 0.342**); **3b
  0.03–0.13** everywhere.
- **What we saw:** the **dissociation** — the model can arrange *explicit* date anchors into
  a rough line, but *real inscription texts do not project onto it* (3b ≈ 0). And random's
  3a ≈ the small trained models → even the anchor-line is largely not learned.
- **Coverage:** 8 models (needs anchor embeddings from J5); MLM/TF-IDF absent.

---

## 6. P7 — k-sparse localization ("Finding Neurons in a Haystack")
- **Code:** `p7_ksparse/probe_p7.py`. **Results:** `results/p7_ksparse__<model>.json`;
  CSV `results/csv/p7_ksparse.csv`.
- **Tests:** is the date in a few neurons? Binary **before/after median-year** task; select
  **top-k NEURONS** (k ∈ {1,2,4,8,16,32,64} — this "k" is a neuron count, *different* from
  PLS n_components) by ANOVA F-score, fit logistic, GroupKFold-by-ruler.
- **Config:** tier0, mean pool, no std saved (fold-mean); baseline = random model.
- **Numbers:** **best macro-F1 0.67–0.72** vs `chance_acc`=0.58 (majority-class **accuracy**;
  a trivial predictor's macro-F1 ≈ 0.37). **random model = 0.667.**
- **What we saw:** barely above trivial and **random ≈ trained** → the date is weak and
  distributed, not cleanly localized; training barely helps.
- **Caveat:** `chance_acc` is an accuracy floor, not an F1 floor — compare the macro-F1
  column to the **random model row**, not to 0.58.

---

## 7. T10 — does prompting change recoverability?
- **Code:** `redo_t10_prompt/{extract_prompted_king_acts,reprobe_king_pv,reprobe_king_mc}.py`;
  prompts `linear_probing/results/orcc_round2_phase1b/prompts/pv*.md`.
- **Results:** `redo_t10_prompt/results/<model>__t10_{king,mc}_summary.json`; CSVs
  `results/csv/t10_{gkf,mc}.csv`.
- **Prompt variants (activations pooled over the fragment span inside the prompt):**
  **pv0 = bare/zero-shot** (no system), **pv1 = framed zero-shot** (expert-Assyriologist
  system prompt), **pv2 = few-shot k=5**, **pv3 = chain-of-thought**.
- **Config:** tier0 text in prompt; span-mean + king_last/king_mean; **GKF and balanced-MC
  (8-ruler draws)**; PLS(best-k)+Ridge. **Causal only:** qwen3 1.7/8/32B (GKF); MC done for
  1.7B & 8B (32B MC file absent — task didn't finish). gpt-oss OOM'd; no random/MLM/TF-IDF.
- **Numbers (MC mean):** **identical ≈0.388–0.390 across pv0–pv3**; king_last 0.42–0.52.
- **What we saw:** **prompting (framing / few-shot / CoT) does not make the date more
  linearly recoverable** from the activations.

---

## 8. EDA / data diagnostics
- **Code:** `eda/class_imbalance_analysis.py`, `eda/plot_fragments_by_ruler.py`.
- **Outputs:** `results/eda/class_imbalance.md`, `fig_ruler_counts.png`,
  `fig_king_coverage.png`, `fig_effective_per_draw.png`, `fig_year_by_ruler.png`,
  `fig_year_tolerance.png`, `fig_fragments_by_ruler_all.png`.
- **Key facts:** `year` is one constant per ruler (⇒ year-probe ≡ ruler-ID); adjacent king
  labels are 12–38 yr apart (⇒ **±50 tolerance too coarse, ±10 used** in maxking);
  king-name coverage is uneven (Sennacherib 0.67 … **Neb II 0.00**), which is why the
  balanced-MC king pool shrinks and Spearman is high-variance.

---

## 9. Where results live (quick index)
```
results/
  RESULTS_stress_tests.md            P1 balanced-MC + P2 geography (formatted)
  RESULTS_stress_tests_explained.md  THIS FILE
  csv/                               one CSV per experiment (machine-readable)
  eda/                               class-imbalance report + all figures
p1_gurnee_tegmark/results/           p1_year__* (GKF), mc/p1_year_mc__* (MC),
                                     maxking/p1_maxking__* + RESULTS_maxking.md
p2_godey_geography/results/          p2_geography__*
p3_matter_of_time/results/           p3_timeline__*
p7_ksparse/results/                  p7_ksparse__*
redo_t9_knowledge/direct_kp_*/scores kp{0,1,2}_metrics.json
redo_t10_prompt/results/             *__t10_{king,mc}_summary.json
```
Activations (`*.npz`) and job logs are gitignored / cluster-only. Only result JSONs/CSVs/MD
and the balanced-subset draws are committed.

---

## 10. Infrastructure (how it runs)
- Cluster: Slurm (`voltagepark`), `conda activate thesis`, repo works on `main`. Jobs
  `sbatch/J*.sbatch`; **all share one NFS working copy**, so a hardened git helper
  `sbatch/_common.sh` (`sync_main`/`push_main`/`commit_push`: flock + rebase-autostash +
  retry) serializes commits. The agent env blocks direct `main` pushes → code lands via a
  feature branch fast-forwarded onto `main` on the login node.
- Extraction is GPU (per-model); probing is CPU. gpt-oss-120B repeatedly **OOMs** on gpu:4
  — it's excluded from maxking/T10 and the ladder stands without it.

---

## 11. Config matrix (one row per experiment)

| Exp | Paper | cleaning | CV / sampling | probe | pool | models | random | std |
|---|---|---|---|---|---|---|---|---|
| T9 | behavioral | raw prompt | n/a (generation) | tolerance / exact-match scoring | — | qwen 1.7/8/32B, gpt-oss | no | n/a |
| P2 | Godey geo (control) | tier0+maximal | GKF-by-site | PLS(k)+Ridge | mean | 8 (incl. both Thalesian) | yes | fold + skill |
| P1a | Gurnee–Tegmark | tier0+maximal / tier0 | GKF-by-ruler (imbal.) | PLS+Ridge+MLP | mean, king_last, king_mean | 8 | mean only | fold + shuffle |
| P1b | Gurnee–Tegmark | tier0+maximal / tier0 | balanced-MC 8×21 | PLS(k)+Ridge | mean, king_last, king_mean | 8 + MLM + TF-IDF | yes (incl. king) | ±std/200 + shuffle |
| P1c | maxking | maximal_keepking | balanced-MC 5×9 | PLS(k)+Ridge; ruler-clf | mean, king_last, king_mean | 7 | yes | ±std/200 + chance+shuffle |
| P3 | Matter of Time | tier0 | none (PCA/Isomap+NN) | 1-D embed + nearest-anchor | anchors; mean | 8 | yes | none (random baseline) |
| P7 | Haystack | tier0 | GKF-by-ruler | top-k-neuron logistic | mean | 8 | yes | none (random baseline) |
| T10 | prompting | tier0 in prompt | GKF + balanced-MC 8×21 | PLS(k)+Ridge | span-mean, king_last, king_mean | qwen 1.7/8/32B | no | fold / ±std |

**"Why not X" recap:** Thalesian-AKK **was** run everywhere. **MLM** appears only in P1b
(needs its own extractor — a TODO to extend). **TF-IDF** is a bag-of-signs with no
per-token/per-layer reps, so it's a cited P1 baseline only.

---

## 12. One-paragraph conclusion for the advisor
Across the whole suite the story is consistent and, we argue, robust: the models **know**
the dates (T9) and the **probe pipeline is valid** (P2 geography decodes with real skill),
yet the date is **not** laid out as recoverable structure over text. Whole-text signal is
≈ random ≈ TF-IDF and flat across scale/objective/prompting (P1b, T10); the strong
king-token signal is **name identity** — a **random-initialized network reproduces it**,
and in the fair maximal-with-kings rematch random **matches or beats** every trained model
even on the whole-text mean (P1c); there is no 1-D timeline that real texts inhabit (P3)
and no localized "time neurons" (P7). The lone exception is the cuneiform-domain
**Thalesian** model, which adds a modest, genuine increment over random on geography and on
the whole-text mean. Net: on this hard, leakage-free Akkadian dating task, the
"linear world-model timeline" claim does **not** transfer — the apparent recoverability is
token identity plus probe capacity, not a learned chronology.
