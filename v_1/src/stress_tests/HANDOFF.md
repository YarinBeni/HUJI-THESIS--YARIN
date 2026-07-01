# Stress-Tests — Session Handoff / Context

> Read this first. A fresh agent should be able to carry on from here.
> Branch: `claude/stress-test-timeline-analysis-9sh2vs`. **The cluster runs from
> `main`** (every sbatch does `git pull --rebase origin main` … `git push origin
> HEAD:main`). This agent's environment BLOCKS direct pushes to `main` (auto-mode
> classifier), so new code is pushed to the feature branch, then landed on `main`
> **on the cluster login node** (not blocked there). Because the cluster's own jobs
> also push result JSONs to `main`, `main` may diverge from the branch's base — use
> a MERGE, not `--ff-only`: `git fetch origin && git checkout main && git merge
> --ff-only origin/main && git merge --no-edit origin/claude/stress-test-timeline-analysis-9sh2vs && git push origin main`.
> Code and result JSONs touch disjoint files, so the merge is conflict-free.
> (Last landed: `85db68b8`.) All work under `v_1/src/stress_tests/`. The user runs
> every cluster job by pasting `sbatch`; the agent NEVER SSHes.

---

## 1. The thesis question and the current (refined) finding
We stress-test the "LLMs build a world-model timeline" literature (Gurnee–Tegmark,
Godey geography, A Matter of Time, k-sparse "Finding Neurons in a Haystack") on
**low-resource, indirect, no-web-leakage Akkadian dating** (ORCC royal inscriptions,
year BCE from ruler).

**Refined claim from the balanced-MC results (see §6):** the date is **linearly
recoverable at the king's-name token** (`king_last` ≈ 0.5–0.7 Spearman) but the
whole-text **mean-pool ≈ 0.33–0.41 ≈ random (0.376)** — i.e. the model encodes the
date *locally at the explicit carrier* (replicating Gurnee–Tegmark's last-entity-token
result) but **does NOT diffuse it into a recoverable text-level chronological geometry**,
and this is not moved by scale, prompting, or objective. Declarative knowledge is
present locally (king token) and statable behaviorally (T9), yet absent as a global
structure over text. **The decisive missing control is `random king_last`** (§7.1).

---

## 2. Data + protocol (must respect)
- Corpus: `v_1/data/evaluation/corpora/orcc_corpus.parquet` (1,202 texts, `year` BCE
  with 9 nulls, 41 imbalanced rulers, `provenance`).
- **Pooling sites:** `mean` (tier0+maximal) and `king_last`/`king_mean` (last / mean of
  the commissioning ruler's name span, **tier0 ONLY** — maximal strips logographic names).
- **King-name coverage is intrinsically ~37–44%** (Neo-Babylonian texts are admin, never
  name the king). `shared/ruler_spellings.csv` is first-pass; needs Assyriologist review.
- **Two CV protocols, both present:**
  - **balanced-MC** = `draws_matrix.npy` (200 balanced draws) × GroupKFold-by-ruler within
    each, best-k Spearman averaged. THIS is the thesis-headline protocol (the 0.41 lineage).
    Files: `shared/mc_probe.py`, `p1_gurnee_tegmark/probe_p1_mc.py`, `redo_t10_prompt/reprobe_king_mc.py`.
  - GroupKFold-by-ruler (single) = the earlier run (`probe_p1.py`, `reprobe_king_pv.py`).
- Random baseline = Qwen3-8B, `from_config`, seed 42, bf16 (matches on-disk `random` acts).

---

## 3. Code map (all committed on `main`)
```
shared/
  king_token.py       locate commissioning ruler's name span (word + tokenizer offsets)
  probe_sites.py      mean / king_last / king_mean poolers
  extract_lib.py      model load (causal/encoder; sdpa; random=from_config; umt5 fallback) + pooling
  mc_probe.py         balanced-MC engine (draws_matrix × GroupKFold; partial-coverage king)
  geo_loader.py       unambiguous import of geodesic/utils.py (avoids utils.py name clash)
  metrics.py          reuse pls_utils.compute_metrics + proximity_error + great_circle
  anchors.py          P3 ruler/year anchor prompts
  ruler_spellings.csv NEEDS EXPERT REVIEW (raises king coverage)
  sites_gazetteer.csv provenance -> lat/lon/region (P2), 97.5% row coverage
p1_gurnee_tegmark/  extract_king_acts.py (J4/J4c HF king), extract_mlm_king_acts.py (J4d
                    MLM king+mean, sign-level), probe_p1.py (GKF), probe_p1_mc.py (MC)
p2_godey_geography/ probe_p2.py (J7) — now sweeps PLS k + Ridge, best-k per lat/lon
p3_matter_of_time/  extract_anchor_acts.py (J5/J5b/J5c; --random flag), timeline_p3.py (J8)
p7_ksparse/         probe_p7.py (J9)
redo_t9_knowledge/  uses round2_phase1a run_kp/parse_kp/score_kp (J2)
redo_t10_prompt/    extract_prompted_king_acts.py, reprobe_king_pv.py (GKF), reprobe_king_mc.py (MC)
aggregate_tables.py J11 — builds results/RESULTS_stress_tests.md (labels + TF-IDF cite)
sbatch/             J2a,J2b,J3a,J3b,J3r_t10_reprobe_mc,J4,J4b,J4c_king_random,J4d_king_mlm,
                    J5,J5b_p3_anchors_gptoss,J5c_p3_anchors_random,J6_p1_probe,J6_p1_mc,
                    J7,J8,J9,J11_aggregate, submit_all.sh
```

**mc_probe / probe_p1_mc / probe_p2 now report BOTH PLS (swept k∈{1,2,3,5},
best-k surfaced + full per_k) AND a Ridge arm** (the user wanted both). Result
JSONs gained `best_k`, `per_k`, and `ridge{spearman_mean,…}`; old printers still
read the flat best-k keys. Re-run J6_p1_mc + J7 to regenerate with these.

## 4. The jobs (what each wanted to do)
| Job | Purpose | GPU |
|---|---|---|
| J2a/J2b | T9 direct knowledge (kp0/kp1/kp2) on qwen3×3 / gpt-oss | yes |
| J3a/J3b | T10 prompt-reprobe (pv0-3), extract prompted acts (mean+king) | yes |
| J3r_t10_reprobe_mc | T10 reprobe under balanced-MC on existing prompted acts | CPU |
| J4/J4b | king-token extraction (tier0) qwen3×3+thal×2+umt5 / gpt-oss | yes |
| J4c_king_random | king-token extraction for RANDOM Qwen3-8B (the control) | yes |
| J4d_king_mlm | MLM king+mean acts on balanced-MC setup (mlm_{tier0,maximal}_mean + kinglast/kingmean) | yes |
| J5 | P3 anchor embeddings (qwen×3, thal×2, umt5) | yes |
| J5b/J5c | P3 anchors for gpt-oss-120B / random-Qwen3-8B | yes |
| J6_p1_probe | P1 year-probe (GroupKFold) | CPU |
| J6_p1_mc | P1 year-probe balanced-MC (mean + king sites) | CPU |
| J7 | P2 geography (positive control) | CPU |
| J8 | P3 timeline (3a anchors-form-line, 3b texts-project); now incl. gpt-oss + random | CPU |
| J9 | P7 k-sparse localization | CPU |
| J11_aggregate | build results/RESULTS_stress_tests.md (P1+P2 tables, labels, TF-IDF cite) | CPU |

## 5. Where results live
- **In git / local (pull `main`)** — all *result JSONs*:
  - `p1_gurnee_tegmark/results/*.json` (GKF) + `results/mc/*.json` (balanced-MC)
  - `p2_godey_geography/results/*.json`, `p7_ksparse/results/*.json`
  - `p3_matter_of_time/results/p3_timeline__*.json`
  - `redo_t9_knowledge/direct_kp_*/scores|parsed|raw/*.json`
  - `redo_t10_prompt/results/*__t10_king_summary.json` (GKF) + `*__t10_mc_summary.json` (MC)
  - king coverage: `…/orcc__embed/activations/<method>_tier0_king{last,mean}/{metadata,king_coverage}.json`
- **Cluster-only (gitignored, `*.npz` + `*.out`)** — activations (mean/king/prompted/anchor)
  and job logs. NEVER commit these (they broke pushes; `*.out` and `**/logs/` are gitignored).

## 6. Results so far (headline)
- **T9 knowledge (kp0 ±50yr):** gpt-oss 8/8, qwen 1.7B 7/8, 8B 7/8, 32B 6/8 → models KNOW dates.
- **P2 geography (positive control, PASSES):** find-spot decodes 174–207 km, skill +0.22–0.35 vs
  centroid; random +0.221; thalesian-cunei400m best +0.347. Pipeline valid; mild scale effect.
- **P7 k-sparse (chance 0.58):** best macro-F1 only 0.67–0.72; random 0.667 → date weak/distributed.
- **P1 balanced-MC (200 draws) — Spearman:**
  | model | mean t0 | mean max | king_last | king_mean |
  |---|---|---|---|---|
  | qwen3-1.7B | 0.371 | 0.355 | 0.622 | 0.214 |
  | qwen3-8B | 0.365 | 0.363 | 0.507 | 0.207 |
  | qwen3-32B | 0.399 | 0.340 | 0.658 | 0.209 |
  | gpt-oss-120B | 0.404 | 0.330 | 0.666 | 0.224 |
  | thal-akk300m | 0.344 | 0.322 | 0.691 | 0.083 |
  | thal-cunei400m | 0.411 | 0.411 | 0.574 | 0.072 |
  | umt5-base | 0.334 | 0.295 | 0.454 | 0.272 |
  | random Qwen3-8B | 0.376 | 0.303 | **PENDING (J4c→J6)** | PENDING |
  | MLM (J4d) | ~0.42* | PENDING | PENDING | PENDING |
  | TF-IDF (cited) | 0.407 | — | n/a (no token) | n/a |
  (*MLM mean tier0 ≈ 0.424 in the former `balanced_mc_scoreboard.json`; J4d adds its
  maximal-mean + king sites. TF-IDF cited from that scoreboard: PLS 0.407 / Ridge 0.355.)
  (null/shuffled ≈ 0.01 everywhere.) mean-pool ≈ random & flat across scale/objective;
  king_last much higher; king_mean washes out. **These PLS numbers are pre-ridge;
  re-run J6_p1_mc to add the Ridge column + best-k, then J11 rebuilds the table.**
- **T10 balanced-MC (qwen3-1.7B):** mean = 0.406 across ALL pv0–pv3 (prompting doesn't change it);
  king_last 0.49–0.53; king_mean ≈ 0.

## 7. NEXT STEPS (what's left)
0. **DONE — code merged to `main`** at `85db68b8` (ort merge, no conflict) and the **full
   re-run wave was submitted 2026-07 as jobs 12274–12282** with dependencies:
   `J4c(12274) J4d(12275)` → `J6(12276)`; `J7(12277)`, `J3r(12278)`; `J5b(12279) J5c(12280)`
   → `J8(12281)`; all → `J11(12282)`. Verify with
   `sacct -j 12274-12282 --format=JobID,JobName%22,State,ExitCode,Elapsed -X`.
   **Watch:** J5b (gpt-oss anchors, gpu:4) may OOM and finish in <4 min — if so J8 still runs
   (afterany) and just drops the gpt-oss timeline curve. J4d (MLM) finishing fast is normal
   (tiny model); confirm its log ends `[mlm] DONE king_coverage=0.44…`.
1. **When the queue drains:** `git pull origin main` then read
   `v_1/src/stress_tests/results/RESULTS_stress_tests.md` (J11 output). **[DECISIVE] random
   `king_last`:** if ≈0 while pretrained ≈0.66 → "date-at-the-name" is real signal (claim
   airtight); if high → name-token identity, reinterpret. If any job FAILED, re-submit just
   that one (+ its dependents) — the sbatch are idempotent (they overwrite their own outputs).
2. **T10 balanced-MC for qwen3-8B/32B (+gpt-oss).** J3r_t10_reprobe_mc only produced qwen3-1.7B
   (others' prompted acts may not have been on disk at run time). Re-run J3a for 8B/32B if their
   `acts_<model>/prompted_king` are missing, then `sbatch J3r_t10_reprobe_mc.sbatch`.
3. **P3 timeline** (`p3_matter_of_time/results/p3_timeline__*.json`) — results exist now; interpret
   3a (anchors form ordered line) vs 3b (texts project) for the dissociation figure.
4. **gpt-oss-120B T10** never succeeded (OOM even sdpa+gpu:4). Optional; ladder stands without it.
5. **ruler_spellings.csv** expert review to raise king coverage above ~44%.
6. **GUI (J10)** — add new embeddings to `v_1/src/viz/seal_eda*.html`. **J11 aggregate DONE**
   (`aggregate_tables.py` → `results/RESULTS_stress_tests.md`, labels + TF-IDF cite); rerun after
   the wave above to fill in Ridge/best-k/MLM/random-king cells.
7. Write-up: the claim in §1, with the balanced-MC table as the centerpiece.

## 8. Operational notes (cluster + git)
- Cluster: Slurm `voltagepark`, `conda activate thesis`, repo `~/projects/HUJI-THESIS--YARIN`, work on `main`.
- **Only commit result JSONs.** `*.out` and `**/logs/` are gitignored; `*.npz` gitignored. Big logs
  previously broke pushes (100 MB limit) — if it happens: `git rm --cached` the *.out / delete them.
- Divergence fix (if `git pull` complains): `git config pull.rebase true` then `git pull`; or grab
  specific files with `git checkout origin/main -- <path>`.
- HF ids: Qwen/Qwen3-{1.7B,8B,32B}, openai/gpt-oss-120b, Thalesian/AKK_300m,
  Thalesian/cuneiformBase-400m, google/umt5-base. draws_matrix + fragment_order live in
  `v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/`.
- The stop-hook "Unverified commits" warning is cosmetic (no signing key) — ignore.
