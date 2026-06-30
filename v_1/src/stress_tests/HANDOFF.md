# Stress-Tests — Session Handoff

> Pick up here in a new session. Branch: `claude/stress-test-timeline-analysis-9sh2vs`
> (also pushed to `main`). PR #1. Everything lives under `v_1/src/stress_tests/`.

## 1. The thesis claim being tested
**Declarative knowledge and linearly/geometrically recoverable representation are
separable in LLMs.** Models *state* Assyrian reign dates well, but their
representations recover the date only weakly, near surface statistics, and that
doesn't improve with scale, prompting, or NTP finetuning — while a translation
model (Thalesian) recovers more (→ objective, not scale). We stress-test the
"LLM world-model geometry" papers (Gurnee–Tegmark, Godey, A Matter of Time,
k-sparse) on low-resource, indirect (date-not-in-text), no-web-leakage Akkadian.

## 2. Data / protocol facts
- Corpus: `v_1/data/evaluation/corpora/orcc_corpus.parquet` — 1,202 royal
  inscriptions, `year` BCE (9 nulls), 41 rulers (very imbalanced), `provenance`.
- **Pooling sites:** `mean` (whole text, tier0+maximal), `king_last`, `king_mean`
  (last / mean of the commissioning ruler's name span, **tier0 ONLY** — maximal
  cleaning strips the logographic names).
- **King-token coverage is intrinsically ~37–44%** (Neo-Babylonian texts are
  administrative, never name the king). `shared/ruler_spellings.csv` is a
  first-pass map and **needs the Assyriologist's review** to raise coverage.
- **Two CV protocols:**
  - **balanced-MC** (`draws_matrix.npy`, 200 balanced draws × GroupKFold-by-ruler)
    — *this is the thesis's headline protocol* (the 0.41 lineage). Code:
    `shared/mc_probe.py`, `probe_p1_mc.py`, `reprobe_king_mc.py`.
  - GroupKFold-by-ruler (single, unseen-ruler generalization) — the first run.
- Geography (P2) uses GroupKFold-by-site. P7 GroupKFold-by-ruler.

## 3. What's built (all committed)
```
shared/   king_token.py, probe_sites.py, extract_lib.py, metrics.py, anchors.py,
          mc_probe.py, geo_loader.py, ruler_spellings.csv, sites_gazetteer.csv
p1_gurnee_tegmark/  extract_king_acts.py (J4), probe_p1.py (GKF), probe_p1_mc.py (MC)
p2_godey_geography/ probe_p2.py (J7)
p3_matter_of_time/  extract_anchor_acts.py (J5), timeline_p3.py (J8)
p7_ksparse/         probe_p7.py (J9)
redo_t9_knowledge/  (reuses round2_phase1a run_kp/parse_kp/score_kp)  J2
redo_t10_prompt/    extract_prompted_king_acts.py, reprobe_king_pv.py (GKF),
                    reprobe_king_mc.py (MC)              J3 / J3r
sbatch/   J2a,J2b, J3a,J3b, J3r_t10_reprobe_mc, J4,J4b, J5, J6_p1_probe,
          J6_p1_mc, J7, J8, J9, submit_all.sh
```
`submit_all.sh` launches everything with afterany deps (J6 after J4/J4b, J8 after J5).

## 4. Results so far (GroupKFold run; MC running now in jobs 12258/12259)
- **T9 knowledge (kp0 ±50yr):** gpt-oss 8/8, qwen 1.7B 7/8, 8B 7/8, 32B 6/8 → models KNOW dates.
- **P2 geography (positive control, PASSES):** all decode find-spot ~174–207 km,
  skill +0.22–0.35 vs centroid; **random weights +0.221**; **thalesian-cunei400m
  best +0.347**. Pipeline is valid; mild scale effect.
- **P1 dating, mean-pool (GKF):** qwen year-Spearman ≈ 0.48 (tier0)/0.40 (maximal);
  **MLP ≈ 0** (no nonlinear gain). King-site numbers come from the MC run.
- **P7 k-sparse (chance 0.58):** best macro-F1 only 0.67–0.72; **random 0.667** →
  date weak/distributed, not localized.
- **Emerging story:** state-fact ✓, geography(explicit) decodes, date(indirect)
  weak/near-surface, not nonlinear, not localized; Thalesian tops the ladder.

## 5. What's RUNNING (as of handoff)
- `12258 J6_p1_mc` — P1 balanced-MC (mean tier0/maximal + king_last/king_mean tier0), all 8 methods.
- `12259 J3r_t10_mc` — T10 balanced-MC reprobe on existing prompted acts.
- Possibly still: an old GroupKFold `J3a` 32B (`12221_2`) — harmless.

## 6. What's LEFT / known issues
1. **Pull MC results & interpret** when 12258/12259 finish → fill the balanced-MC
   table (mean vs king_last vs king_mean, with shuffled null) — the apples-to-apples
   dissociation vs the 0.41 headline.
2. **P3 timeline (J8) produced no results** — needs a look (anchor `.npz` are
   gitignored/cluster-local; J8 reads them locally, but `p3_matter_of_time/results/`
   is empty). Re-run J8 on the cluster and check its log.
3. **gpt-oss-120B T10** never succeeded (OOM even with sdpa+gpu:4). Optional; the
   7-model ladder stands without it. If wanted: shard more / shorter prompts / fewer layers.
4. **Cluster git divergence:** local `main` diverged from origin. Jobs' result
   push-back may fail ("WARN pull failed") → results sit on cluster disk. **Salvage** with:
   ```bash
   cd ~/projects/HUJI-THESIS--YARIN
   git config pull.rebase true
   git add -A v_1/src/stress_tests && git commit -m "salvage results" || true
   git pull --rebase origin main      # resolve any conflicts: prefer both (different files)
   git push origin HEAD:main
   ```
5. **king coverage metadata** only committed for thalesian×2 + gpt-oss so far; qwen3
   king coverage json may need salvage (acts exist on cluster; J6_p1_mc reads them).
6. `ruler_spellings.csv` expert review (raises king coverage).
7. **GUI (J10) + final aggregate tables/figures (J11)** — not built yet.
8. Cosmetic: stop-hook flags "Unverified" commits (no signing key here) — ignore.

## 7. Next commands (after 12258/12259 finish)
```bash
# salvage + interpret (see §6.4). Then if P3 empty, re-run timeline:
sbatch v_1/src/stress_tests/sbatch/J8_p3_timeline.sbatch
# to interpret locally in a session: git pull, then read
#   p1_gurnee_tegmark/results/mc/p1_year_mc__*.json
#   redo_t10_prompt/results/*__t10_mc_summary.json
```
HF model ids: Qwen/Qwen3-{1.7B,8B,32B}, openai/gpt-oss-120b, Thalesian/AKK_300m,
Thalesian/cuneiformBase-400m, google/umt5-base. Cluster: Slurm `voltagepark`,
`conda activate thesis`, repo at `~/projects/HUJI-THESIS--YARIN`, work on `main`.
