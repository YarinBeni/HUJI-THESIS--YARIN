# Handoff prompt — the thesis presentation and the code behind every slide

**You are picking up a master's-thesis project mid-flight.** Its deliverable is one
self-contained HTML presentation. This file exists so you can understand *everything on
that presentation* — what each slide claims, which experiment produced it, which code
produced that experiment, and where its results live — without spelunking the repo.

Read this file top to bottom first. Repo root: `HUJI-THESIS--YARIN`; every path below is
relative to it. Branch: `main` (mirrored to `claude/thesis-story9-review-j8k29i`).

---

## 1. The deliverable

| What | Where |
|---|---|
| **The presentation** | `v_1/src/stress_tests/results/thesis_story_9.html` — committed on `main`, ~10.5 MB |
| Slide count | **43** (`data-index` 0–42; the on-screen counter is +1, so `data-index="32"` is "slide 33") |
| Self-containment | one file; every figure is inlined as a base64 data URI or hand-written inline SVG. Open it in a browser, no server, no assets |
| Navigation | arrow keys; `F` for fullscreen; `#N` in the URL jumps to slide N |

It is committed — verify with `git ls-files v_1/src/stress_tests/results/thesis_story_9.html`.

Screenshot a slide (Chromium is preinstalled; the explicit `executable_path` is required):

```python
from playwright.sync_api import sync_playwright
url = "file:///ABS/PATH/v_1/src/stress_tests/results/thesis_story_9.html"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(f"{url}#35"); pg.reload(); pg.wait_for_timeout(700)   # reload = actually land on 35
    pg.screenshot(path="s35.png"); b.close()
```

---

## 2. Deck mechanics — the one thing to get right before editing

The file has **two regimes**, and they are edited in completely different ways:

| Slides | Regime | How to change them |
|---|---|---|
| **1–31** (phase 1) | hand-maintained HTML inside the file | edit the HTML directly |
| **32–43** (phase 2) | **generated** — rebuilt from results on every run | edit `v_1/src/phase2/figs/make_deck_slides.py`, then re-run it |

The generated block sits between the markers `<!-- PHASE2-BEGIN -->` and
`<!-- PHASE2-END -->`. The generator is **idempotent**: it strips the old block and
rewrites it, so re-running after new results land is always safe, and it never touches
slides 1–31. Rebuild with:

```bash
cd v_1/src/phase2/figs && python make_deck_slides.py     # prints "[done] 31 -> 43 slides"
```

Three pieces of navigation state must stay in sync with the sections:

* `const TOTAL = 43;`
* `const TITLES = [...]` — the **31 hand-maintained** titles only
* `/*P2TITLES*/TITLES.push(...)/*P2TITLES-END*/` — the 12 generated titles, appended

`make_deck_slides.py` patches `TOTAL` and the push block itself, and assigns every
generated section's `data-index` **from its position in the list**, so inserting or
deleting a generated slide cannot desynchronise the numbering. If you hand-edit slides
1–31, you must update `TOTAL` and `TITLES` yourself. Verify after any edit:

```bash
python3 - <<'PY'
import re, json
h = open("v_1/src/stress_tests/results/thesis_story_9.html", encoding="utf-8").read()
base = json.loads(re.search(r'const TITLES = (\[.*?\]);', h).group(1))
push = re.search(r'/\*P2TITLES\*/TITLES\.push\((.*?)\);/\*P2TITLES-END\*/', h, re.S)
extra = json.loads('[' + push.group(1) + ']') if push else []
print("sections", len(re.findall(r'<section class="slide', h)),
      "| TOTAL", re.search(r'const TOTAL = (\d+)', h).group(1),
      "| titles", len(base) + len(extra))
PY
```

All three numbers must read 43.

**House format** (follow it for any new slide): `.eyebrow` kicker → `h2.sh` one-sentence
claim headline → `.cfg.tight` rows (Task / Method / Data & pooling — prose with the exact
configuration, the formula in `<span class="frm2">`, and a **direct link to the source
paper**) → the figure → `.takeaway.tight` with a bolded lead. There is deliberately **no
dedicated methods slide**; every slide carries its own method.

---

## 3. The scientific story, in six sentences

We extend Gurnee & Tegmark (*Language Models Represent Space and Time*, ICLR 2024), who
show that a linear probe on a frozen LLM's hidden states recovers real-world time and
space for **famous entities, in English, at the last token**. Our thesis is about dating
**low-resource ancient Akkadian royal inscriptions**, so we built a 2×2 climbing map —
salient↔obscure entities × high↔low-resource language — and asked which move breaks the
claim. It survives obscure entities (our 34 Assyrian kings are dated at ρ≈.53–.70) and
dies at **whole documents**, already in English translation, before the language changes.
Phase 1 (slides 1–31) establishes that boundary and kills the obvious rescues (prompting,
more Akkadian, word order, non-linear probes, supervision, tokenizer, training-data
exposure). Phase 2 (slides 32–43) asks *why*, mechanistically: it finds a real, semantic,
causally-usable time axis on the entity side built out of **name-culture (onomastic)**
features, shows it is **orthogonal** to whatever documents encode, and shows **no channel**
carries it into a document representation even under forcing. The conclusion is that the
entity→document collapse is a **disconnection**, not missing knowledge.

---

## 4. Slide → experiment → code → results

Numbers are the **on-screen** slide numbers (`data-index` + 1).

### Phase 1 — slides 1–31 (hand-maintained HTML)

| # | Slide | Cell | Code | Results |
|---|---|---|---|---|
| 1–5 | title, motivation, the paper, the climbing map, protocol | — | narrative only | — |
| 6 | the paper reproduces on our models, with controls it never ran | A | `world_models/extract_acts.py`, `probe_wm.py` | `world_models/results/probes/**`, `summary_best_layer_*.csv` |
| 7 | where space and time live (depth) | A | `probe_wm.py` (per-layer) | `results/summary_layerwise.csv` |
| 8 | how many PLS directions the world model needs | A | `probe_eng_pls.py` | `results/eng_pls/**` |
| 9 | obscure entities in English: date survives, place does not | B | `world_models/akkadian/extract_entity.py`, `probe_entity.py`, `probe_entity_pls.py` | `akkadian/results/probes_entity/**`, `summary_entity_best.csv` |
| 10–11 | whole English fragments: n-grams win; place ≈ untrained twin | B′ | `akkadian/extract_akk.py`, `probe_akk.py`, `tfidf_akk.py`, `probe_geo_site.py` | `akkadian/results/probes/**`, `probes_geosite/**` |
| 12 | inside the best English embedding | B′ | `world_models/manifold/run_manifold.py`, `manifold_figs.py` | `world_models/results/manifold/**` |
| 13 | our own Akkadian MLM | C | `v_1/src/finetune/` (AKK-300M / cuneiform-400M training) | model checkpoints (cluster-local) |
| 14–15 | raw Akkadian: king name readable, chronology not; the twin does it too | C | `akkadian/probe_akk.py`, `probe_entity.py` | `akkadian/results/**` |
| 16–17 | raw Akkadian whole fragments: every model falls to its twin; find-spot survives | C | `probe_akk.py`, `probe_geo_site.py` | `akkadian/results/probes/**`, `probes_geosite/**` |
| 18–19 | depth and dimensionality in Akkadian | C | `akkadian/probe_layers_pls.py` | `akkadian/results/layers_pls/**` |
| 20 | inside the winner's Akkadian embedding | C | `world_models/manifold/` | `results/manifold/**` |
| 21 | the models DO know these kings when asked (rescue 1) | — | `stress_tests/redo_t9_knowledge/` | `stress_tests/results/csv/**` |
| 22 | prompting harder changes nothing (rescue 2) | — | `stress_tests/redo_t10_prompt/`, `t11_gen_dating/`, `t12_forced_dating/` | `stress_tests/results/csv/**` |
| 23 | more Akkadian finetuning moves nothing (rescue 3) | — | `v_1/src/finetune/` + `stress_tests/` probes | `stress_tests/results/csv/**` |
| 24 | word-order scrambling costs almost nothing (rescue 4) | — | `stress_tests/e5_shuffle/` | `results/csv/**` |
| 25 | curved and kernel probes do no better (rescue 5) | — | `stress_tests/p9_gkpls/`, `p10_reduce_kernels/` | `results/csv/**` |
| 26 | the supervision dial (rescue 6) | — | `stress_tests/p8_lambda_probe/` | `results/csv/p8_lambda.csv` |
| 27 | the one arm that separates, only under PLS | C | `stress_tests/p1_gurnee_tegmark/`, `aggregate_tables.py` | `results/csv/table1_best_models.csv` |
| 28 | why: translation finetuning, multilingual most of all | C | `stress_tests/translation/` | `results/csv/translation_mc.csv` |
| 29 | it is not the tokenizer | C | `stress_tests/eda/` | — |
| 30 | **the boundary condition** (the thesis statement) | — | narrative | — |
| 31 | counting the training data: exposure is not the wall | — | `v_1/src/olmo_frequency/` (`count_frequencies.py`, `analyze_frequency.py`, `count_surnames.py`, `plot_frequency_fig.py`) | `olmo_frequency/results/**`, `RESULTS.md` |

### Phase 2 — slides 32–43 (generated by `phase2/figs/make_deck_slides.py`)

Each row names the chart builder inside that generator, so you can trace a pixel to a file.

| # | Slide | Experiment | Chart builder | Results consumed |
|---|---|---|---|---|
| 32 | phase-2 opener: the five moves | — | inline text | — |
| 33 | **erasure ladder** — what the document ordering was made of | E1 + **F28** | `chart_ladder()` | `phase2/erasure/results/ladder.*.json` (48 files) |
| 34 | the two time axes are orthogonal; the ruler axis transfers only by identity | E3 + **E3b** | `chart_orthogonal()` | `phase2/transfer/results/*.mean.json`, `*.mean.assyrian_ruler.json` |
| 35 | logit lens: the year direction reads "ancient", the document direction reads nothing | F6 + F14 + **F29** + **F31** | `chart_lens_tokens()` | `phase2/traces/results/{method}.json` |
| 36 | whole-vocabulary spectroscopy | F21 + **F29/F31** | `chart_spectrum()` | `phase2/traces/results/spectroscopy.{method}.json`, `tuned.{method}.json` |
| 37 | where the year features fire (the gate) | F8 + F11 + F22 | `chart_gating()` | `phase2/sae/results/token_firing.layer24.json`, `feature_hunt.layer24.csv`, `phase2/sae2/results/pipeline.json` |
| 38 | no single "year neuron" — a distributed code | F8 + F22 | `chart_decomposition()` | `sae/results/feature_hunt.layer24.csv`, `sae2/results/feature_hunt2.layer9.csv` |
| 39 | what the features mean — onomastic detectors, in two dictionaries | F25 + **F30** | `chart_feature_cards()` | `sae2/results/feature_interp.layer9.json`, `feature_interp.layer24.json`, `l18_peek.json` |
| 40 | causal test: clamping a feature drags the year read-out | F23 | `chart_causality()` | `sae2/results/steer.layer9.json` |
| 41 | the bridge: forced ON across a document, nothing arrives | F23 bridge | `chart_bridge()` | `sae2/results/steer.layer9.json` (`runs.*.bridge`) |
| 42 | ignition at the ruler's own name | **F26** | `chart_ignite()` | `phase2/steering/results/ignite.json` |
| 43 | synthesis | — | inline text | — |

---

## 5. Phase-2 experiment inventory (the F-numbers)

Every phase-2 experiment is pre-registered in `v_1/src/phase2/DECIDED_EXPERIMENTS.md` and
tracked wave by wave in `v_1/src/phase2/README.md`. The program ran F1–F31 and is closed.

| Code | What it does | Entry point | Job |
|---|---|---|---|
| E1 / F1–F5, F19 | pairwise ordering (Bradley–Terry) of fragments, spec curve, `site=last` | `phase2/pairs/probe_pairs.py`, `spec_curve.py`, `pairs_data.py` | `pairs/sbatch/F19_last_and_spec.sbatch` |
| E8 | ruler-level permutation with full refit + dyadic bootstrap | `phase2/pairs/e8_inference.py` | — |
| E3 / E3b | frozen transfer of an entity axis to documents + LEACE mediation + cos vs the document direction; `--entity-set assyrian_ruler` fits and caches the ruler axis | `phase2/transfer/e3_transfer.py` | `transfer/sbatch/E3b_ruler_axis.sbatch` |
| **F28** | single-concept erasure ladder (ruler / period / sub-genre / provenance / length / year-decile) | `phase2/erasure/e4_confounders.py --concept X` | `erasure/sbatch/F28_ladder.sbatch` |
| F27 | non-linear probes (RidgeCV / kernel-RBF / MLP) under GroupKFold-by-ruler | `phase2/erasure/e4_nonlinear.py` | `erasure/sbatch/F27_nonlinear.sbatch` |
| F6 / F14 | direction-level logit lens + random-direction controls | `phase2/traces/logit_lens.py` | `traces/sbatch/` |
| F21 | whole-vocabulary spectroscopy (10 deciles × 9 categories vs 50 random directions) | `phase2/traces/lens_spectroscopy.py` | `traces/sbatch/` |
| **F29 / F31** | tuned-lens translators + every direction re-read raw and tuned, at **both pooling sites** | `phase2/traces/lens_tuned.py` | `traces/sbatch/F29_tuned_lens.sbatch`, `F31_lens_last.sbatch` |
| F8 / F11 | SAE dictionary #1 (Qwen-Scope, L24): feature hunt, FVU gate, token firing | `phase2/sae/{fvu_gate,feature_hunt,token_firing}.py` | `sae/sbatch/` |
| F22 | SAE dictionary #2 (Karvonen batch-TopK, L9 — the only layer passing FVU ≤ .35) | `phase2/sae2/run_pipeline.py`, `karvonen.py` | `sae2/sbatch/F22_sae2_pipeline.sbatch` |
| F23 / F23b | feature clamping with rate-matched controls: amplify, ablate, bridge; `--population cellB` | `phase2/sae2/feature_steer.py` | `sae2/sbatch/F23_feature_steer.sbatch` |
| F25 / **F30** | feature reading: max-activating contexts + Golden-Gate clamped generation; `--sae1` reads dictionary #1; labelled L18 peek | `phase2/sae2/feature_interp.py`, `l18_peek.py`, `fetch_labels.py` | `sae2/sbatch/F30_wave.sbatch` |
| **F26** | ignition: clamp confined to the offset-mapped ruler-NAME token span | `phase2/steering/ignite_anchor.py` | `steering/sbatch/` |
| F15–F18 | length / find-spot / pooling / seriation controls | `phase2/seriation/e7_seriation.py`, `phase2/esarhaddon/e6_micro.py` | respective `sbatch/` |

---

## 6. Data, conventions, and the traps that have already bitten

**The four cells.** A = 7,507 famous historical figures (English name prompt, last token of
the name). B = our 34 Assyrian/Babylonian rulers, same protocol. B′ = 1,187 dated fragments
as literal English glosses. C = the same 1,187 as cleaned Akkadian transliteration. Genre is
constant — every dated fragment is a Royal Inscription — so genre cannot be a confounder.

| Data | Path |
|---|---|
| Akkadian corpus | `v_1/data/evaluation/corpora/orcc_corpus.parquet` (`fragment_id, ruler, year, provenance, period, sub_genre, word_count, text_maximal, text_tier0`) |
| Translations | `v_1/src/stress_tests/translation/translations.parquet` — use `eng_tier0` (faithful gloss); **`eng_maximal` hallucinates king names and is excluded everywhere** |
| Entity datasets | `v_1/src/world_models/data/entity_datasets/*.csv` (`historical_figure.csv`, `assyrian_ruler.csv`, …) |

**Traps, in order of how much time they have cost:**

1. **`*.npz` is gitignored.** Activations *and* probe direction vectors live only on the
   cluster. Anything that reads a direction (`e3_transfer.py`, `logit_lens.py`,
   `lens_tuned.py`) must run there. Locally these globs return nothing — that is expected,
   not a bug.
2. **Year polarity.** The fragment `year` column is **BC-positive** (Ashurbanipal = 631,
   larger = *earlier*), while entity `death_year` is CE-signed (larger = later). Trained
   probes are immune (a global label flip leaves accuracy invariant), but a **frozen**
   scorer imported from a CE-signed dataset reads as ρ < 0 / macro < .5 when it is working.
   `e3_transfer.py` now also stores `spearman_lateness` / `pairwise_macro_lateness`; older
   result files carry only the raw keys. Always compare magnitudes **against the untrained
   twin**, never against .5. Documented in `pairs_data.draw_pairs`.
3. **ICC = 1.** Ruler identity determines the year almost perfectly in this corpus, so
   "erase ruler" is necessarily also "erase most of the era". Read the ladder's ruler rung
   as a joint upper bound, which is why year-decile is carried as a positive control.
4. **Never claim an effect without its control.** Every steering result is
   treated-minus-**firing-rate-matched-control**; every lens result is calibrated against
   random directions pushed through the *identical* pipeline (including the tuned-lens
   translator). A null *with* controls is a publishable result here and several of the
   headline findings are exactly that.
5. **Effective n is 40 rulers, not 1,187 fragments.** Every evaluation uses
   both-rulers-held-out folds and macro-averaging over ruler pairs; significance is a
   ruler-level permutation with full refit (B = 150, so the floor p is 1/151 = .0066).
6. **Cluster jobs self-sync.** Every sbatch sources `v_1/src/stress_tests/sbatch/_common.sh`
   and calls `sync_main` (fetch + rebase `--autostash`) then `commit_push`, so a job always
   runs the latest `main` and pushes its own results. A dirty working tree on the login node
   does not affect queued jobs.

---

## 7. Documentation index

| File | What it is |
|---|---|
| `HANDOFF_PROMPT.md` | **this file** — the entry point |
| `PRESENTATION_CONTEXT.md` | earlier orientation doc; sections 1–3 (what we extend, the matrix, where phase-1 code lives) are still good, **its slide map is stale** (written at 33 slides) |
| `v_1/src/phase2/README.md` | wave-by-wave tracker with the verdict of every phase-2 experiment |
| `v_1/src/phase2/TEACHING_GUIDE_HE.md` | full Hebrew teaching guide — per-experiment method, configuration, numbers, and the slide-mapping table |
| `v_1/src/phase2/DECIDED_EXPERIMENTS.md` | the pre-registration: hypotheses and decision rules, written before the runs |
| `v_1/src/phase2/pairs/RESULTS.md`, `sae2/RESULTS.md` | detailed result tables + a bug-audit table |
| `v_1/src/stress_tests/results/DECK_DEEP_DIVE.md`, `STORY_SPINE.md` | phase-1 deck rationale, slide by slide |
| `v_1/src/world_models/EXPERIMENT_MAPPING.md` | paper-dataset ↔ our-analog table |

---

## 8. Where things stand

The experimental program is **closed**: F1–F31 have all run, and their verdicts are folded
into the deck, the README and the teaching guide. The deck is at 43 slides. Phase 2's
findings, in one line each:

* the document "ordering" decomposes into ruler identity, era, object type and find-spot —
  and an **untrained twin loses just as much** at every rung (F28);
* the entity time axis and the document axis are **orthogonal at chance level**, and the
  ruler axis transfers only through ruler identity (E3, E3b);
* the entity axis is **semantically temporal** across the whole vocabulary, under a trained
  translator and at both pooling sites; the document directions are indistinguishable from
  random directions (F6, F21, F29, F31);
* the axis is a **distributed code of onomastic features**, replicated in two independent
  SAE dictionaries and corroborated by third-party labels (F8, F22, F25, F30);
* those features **causally** drive the year read-out on entities, but nothing propagates
  to documents — not across a whole document, not at the ruler's own name (F23, F26).

Open, optional: nothing blocking. The next step is writing the thesis.

**House rules for whoever continues:** verify every number against the result files before
putting it on a slide (the generator reads them directly — prefer that over typing numbers
in); keep the per-slide method/config/formula/paper-link format; commit to `main` and
mirror to `claude/thesis-story9-review-j8k29i`; do not open pull requests unless asked.
