# E-MIN result (P1 minimal experiment) — 2026-09-02

Features: `Thalesian/AKK_300m`, L8 (top block), mean pooling, from EmbStore (C1).
Folds: frozen `gkf_ruler` (5). Read-out: SLA §7 — gkf pooled over centred
out-of-fold scores; mc = mean Spearman over the frozen `mc_balanced` draws.
Per-condition score = mean over that document's views of the chain, **both
languages pooled** (Akkadian + English gloss), for head and baselines alike.

## mc ρ by condition

| condition | PLS k=2 | ridge (orig views) | ridge (all views) | **Chrono-Barlow head** (5 seeds) |
|---|---|---|---|---|
| `orig` | .165 | .376 | .377 | **.398 ± .022** |
| `mask_ruler` | .168 | .333 | .345 | **.370 ± .018** |
| `strip_formula` | .168 | .374 | .377 | **.398 ± .023** |
| `mask_ruler,strip_formula` | .173 | .344 | .346 | **.372 ± .020** |
| `mask_ruler,crop16` | .093 | .271 | .277 | **.320 ± .013** |
| `mask_ruler,crop32` | .128 | .305 | .318 | **.359 ± .014** |
| `orthonorm` | .152 | .389 | .394 | **.398 ± .024** |
| `mask_ruler,drop_span` | .174 | .332 | .340 | **.373 ± .020** |

gkf-pooled ρ on `orig`: PLS .246 · ridge .313 · ridge-all-views .271 · head .317 ± .025.
Ruler-block null of the reported statistic: .002 ± .013 (95 % ≤ .03). Doc placebo ≈ 0.
Baselines are one cross-fit each (no seed spread); the head has five.

Language check (ridge, orig views): Akkadian-only .288 (= C2's .287, so the
pipeline is consistent), English-gloss-only .378. Pooling the two languages
is what lifts every method's `orig` into the high .30s.

## What it says

1. **On unmodified text the head ≈ ridge.** +.02 mc, +.004 pooled — inside
   the head's own seed spread. No headline gain from the objective.
2. **Under corruption the head degrades least, by a small margin.** Name
   masking: head −.03, ridge −.04, ridge-all-views −.03. crop16: head −.08,
   ridge −.11, ridge-all-views −.10. Training ridge on the views buys about
   half of the head's robustness margin; the Barlow objective buys the rest
   (+.02 to +.04 mc). Real but small, and from single baseline fits.
3. **Masking ruler names barely hurts anyone** (−.01 to −.04). At these
   features the probes do not lean on the names — the confound the
   invariance objective is built to remove is small here. Cropping to 16
   words hurts every method by a similar amount.
4. **PLS k=2 is not a serious baseline at this layer** (.165). The M.Sc.'s
   .352 does not reproduce under the pooled read-out (see
   `docs/gate_reference.md`); ridge is the baseline to beat.

The plan's pre-stated success criterion ("degrades ≤ half as much as PLS
under masking / formula removal") is **not met as written**, because PLS
does not degrade under those conditions at all. The criterion assumed a
fragility that the frozen probes do not show at this encoder.

## Caveats to carry forward

* Per-language read-out is not possible from the current score schema
  (`lang` missing); add it before any P1 run.
* 136 Akkadian documents share byte-identical text in 50 groups (4 groups
  span rulers): decide dedupe / exemplar-group ids in the corpus contract.
* One baseline fit per arm; give the baselines a seed spread (bootstrap the
  fold assignment or the alpha path) before quoting margins of .02–.04.
* The 13 century-coded documents (M.Sc. data bug) are still in the corpus.

Sources: `emin_summary.md`, `emin_baseline_*.md`, `scores/*.parquet`,
`results.parquet`; scripts `aggregate_emin.py`, `baseline_conditions.py`.
