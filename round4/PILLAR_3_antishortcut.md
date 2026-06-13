# Pillar 3 — Anti-shortcut training (the thesis's real contribution)

> **Agent brief.** This is where the thesis stops being "another dating model" and becomes
> "honest, shortcut-resistant dating". Read `README.md` first. **Requires P0 and P2 merged.**
> CPU-only. The deliverable is not a higher Spearman — it's a model whose score *survives*
> name/formula masking and a ruler/genre adversary.

## Goal

Add the plan's **Stage 2 + Stage 3** on top of P2's ChronoHead:
```
L = L_pairwise_rank + L_interval + λ2·L_consistency + λ4·L_nuisance_adversary
```
plus **better pair design** (cross-genre positives, same-genre hard negatives) and the full
**anti-leak evaluation battery**. The point: prove the chronological coordinate is not a
"dear-X" / ruler-name / length shortcut.

## Dependencies

**P0:** `transforms.py` (the **runtime** masking/crop/normalize callables — apply them in the data
loader's `__getitem__`, never pre-materialize; see README "Engineering principle"), `labels.py`
(pairs). **P2:** `model.py`, `train.py` (you extend these, don't fork). No GPU.
(Note: the P4 graph-smoothness term is **PARKED** — do not add `--graph-weight`; ignore any
mention of a Stage-4 graph term below.)

## What to read (repo)

- `README.md` §3 — **the genre caveat is critical here:** ORCC is ~all royal inscriptions, so
  leave-one-genre-out is cross-corpus (ORCC vs letters/SEAL) and coarse. Do not claim a clean
  genre-held-out year test; report what is actually possible and say so.
- P2's `model.py` / `train.py` (extend with new loss terms behind flags).
- `v_1/src/chronorank/transforms.py` (P0, runtime callables) and `v_1/src/linear_probing/name_masking.py`.
- `v_1/src/geodesic/results/loro_robustness*.json` — there is **already** a leave-one-ruler-out
  robustness result for the geodesic probe. Match its protocol so your LORO numbers are comparable.

## What to read (papers)

- **DANN / Gradient Reversal Layer** (Ganin & Lempitsky) — the nuisance adversary mechanics.
  Train an MLP to predict {ruler, genre, length-bucket, corpus, name-present, formula-type} from
  the representation; reverse its gradient into the encoder.
- **SimCSE** — augmentation-consistency intuition (pull views of the same text together). Here the
  views are historically motivated (P0 `transforms.py`, applied at runtime), not random.
- **Rank-N-Contrast (RnC)** — for cross-genre positive / same-genre hard-negative pair mining.

## What to build

### Extend `train.py` with flags
```python
--consistency-weight λ2     # |s(x) - s(view(x))|^2 averaged over P0 runtime transforms (name_masked, formula_removed, crop32, normalized), applied in __getitem__
--adversary ruler,genre,length,corpus,name_present,formula_type   # GRL heads; comma list
--adversary-weight λ4
--pair-mining cross_genre   # positive=near-date/different-corpus, hard-neg=same-corpus/far-date
```
**Ruler-adversary caveat (from the plan):** ruler is both label-source and confound. Run **both**
`--adversary ruler` and without, and report the tradeoff — do not silently strip all ruler info.

### New: `v_1/src/chronorank/eval_robustness.py`
The anti-leak battery. For a trained head, produce one table:
```
clean maximal-balanced
leave-one-ruler-out            (match geodesic loro protocol)
leave-one-corpus-out           (ORCC vs letters/SEAL — coarse, labeled as such)
name-masked  (delta vs clean)
formula-removed (delta vs clean)
counterfactual formula insertion (cross-period formula spliced in; score should NOT fully jump)
fragment-length curve  {8,16,32,64,full} words
date-shuffle placebo   (shuffle years within ruler; score should collapse to ~chance)
```
Report metric block (`eval_ordinal.full_report`) per row, and **deltas** vs the clean row.

## Cluster / sbatch

CPU-only. One sbatch that trains the staged variants and runs the battery:
```bash
# v_1/src/chronorank/sbatch/P3_antishortcut.sbatch  (no --gres)
#SBATCH --cpus-per-task=32 --mem=64G --time=04:00:00
# ... standard preamble ...
# Stage ablation on Thalesian L11 maximal/mean:
python train.py --method thalesian_akk300m --layer 11 --head mlp1 --consistency-weight 0.5 --out results
python train.py --method thalesian_akk300m --layer 11 --head mlp1 --consistency-weight 0.5 --adversary ruler,genre,length,corpus --adversary-weight 0.3 --out results
python train.py --method thalesian_akk300m --layer 11 --head mlp1 --consistency-weight 0.5 --adversary genre,length,corpus --adversary-weight 0.3 --out results   # no-ruler variant
python eval_robustness.py --runs results --out results/robustness_table.json
# commit results + push (FT3 pattern)
```
Give Yarin the one-line paste. Run a 1-epoch local smoke first.

## Report back / success criterion

**PASS** when `robustness_table.json` shows the model's predictions are **stable under name and
formula masking** (small delta) and **the date-shuffle placebo collapses to chance** (proving no
leakage path). Per README §4: a model at Spearman ≈ 0.43 that holds up under masking and LORO
**beats** a 0.48 model that collapses. State the consistency deltas and the placebo result
explicitly — those two numbers are the headline of this pillar.
