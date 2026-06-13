# Pillar 6 — ChronoAtlas: retrieval + evidence reports

> **Agent brief.** This turns the dating head into a **historian-facing research tool** (the
> Ithaca/Aeneas model): not just a year, but an interval + confidence + nearest dated parallels +
> earlier/later evidence + confound warnings. Read `README.md` first. **Requires P2; P3 and P5
> make it much better.** CPU-only.

## Goal

For any (dated or undated) Akkadian fragment, output:
```
Predicted interval: 690–650 BCE   Confidence: medium
Nearest dated parallels:  Text A (ruler X, 681–669 BCE, sim 0.82) ; Text B ... ; Text C ...
Evidence pushing later:   <SAE feature / sparse-head weights>     (from P5 if available)
Evidence pushing earlier: <...>
Confound warnings:        <features flagged as ruler/length/genre leakage>
```

## Dependencies

**P2 (required):** `model.py` (mu/sigma → interval, confidence from sigma), the frozen embedding
space for similarity. **P3 (preferred):** confound flags come from the adversary/robustness work.
**P5 is DEFERRED**, so for now **evidence = the sparse-linear head's per-feature weights** (the
default path). If/when P5 is un-deferred, swap in its SAE dossiers — but do not wait on it. No GPU.

## What to read (repo)

- P2 `model.py` / `train.py` (the trained head + learned space).
- `v_1/src/geodesic/utils.py` — cosine kNN over the frozen embeddings = your retrieval index.
- `v_1/data/evaluation/corpora/orcc_corpus.parquet` — the dated pool to retrieve parallels from
  (`fragment_id, ruler, year, text`).
- P5 outputs (`sae/feature_dossier.py` JSON) for evidence; P3 `eval_robustness.py` for confound flags.

## What to read (papers)

- **Aeneas (Nature 2025)** — retrieval of historically grounded parallels as evidence for historians;
  evaluation showed historians used parallels as inquiry starting points. This is the design target.
- **Ithaca (Nature 2022)** — restoration + geographic + chronological attribution with interpretability
  and historian collaboration. Frames the output contract.

## What to build

### `v_1/src/chronorank/atlas.py`
```python
class ChronoAtlas:
    def __init__(self, head, embed_index, orcc_df, feature_dossiers=None, confound_flags=None): ...
    def predict(self, text_or_embedding) -> dict:
        """Returns {interval:(low,high), confidence, mu, sigma,
                    parallels:[{fragment_id, ruler, year, sim}],
                    evidence_later:[...], evidence_earlier:[...], confound_warnings:[...]}.
        Interval from mu±z·sigma; confidence bucketed from sigma; parallels = top-k cosine kNN
        among DATED fragments; evidence from P5 dossiers or sparse-head weights."""
    def report(self, ...) -> str:  # the human-readable block above
```

### Evaluation of the retrieval itself
`atlas_eval.py`: for held-out dated fragments, measure (a) parallel quality — are retrieved
neighbors temporally close (median |year_query − year_neighbor|)? (b) interval calibration —
does picp@80/90 from P0 hold on the held-out set? Produce a small table + a few example reports
on real ORCC fragments for the thesis appendix.

## Cluster / sbatch

CPU-only. One sbatch:
```bash
# v_1/src/chronorank/sbatch/P6_atlas.sbatch  (no --gres)
#SBATCH --cpus-per-task=16 --mem=64G --time=01:00:00
# ... standard preamble ...
python -u v_1/src/chronorank/atlas_eval.py --head v_1/src/chronorank/results/<best_run> \
       --out v_1/src/chronorank/results/atlas
# commit results/atlas/*.json + example_reports.md + push
```
Give Yarin the paste command. Run a local example on one ORCC fragment first and paste the report.

## Report back / success criterion

**PASS** when: (a) `ChronoAtlas.predict` returns the full structured output on real ORCC fragments;
(b) `atlas_eval.py` shows retrieved parallels are temporally coherent (neighbors materially closer in
year than random) and intervals are calibrated; (c) you produce 3–5 example reports a historian could
read. Per README §4, the value is the *evidence + parallels + honest uncertainty*, not the point year.
This is the final user-facing artifact of the thesis system.
