# Embedding panels — six-view metadata maps

Each PNG shows ONE 2-D embedding map of the 1,202 ORCC royal-inscription
fragments, colored six ways: **year BCE / ruler / period / sub-genre /
provenance / fragment-length**. Use them to see which metadata (if any) each
model's embedding organizes around.

## Layout

```
embedding_panels/<cleaning>/<reduction>/<model>.png
```

- **cleaning**: `maximal` (name-stripped Akkadian) · `engtier0` (English tier0,
  the valid translation)
- **reduction**: `tsne` · `pca` · `umap` (all unsupervised, at each model's
  best year layer) · `pls` (SUPERVISED PLS k=3, comp1 vs comp2 — fit on year,
  so year separation is partly baked in; read models against the tfidf/random
  panels in the same folder)
- **model**: the 9 activation models + `tfidf` (character-n-gram control) and
  `random` (untrained random-init control)

## How to read

Regenerate with `python v_1/src/stress_tests/eda/make_embedding_panels.py`.
Coordinate sources: viz/stress_coords.json (tsne/pca), viz/stress_umap_coords.json
(umap), viz/pls3d_coords.json (pls). Quantitative companion: ../../results/
SUMMARY_TABLES.md (Tables 2-4) and ../results/ (E6 cluster metrics).

Takeaway across all panels: every model's visible "year" structure is the
Neo-Babylonian-vs-Neo-Assyrian era split; within-era year is never resolved,
and where clusters exist they track provenance/sub-genre, not reign or year.
tfidf and random controls look no worse than the trained models.

## Which models to compare (the deck's top-3 + controls)

`../../results/csv/table1_best_models.csv` lists, per deck experiment
(year/geo/T12/P8/P9 x maximal/engtier0), the **top-3 models and the TF-IDF\* +
random\* controls** with their scores. Use it to pick which `<model>.png`
panels to line up: e.g. for *Year, Akk maximal* the top-3 are cunei-400m /
Qwen3-8B / Qwen3-1.7B — open those three plus `tfidf.png` and `random.png` in
the same `maximal/<reduction>/` folder to see whether the winners' maps look
any different from the controls' (they mostly do not). Columns: experiment,
cleaning, metric, rank1..3 model+value, tfidf_control, random_control.
Rendered narrative: ../../results/SUMMARY_TABLES.md (Table 1).
