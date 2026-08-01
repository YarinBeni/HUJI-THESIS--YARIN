# Which numbers the figures show

Two orthogonal knobs decide every number in `figures/`. Both are now explicit, and
both have a committed CSV, so no figure depends on an unstated choice.

## 1. Protocol — how the fragment cells are cross-validated

`year` in the r8 fragment set takes only **17 distinct values across 8 rulers**, so the
label is very nearly the ruler's identity. That makes the splitter decisive:

| block      | splitter                 | a ruler is…                | table |
|------------|--------------------------|----------------------------|-------|
| `mc`       | StratifiedKFold-by-ruler | in train **and** test      | `TIDY_all_year_results__mc_LEAKY.csv` |
| `mc_group` | GroupKFold-by-ruler      | wholly train **or** test   | `TIDY_all_year_results__mc_group.csv` |
| `loro`     | leave-one-ruler-out      | wholly held out, pooled    | — |

`mc_group` mirrors `stress_tests/shared/mc_probe.py`, the engine behind the deck's
headline table (`stress_tests/results/csv/p1_year_mc.csv`) and the protocol stated on
deck slide 2. **It is the correct setting for anything compared against the thesis.**
Under `mc` a char-n-gram TF-IDF baseline reaches ρ = .707 on name-stripped Akkadian —
that number is the leak made visible, not a result.

    python build_tidy.py --mode mc_group          # canonical
    python build_tidy.py --mode mc                # reproduces the older, leaky figures

## 2. Read-out — which probe's ρ is reported

The thesis does **not** use one probe everywhere, and `p1_year_mc.csv` carries both
`ridge_spearman` and `pls_spearman_mean` side by side. Slide 4's headline numbers
(cuneiform-400M .391) are the **PLS** column.

| cell                       | thesis read-out |
|----------------------------|-----------------|
| A · salient entities (EN)  | ridge (Gurnee & Tegmark's own probe) |
| B · obscure entities (EN)  | PLS, k = 5 |
| B′/C · fragments           | PLS, best k ∈ {1,2,3,5} |

    python build_tidy.py --mode mc_group --readout deck

`--readout deck` keeps one row per cell and relabels the winner `ridge`, so the figure
scripts need no per-cell probe logic. PLS reports no R², so `r2` rows fall back to
ridge and are marked as such in the `readout` column.

### This choice changes the headline

At the fragment level, on `akk_maximal` under `mc_group`:

| arm                 | ridge ρ | PLS ρ |
|---------------------|---------|-------|
| cuneiform-400M      |  .329   | **.349** |
| AKK-300M            |  .288   |  .292 |
| random Qwen (ref.)  |  .280   |  .280 |
| **TF-IDF**          | **.330** (rank 1/15) | **−.016** (rank 15/15) |

Ridge on 256-dim TF-IDF-SVD still fits the surviving orthographic signal; the low-rank
PLS projection does not. The deck reports the PLS column, so the deck-facing figures
must too — otherwise the char-n-gram floor appears to win.

## Rendering

Every design script honours two env vars:

    TIDY_CSV   path to the table to read   (default: TIDY_all_year_results.csv)
    FIG_TAG    suffix for the output PNG   (default: none)

    # deck read-out
    TIDY_CSV=../TIDY_all_year_results__mc_group__deck.csv FIG_TAG=__deck python3 slopegraph.py
    # ridge everywhere
    python3 slopegraph.py

`heatmap.py` is the exception: it shows a RIDGE panel and a PLS panel side by side by
construction, so it reads the raw table only and has no `__deck` variant.

## Resolution and format

All figure scripts write through `lib/_save.py`, so output quality is decided in one
place instead of per-script. Every figure is emitted twice:

- **PNG at 300 dpi** — for the HTML deck. The designs now come out 2,700–4,500 px wide.
- **PDF, vector** — for the thesis. Text stays sharp at any zoom; embed this one in
  LaTeX, not the PNG.

Earlier renders were 120–220 dpi, which is legible on screen and falls apart when
projected or printed — an 8 pt footnote at 120 dpi is about 13 px tall.

    FIG_DPI=450 python3 slopegraph.py    # poster
    FIG_PDF=0   python3 slopegraph.py    # PNG only
