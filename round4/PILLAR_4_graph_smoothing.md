# Pillar 4 — Graph smoothing / seriation over the unlabeled corpus  ⛔ PARKED

> **STATUS: PARKED (Yarin's call, 2026-06-13). Do not implement now.**
> Reason: the 2M-word unlabeled corpus has **no chronological labels at all** — only the ~1.2k
> royal inscriptions are dated. Unsupervised seriation/graph-smoothing over unlabeled embeddings
> is roughly what the frozen-model geodesic work already tried, and it did **not** do well, so
> there is little reason to expect the unlabeled pool to yield a clean timeline. The idea is still
> interesting as a *future* experiment **restricted to the labeled royal inscriptions** (where we
> can actually check the order), but it is out of scope for Round 4. Skip to P0→P2→P3→P6 and P1.
>
> The original design is preserved below for whenever we revisit it on the labeled set.

---

> **Agent brief (FROZEN — only if un-parked).** The principled, non-generative version of Yarin's
> "diffusion / path" intuition: a smooth chronological coordinate anchored by dated texts and
> regularized by neighbors. Read `README.md` first. **Requires P0.** Needs **1 GPU** to embed once.

## Goal

Two things:
1. **Seriation baseline (answer first, cheap):** does a 1-D chronological order *already exist*
   as a global manifold in the unlabeled corpus, before any training? Build a kNN graph over
   frozen embeddings of all fragments, run 1-D Isomap/spectral seriation, align to the 1.2k dated
   anchors, evaluate under maximal-balanced. (This is CLSS-style.)
2. **Graph-smoothness training term** for P3's head: `L_graph = Σ w_ij (s_i − s_j)²` over nearby
   unlabeled texts → "very-similar texts should not jump on the timeline." This is the plan's **Stage 4**.

## Dependencies

**P0:** `labels.py`, `eval_ordinal.py`. **Reuses `v_1/src/geodesic/utils.py` heavily** (the graph
machinery already exists). Feeds an optional `--graph-weight` term back into P3's `train.py`.

## What to read (repo)

- `v_1/src/geodesic/utils.py` — `build_knn_graph`, `geodesic_dist`, `isomap_1d`, `earliest_bin_coord`,
  `sign_flip_coord`, `pairwise_order_acc`, `pca_l2`. **The seriation baseline is ~50 lines on top of
  this.** Phase A/B already did 1-D Isomap on labeled ORCC — extend to the unlabeled pool.
- `v_1/src/geodesic/results/RESULTS_BY_TEST.md` and `geodesic_layer_scoreboard*.json` — the existing
  geodesic numbers you must be consistent with.
- `README.md` §3 — `unified_corpus.parquet` is **word-level (2.45M rows)**; you must reconstruct
  fragment-level text first (sort by `line_num, word_idx`, space-join `value_clean` per `fragment_id`
  — the same reconstruction `corpus/` uses). Filter to Akkadian.

## What to read (papers)

- **CLSS (Dai et al., NeurIPS 2023)** — spectral seriation extracts ordinal rankings from unlabeled
  samples as an extra training signal for deep regression. This is the direct precedent; it validates
  "unlabeled → ordinal structure → semi-supervised regression."
- **Isomap (Tenenbaum)** — for the 1-D manifold coordinate (already used in geodesic).
- **Snorkel (weak supervision)** — *optional*, only if you add weak date priors (ruler/eponym
  mentions) as graph anchors; treat as probabilistic, never hard labels.

## What to build

### `v_1/src/chronorank/graph.py`
```python
def reconstruct_fragments(unified_df, language="akkadian") -> pd.DataFrame:
    """word-rows -> fragment-level text. Sort (line_num, word_idx), join value_clean per fragment_id."""
def seriation_1d(Z, k) -> np.ndarray:
    """kNN graph -> 1-D Isomap/spectral coord. Thin wrapper over geodesic.utils."""
def align_to_anchors(coord, anchor_idx, anchor_years) -> np.ndarray:
    """orient + scale the unsupervised coord to BCE using ONLY train-split dated anchors."""
def graph_smoothness_term(s, knn_adj):  # Σ w_ij (s_i - s_j)^2  — used as a loss add-on in P3
```

### Seriation baseline script (the cheap answer)
`graph_seriation_eval.py`: embed unlabeled+labeled with Thalesian (best layer) and TF-IDF; build
graph; seriate; align to train anchors; evaluate held-out rulers with `eval_ordinal.full_report`.
**Decision output:** "Is chronology a global manifold in the unlabeled corpus, or only a supervised
artifact in the labeled 1.2k?" If seriation beats PLS → strong standalone chapter. If it fails →
you've shown supervised ordinal shaping is necessary (also a real finding).

### Training hook
Expose `graph_smoothness_term` so P3's `train.py --graph-weight λ5` can add it (build the kNN adj
once, cache to disk, pass to the loss). Keep λ5 small; use only high-confidence local neighborhoods.

## Cluster / sbatch

Two sbatch files in `v_1/src/chronorank/sbatch/`:
- `P4a_embed_unlabeled.sbatch` — `--gres=gpu:1`, `--time=08:00:00`, `--mem=128G`. Reconstruct
  fragments from `unified_corpus.parquet`, extract Thalesian embeddings (reuse
  `03_extract_seal_activations.py`), cache to `v_1/src/chronorank/results/unlabeled_embed/`
  (gitignore the large arrays; commit only a manifest). **One-time.**
- `P4b_seriation_eval.sbatch` — **CPU only**, builds graph + seriation + eval, commits the JSON.

Give Yarin both paste commands; note P4a must finish before P4b.

## Report back / success criterion

**PASS** when `graph_seriation_eval.py` produces a signed answer to the manifold question (with
numbers vs the PLS baseline), and `graph_smoothness_term` is wired into P3's trainer with a
documented λ5. Either outcome (manifold exists / doesn't) is a PASS — it's a diagnostic. Report
the seriation Spearman vs PLS and whether adding the graph term to P3 changed robustness.
