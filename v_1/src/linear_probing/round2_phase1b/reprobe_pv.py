"""reprobe_pv.py — Phase 1b re-probing of prompted activations.

For each prompt variant (pv0/pv1/pv2/pv3) and each pulled layer (e.g. 0, 15, -1),
load the prompted-activation npz produced by `extract_prompted_acts.py`, align
its fragment_ids to the ORCC corpus, and run the Round-1 probe pipelines
(CLS for ruler + PLS for year, raw & log) ON ORCC-ONLY.

Comparable Round-1 baseline (Qwen, raw fragments, last-token pooling) had
Macro-F1 ~0.117 at layer 5 (cls_best_layers.json:qwen__tier0__last__ruler) and
Spearman ~0.08 / MAE ~131 (PLS year-raw).

Rationale for ORCC-only:
Round-1 scripts (05_compute_cls.py, 05_compute_pls.py) concatenate SEAL + ORCC
arrays then index labeled-ORCC rows out for the probe. The probe math
*only ever sees the labeled-ORCC rows*; SEAL is just along for the ride so
projections can be co-emitted. We have no SEAL prompted activations and no
need for projections here, so we import the underlying `fit_cls_cv` /
`fit_pls_groupkfold` utilities directly and feed them ORCC-only data.

This is option (a) from the spec: import the script's core functions,
point them at the new path explicitly.

See:
  v_1/src/linear_probing/cls_utils.py:18    fit_cls_cv()
  v_1/src/linear_probing/pls_utils.py:28    l2_normalize()
  v_1/src/linear_probing/pls_utils.py:77    fit_pls_groupkfold()
  v_1/src/linear_probing/05_compute_cls.py:124-141  min-count + label filter pattern
  v_1/src/linear_probing/05_compute_pls.py:194-206  labeled mask + group construction

CLI
---
  python reprobe_pv.py \
      --acts_root v_1/src/linear_probing/results/orcc_round2_phase1b/prompted_activations \
      --corpus    v_1/data/evaluation/corpora/orcc_corpus.parquet \
      --out_dir   v_1/src/linear_probing/results/orcc_round2_phase1b/reprobing \
      [--variants pv0,pv1,pv2,pv3] \
      [--layers 0,15,-1]

Layer convention matches extract_prompted_acts.py: -1 means the FINAL hidden
state. The script discovers the actual stored layer index from npz file name
patterns L{NN}.npz / layer_{NN}.npz; -1 is resolved by looking for the highest
numeric layer file under each variant dir (assumed = last layer extracted).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
_LP_DIR = _THIS_DIR.parent  # v_1/src/linear_probing/
_REPO_ROOT = _THIS_FILE.parents[4]

# Import Round-1 probe utilities directly (option a).
sys.path.insert(0, str(_LP_DIR))
from cls_utils import fit_cls_cv               # noqa: E402
from pls_utils import (                         # noqa: E402
    l2_normalize,
    fit_pls_groupkfold,
)


# ---------------------------------------------------------------------------
# Round-1 baseline (last-token, tier0). Hard-coded snapshot for the summary;
# source files referenced for traceability.
# ---------------------------------------------------------------------------
ROUND1_BASELINE_FILES = {
    "cls": str(_LP_DIR / "results" / "orcc__probe_cls" / "cls_best_layers.json"),
    "pls": str(_LP_DIR / "results" / "orcc__probe_pls" / "pls_best_layers.json"),
}


def load_round1_baseline() -> dict[str, Any]:
    """Surface qwen tier0 best-layer metrics for BOTH `last` (apples-to-apples
    for Phase 1b's last-token pooling) AND `mean` (the headline Round-1
    pooling) for direct comparison.
    """
    out: dict[str, Any] = {"source_files": ROUND1_BASELINE_FILES}
    try:
        with open(ROUND1_BASELINE_FILES["cls"]) as f:
            cls_best = json.load(f)
        out["qwen_tier0_last_ruler_best"] = cls_best.get(
            "qwen__tier0__last__ruler", {}
        )
        out["qwen_tier0_mean_ruler_best"] = cls_best.get(
            "qwen__tier0__mean__ruler", {}
        )
    except Exception as e:
        out["cls_load_error"] = str(e)
    try:
        with open(ROUND1_BASELINE_FILES["pls"]) as f:
            pls_best = json.load(f)
        out["qwen_tier0_last_year_raw_best"] = pls_best.get(
            "qwen__tier0__last__year-raw", {}
        )
        out["qwen_tier0_last_year_log_best"] = pls_best.get(
            "qwen__tier0__last__year-log", {}
        )
        out["qwen_tier0_mean_year_raw_best"] = pls_best.get(
            "qwen__tier0__mean__year-raw", {}
        )
        out["qwen_tier0_mean_year_log_best"] = pls_best.get(
            "qwen__tier0__mean__year-log", {}
        )
    except Exception as e:
        out["pls_load_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def _resolve_layer_token(token: str, variant_dir: Path) -> int | None:
    """Convert a layer arg (e.g. '0', '15', '-1') into a stored npz index.

    Stored files are L{NN}.npz / layer_{NN}.npz with NN = zero-padded int.
    -1 means "the highest numeric layer found in this variant dir".
    Returns None if cannot resolve.
    """
    token = token.strip()
    try:
        ti = int(token)
    except ValueError:
        return None
    if ti >= 0:
        return ti
    # Negative: look at available files.
    if not variant_dir.exists():
        return None
    candidates: list[int] = []
    pat = re.compile(r"(?:L|layer_)(\d{2})\.npz$")
    for p in variant_dir.iterdir():
        m = pat.search(p.name)
        if m:
            candidates.append(int(m.group(1)))
    if not candidates:
        return None
    if ti == -1:
        return max(candidates)
    # For more negative offsets we cannot know n_layers; fall back to error.
    return None


def load_prompted_acts(
    variant_dir: Path, layer_idx: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load activations for (variant_dir, layer_idx).

    Tries Round-1-compatible `layer_{NN}.npz` (key=`activations`) first; falls
    back to rich `L{NN}.npz` (key=`acts` + `fragment_ids`).

    Returns (acts, fragment_ids_or_None). fragment_ids is None when only the
    Round-1-compatible npz exists — the caller must then trust corpus order.
    """
    LL = f"{layer_idx:02d}"
    r1_path = variant_dir / f"layer_{LL}.npz"
    rich_path = variant_dir / f"L{LL}.npz"

    # Prefer rich (has fragment_ids); fall back to r1.
    # allow_pickle=True is required because extract_prompted_acts.py stores
    # fragment_ids / rulers as object-dtype arrays from `np.asarray(list_of_str)`.
    if rich_path.exists():
        d = np.load(rich_path, allow_pickle=True)
        acts = d["acts"].astype(np.float32)
        fids = (
            np.asarray(d["fragment_ids"]).astype(str)
            if "fragment_ids" in d.files else None
        )
        return acts, fids
    if r1_path.exists():
        d = np.load(r1_path, allow_pickle=False)
        acts = d["activations"].astype(np.float32)
        return acts, None
    raise FileNotFoundError(
        f"No npz for layer {layer_idx} under {variant_dir} "
        f"(checked {rich_path.name} and {r1_path.name})"
    )


# ---------------------------------------------------------------------------
# Alignment to corpus
# ---------------------------------------------------------------------------

def align_to_corpus(
    acts: np.ndarray,
    fragment_ids: np.ndarray | None,
    orcc_df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (X, sub_df) with rows aligned to orcc_df fragment_id order.

    If fragment_ids are supplied, reorder acts to corpus order. If not, assume
    acts already follow corpus order (Round-1 contract).
    """
    if fragment_ids is None:
        if acts.shape[0] != len(orcc_df):
            raise ValueError(
                f"acts has {acts.shape[0]} rows but corpus has {len(orcc_df)} "
                "and no fragment_ids stored — cannot align."
            )
        return acts, orcc_df.reset_index(drop=True)

    corpus_ids = orcc_df["fragment_id"].astype(str).values
    id_to_corpus_idx = {fid: i for i, fid in enumerate(corpus_ids)}
    rows: list[int] = []
    keep_acts_idx: list[int] = []
    for j, fid in enumerate(fragment_ids):
        ci = id_to_corpus_idx.get(str(fid))
        if ci is None:
            continue
        rows.append(ci)
        keep_acts_idx.append(j)
    if not rows:
        raise RuntimeError("No fragment_ids matched the corpus.")
    sub_df = orcc_df.iloc[rows].reset_index(drop=True)
    X = acts[keep_acts_idx]
    return X, sub_df


# ---------------------------------------------------------------------------
# Probe runners
# ---------------------------------------------------------------------------

def run_cls_ruler(
    X_norm: np.ndarray,
    sub_df: pd.DataFrame,
    min_count: int = 5,
    C: float = 1.0,
) -> dict[str, Any]:
    """Run the Round-1 CLS ruler probe (StratifiedKFold(5), logreg).

    Matches the filtering in 05_compute_cls.py:124-141 (drop ruler classes
    with fewer than `min_count` fragments).
    """
    raw = sub_df["ruler"].astype(str).values
    counts = pd.Series(raw).value_counts()
    keep_classes = counts[counts >= min_count].index
    mask = np.isin(raw, keep_classes)
    n_dropped = int((~mask).sum())
    y = raw[mask]
    X = X_norm[mask]

    m = fit_cls_cv(X, y, cv_strategy="stratified", n_splits=5, C=C)
    return {
        "task": "ruler",
        "n_fragments": int(mask.sum()),
        "n_classes": int(keep_classes.shape[0]),
        "n_dropped": n_dropped,
        "min_count": min_count,
        **m,
    }


def run_pls_year(
    X_norm: np.ndarray,
    sub_df: pd.DataFrame,
    n_components_list: tuple[int, ...] = (1, 2, 3, 5),
) -> dict[str, Any]:
    """Run Round-1 PLS year probe (raw + log), GroupKFold by ruler."""
    labeled_mask = ~sub_df["year"].isna()
    sub_lab = sub_df.loc[labeled_mask].reset_index(drop=True)
    X_lab = X_norm[labeled_mask.values]

    y_raw = sub_lab["year"].astype(float).values
    y_log = np.log(y_raw)
    groups = sub_lab["ruler"].astype(str).values

    n_groups = int(len(np.unique(groups)))

    result: dict[str, Any] = {
        "task": "year",
        "n_labeled": int(len(sub_lab)),
        "n_groups": n_groups,
    }

    for yt, y in (("raw", y_raw), ("log", y_log)):
        metrics_per_k = {}
        for k in n_components_list:
            try:
                metrics_per_k[str(k)] = fit_pls_groupkfold(X_lab, y, groups, k)
            except Exception as e:
                # PLS NIPALS can divide by zero when X is rank-deficient (e.g.,
                # L0 embeddings where the pooled token is identical across rows).
                # Skip with NaN metrics so downstream summary still renders.
                print(f"    [pls-skip] k={k} year-{yt}: {type(e).__name__}: {e}", flush=True)
                metrics_per_k[str(k)] = {
                    "spearman_mean": float("nan"), "spearman_std": float("nan"),
                    "mae_mean": float("nan"), "mae_std": float("nan"),
                    "r2_mean": float("nan"), "r2_std": float("nan"),
                    "skipped": True, "error": f"{type(e).__name__}: {e}",
                }

        def _best_by(metric: str):
            valid = [k for k in n_components_list
                     if not (isinstance(metrics_per_k[str(k)].get(metric), float)
                             and np.isnan(metrics_per_k[str(k)][metric]))]
            if not valid:
                return n_components_list[0]
            return max(valid, key=lambda k: metrics_per_k[str(k)][metric])

        best_sp = _best_by("spearman_mean")
        best_r2 = _best_by("r2_mean")
        result[f"year_{yt}"] = {
            "metrics_per_k": metrics_per_k,
            "best_k_by_spearman": int(best_sp),
            "best_k_by_r2": int(best_r2),
            "spearman_at_best_k": metrics_per_k[str(best_sp)]["spearman_mean"],
            "mae_at_best_k": metrics_per_k[str(best_sp)]["mae_mean"],
            "r2_at_best_k": metrics_per_k[str(best_r2)]["r2_mean"],
        }
    return result


# ---------------------------------------------------------------------------
# Per-(variant,layer) driver
# ---------------------------------------------------------------------------

def reprobe_one(
    variant: str,
    layer_idx: int,
    variant_dir: Path,
    orcc_df: pd.DataFrame,
    out_dir: Path,
    min_count: int,
    n_components_list: tuple[int, ...],
) -> dict[str, Any]:
    print(f"\n=== variant={variant}  layer={layer_idx} ===", flush=True)
    t0 = time.time()
    acts, fids = load_prompted_acts(variant_dir, layer_idx)
    print(f"  loaded acts.shape={acts.shape}  fragment_ids={'yes' if fids is not None else 'no'}",
          flush=True)

    X_aligned, sub_df = align_to_corpus(acts, fids, orcc_df)
    # Sanity: surface NaN/Inf in loaded activations (would crash SVD downstream).
    n_nan_rows = int(np.isnan(X_aligned).any(axis=1).sum())
    n_inf_rows = int(np.isinf(X_aligned).any(axis=1).sum())
    if n_nan_rows or n_inf_rows:
        print(f"  [warn] {n_nan_rows} NaN rows + {n_inf_rows} Inf rows in X_aligned; "
              "dropping before normalize", flush=True)
        bad = np.isnan(X_aligned).any(axis=1) | np.isinf(X_aligned).any(axis=1)
        X_aligned = X_aligned[~bad]
        sub_df = sub_df.loc[~bad].reset_index(drop=True)
    X_norm = l2_normalize(X_aligned)
    # Replace any residual non-finite values from divide-by-zero norms with 0.
    if not np.isfinite(X_norm).all():
        bad_after = (~np.isfinite(X_norm)).any(axis=1)
        print(f"  [warn] {int(bad_after.sum())} rows became non-finite after l2_normalize; "
              "zeroing those entries", flush=True)
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure contiguous float32 — scipy LAPACK wrappers can reject non-contiguous arrays
    # with "illegal value in Nth argument" errors otherwise.
    X_norm = np.ascontiguousarray(X_norm, dtype=np.float32)
    print(f"  aligned X.shape={X_norm.shape}  sub_df.rows={len(sub_df)}", flush=True)

    # Round-1 evaluates only on the labeled subset (year not null); match.
    labeled_mask = ~sub_df["year"].isna()
    X_labeled = X_norm[labeled_mask.values]
    sub_labeled = sub_df.loc[labeled_mask].reset_index(drop=True)
    print(f"  labeled subset: {len(sub_labeled)} fragments "
          f"(of {len(sub_df)})", flush=True)

    cls_res = run_cls_ruler(X_labeled, sub_labeled, min_count=min_count)
    print(f"  [cls] macro_f1={cls_res['macro_f1_mean']:.3f}  "
          f"acc={cls_res['accuracy_mean']:.3f}  n_classes={cls_res['n_classes']}",
          flush=True)

    # PLS year probe also runs on labeled rows (it filters internally too,
    # but feeding pre-filtered is cleaner and matches Round-1 contract).
    pls_res = run_pls_year(X_labeled, sub_labeled, n_components_list=n_components_list)
    yr_raw = pls_res["year_raw"]
    yr_log = pls_res["year_log"]
    print(f"  [pls year-raw] sp={yr_raw['spearman_at_best_k']:.3f}  "
          f"mae={yr_raw['mae_at_best_k']:.2f}", flush=True)
    print(f"  [pls year-log] sp={yr_log['spearman_at_best_k']:.3f}  "
          f"mae={yr_log['mae_at_best_k']:.4f}", flush=True)

    LL = f"{layer_idx:02d}"
    cls_path = out_dir / f"{variant}__L{LL}__cls.json"
    pls_path = out_dir / f"{variant}__L{LL}__pls.json"
    cls_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cls_path, "w") as f:
        json.dump({"variant": variant, "layer": layer_idx, **cls_res}, f, indent=2)
    with open(pls_path, "w") as f:
        json.dump({"variant": variant, "layer": layer_idx, **pls_res}, f, indent=2)
    print(f"  saved -> {cls_path.name}, {pls_path.name}  "
          f"({time.time() - t0:.1f}s)", flush=True)

    return {
        "variant": variant,
        "layer": layer_idx,
        "ruler_macro_f1": float(cls_res["macro_f1_mean"]),
        "ruler_accuracy": float(cls_res["accuracy_mean"]),
        "ruler_n_classes": int(cls_res["n_classes"]),
        "year_raw_mae": float(yr_raw["mae_at_best_k"]),
        "year_raw_spearman": float(yr_raw["spearman_at_best_k"]),
        "year_log_mae": float(yr_log["mae_at_best_k"]),
        "year_log_spearman": float(yr_log["spearman_at_best_k"]),
    }


# ---------------------------------------------------------------------------
# Summary + markdown report
# ---------------------------------------------------------------------------

def _verdict_label(beats_last: bool, beats_mean: bool) -> str:
    if beats_last and beats_mean:
        return "BEATS BOTH"
    if beats_last and not beats_mean:
        return "BEATS LAST"
    if beats_mean and not beats_last:
        return "BEATS MEAN"
    return "FAILS BOTH"


def build_summary(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    # Last-pooling baseline (apples-to-apples vs Phase 1b which pools at last token).
    r1_cls_last = baseline.get("qwen_tier0_last_ruler_best", {}) or {}
    r1_last_macro_f1 = float(r1_cls_last.get("best_layer_macro_f1", float("nan")))
    r1_last_layer = r1_cls_last.get("best_layer")
    r1_pls_last_raw = baseline.get("qwen_tier0_last_year_raw_best", {}) or {}
    r1_pls_last_log = baseline.get("qwen_tier0_last_year_log_best", {}) or {}

    # Mean-pooling baseline (the headline 0.117 Macro-F1 we've been quoting).
    r1_cls_mean = baseline.get("qwen_tier0_mean_ruler_best", {}) or {}
    r1_mean_macro_f1 = float(r1_cls_mean.get("best_layer_macro_f1", float("nan")))
    r1_mean_layer = r1_cls_mean.get("best_layer")
    r1_pls_mean_raw = baseline.get("qwen_tier0_mean_year_raw_best", {}) or {}
    r1_pls_mean_log = baseline.get("qwen_tier0_mean_year_log_best", {}) or {}

    # Stamp every per-(variant, layer) row with both baselines + both deltas + verdict.
    for r in rows:
        f1 = float(r["ruler_macro_f1"])
        r["r1_qwen_last_macro_f1"] = r1_last_macro_f1
        r["r1_qwen_last_best_layer"] = r1_last_layer
        r["r1_qwen_mean_macro_f1"] = r1_mean_macro_f1
        r["r1_qwen_mean_best_layer"] = r1_mean_layer
        r["delta_vs_r1_last"] = (
            f1 - r1_last_macro_f1 if not np.isnan(r1_last_macro_f1) else None
        )
        r["delta_vs_r1_mean"] = (
            f1 - r1_mean_macro_f1 if not np.isnan(r1_mean_macro_f1) else None
        )
        beats_last = (not np.isnan(r1_last_macro_f1)) and f1 > r1_last_macro_f1
        beats_mean = (not np.isnan(r1_mean_macro_f1)) and f1 > r1_mean_macro_f1
        r["beats_r1_last"] = bool(beats_last)
        r["beats_r1_mean"] = bool(beats_mean)
        r["verdict"] = _verdict_label(beats_last, beats_mean)
        # PLS year deltas (raw + log), against BOTH baselines.
        for yt, base_last, base_mean in (
            ("raw", r1_pls_last_raw, r1_pls_mean_raw),
            ("log", r1_pls_last_log, r1_pls_mean_log),
        ):
            sp = float(r[f"year_{yt}_spearman"])
            mae = float(r[f"year_{yt}_mae"])
            r[f"r1_qwen_last_year_{yt}_spearman"] = base_last.get("spearman_mean")
            r[f"r1_qwen_last_year_{yt}_mae"] = base_last.get("mae_mean")
            r[f"r1_qwen_mean_year_{yt}_spearman"] = base_mean.get("spearman_mean")
            r[f"r1_qwen_mean_year_{yt}_mae"] = base_mean.get("mae_mean")
            if base_last.get("spearman_mean") is not None:
                r[f"delta_vs_r1_last_year_{yt}_spearman"] = sp - float(
                    base_last["spearman_mean"]
                )
            if base_mean.get("spearman_mean") is not None:
                r[f"delta_vs_r1_mean_year_{yt}_spearman"] = sp - float(
                    base_mean["spearman_mean"]
                )
            if base_last.get("mae_mean") is not None:
                # MAE lower=better, so delta = baseline - candidate (positive => improves).
                r[f"delta_vs_r1_last_year_{yt}_mae"] = float(base_last["mae_mean"]) - mae
            if base_mean.get("mae_mean") is not None:
                r[f"delta_vs_r1_mean_year_{yt}_mae"] = float(base_mean["mae_mean"]) - mae

    per_variant: dict[str, dict[str, Any]] = {}
    for r in rows:
        v = r["variant"]
        per_variant.setdefault(v, {"layers": []})
        per_variant[v]["layers"].append(r)

    for v, info in per_variant.items():
        layers = info["layers"]
        best = max(layers, key=lambda x: x["ruler_macro_f1"])
        info["best_layer_for_ruler"] = best["layer"]
        info["best_layer_ruler_macro_f1"] = best["ruler_macro_f1"]
        info["beats_r1_last"] = bool(best.get("beats_r1_last"))
        info["beats_r1_mean"] = bool(best.get("beats_r1_mean"))
        info["verdict"] = best.get("verdict", "FAILS BOTH")

    summary = {
        "round1_baseline": {
            "source_files": baseline.get("source_files"),
            # Last-pooling (apples-to-apples).
            "qwen_tier0_last_ruler_macro_f1": r1_last_macro_f1,
            "qwen_tier0_last_ruler_best_layer": r1_last_layer,
            "qwen_tier0_last_year_raw_mae": r1_pls_last_raw.get("mae_mean"),
            "qwen_tier0_last_year_raw_spearman": r1_pls_last_raw.get("spearman_mean"),
            "qwen_tier0_last_year_log_mae": r1_pls_last_log.get("mae_mean"),
            "qwen_tier0_last_year_log_spearman": r1_pls_last_log.get("spearman_mean"),
            # Mean-pooling (headline Round-1).
            "qwen_tier0_mean_ruler_macro_f1": r1_mean_macro_f1,
            "qwen_tier0_mean_ruler_best_layer": r1_mean_layer,
            "qwen_tier0_mean_year_raw_mae": r1_pls_mean_raw.get("mae_mean"),
            "qwen_tier0_mean_year_raw_spearman": r1_pls_mean_raw.get("spearman_mean"),
            "qwen_tier0_mean_year_log_mae": r1_pls_mean_log.get("mae_mean"),
            "qwen_tier0_mean_year_log_spearman": r1_pls_mean_log.get("spearman_mean"),
        },
        "rows": rows,
        "per_variant": per_variant,
    }
    summary_path = out_dir / "phase1b_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] -> {summary_path}", flush=True)
    return summary


def _fmt(x: Any, spec: str = ".3f") -> str:
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "n/a"
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def _fmt_delta(x: Any, spec: str = "+.3f") -> str:
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "n/a"
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def build_report_md(summary: dict[str, Any], out_dir: Path) -> None:
    r1 = summary["round1_baseline"]
    r1_last_f1 = r1.get("qwen_tier0_last_ruler_macro_f1")
    r1_mean_f1 = r1.get("qwen_tier0_mean_ruler_macro_f1")

    lines = [
        "# Phase 1b — Prompted-Activation Re-probing Report",
        "",
        "Probes the SAME pipelines (CLS ruler, PLS year raw/log) used in Round 1",
        "on activations extracted with Q+A prompt wrapping (4 variants x 3 layers).",
        "Phase 1b pools at the last token of the fragment span, so the apples-to-",
        "apples Round-1 reference is `last`. We also show `mean` (the headline",
        "Round-1 number, ~0.117 Macro-F1) for context.",
        "",
        "## Round-1 baselines (Qwen, raw fragments, tier0)",
        "",
        f"- ruler Macro-F1 **last** (best layer L{r1.get('qwen_tier0_last_ruler_best_layer')}): "
        f"**{_fmt(r1_last_f1)}**",
        f"- ruler Macro-F1 **mean** (best layer L{r1.get('qwen_tier0_mean_ruler_best_layer')}): "
        f"**{_fmt(r1_mean_f1)}**",
        f"- year-raw (last) MAE: {_fmt(r1.get('qwen_tier0_last_year_raw_mae'), '.2f')}  "
        f"Spearman: {_fmt(r1.get('qwen_tier0_last_year_raw_spearman'))}",
        f"- year-raw (mean) MAE: {_fmt(r1.get('qwen_tier0_mean_year_raw_mae'), '.2f')}  "
        f"Spearman: {_fmt(r1.get('qwen_tier0_mean_year_raw_spearman'))}",
        f"- year-log (last) MAE: {_fmt(r1.get('qwen_tier0_last_year_log_mae'), '.4f')}  "
        f"Spearman: {_fmt(r1.get('qwen_tier0_last_year_log_spearman'))}",
        f"- year-log (mean) MAE: {_fmt(r1.get('qwen_tier0_mean_year_log_mae'), '.4f')}  "
        f"Spearman: {_fmt(r1.get('qwen_tier0_mean_year_log_spearman'))}",
        "",
        "## Phase 1b ruler results (vs both Round-1 baselines)",
        "",
        "| variant | layer | macro_f1 | R1 last (Δ) | R1 mean (Δ) | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        last_str = f"{_fmt(r.get('r1_qwen_last_macro_f1'))} ({_fmt_delta(r.get('delta_vs_r1_last'))})"
        mean_str = f"{_fmt(r.get('r1_qwen_mean_macro_f1'))} ({_fmt_delta(r.get('delta_vs_r1_mean'))})"
        lines.append(
            f"| {r['variant']} | {r['layer']} | "
            f"{_fmt(r['ruler_macro_f1'])} | {last_str} | {mean_str} | "
            f"{r.get('verdict', 'FAILS BOTH')} |"
        )
    lines.append("")
    lines.append("## Phase 1b PLS year results (full numbers)")
    lines.append("")
    lines.append(
        "| variant | layer | ruler acc | year-raw MAE | year-raw sp | year-log MAE | year-log sp |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in summary["rows"]:
        lines.append(
            f"| {r['variant']} | {r['layer']} | "
            f"{_fmt(r['ruler_accuracy'])} | "
            f"{_fmt(r['year_raw_mae'], '.2f')} | {_fmt(r['year_raw_spearman'])} | "
            f"{_fmt(r['year_log_mae'], '.4f')} | {_fmt(r['year_log_spearman'])} |"
        )
    lines.append("")
    lines.append("## Per-variant verdict (at best ruler layer)")
    lines.append("")
    for v, info in summary["per_variant"].items():
        lines.append(
            f"- **{v}**: best layer L{info['best_layer_for_ruler']} "
            f"Macro-F1={_fmt(info['best_layer_ruler_macro_f1'])} "
            f"-> {info.get('verdict', 'FAILS BOTH')}"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    n_total = len(summary["per_variant"])
    n_beat_last = sum(
        1 for v in summary["per_variant"].values() if v.get("beats_r1_last")
    )
    n_beat_mean = sum(
        1 for v in summary["per_variant"].values() if v.get("beats_r1_mean")
    )
    n_beat_both = sum(
        1
        for v in summary["per_variant"].values()
        if v.get("beats_r1_last") and v.get("beats_r1_mean")
    )
    if n_beat_both == 0 and n_beat_last == 0 and n_beat_mean == 0:
        lines.append(
            f"None of {n_total} prompted variants beat EITHER Round-1 baseline on ruler "
            "Macro-F1 (neither apples-to-apples `last` nor headline `mean`). "
            "Prompt framing alone does NOT rescue Qwen's representations for ruler "
            "classification — the diagnostic for Phase 1b is negative and we should "
            "move on to scale (Phase 2) or tokenization (Phase 3)."
        )
    else:
        lines.append(
            f"{n_beat_last}/{n_total} variants beat Round-1 `last`, "
            f"{n_beat_mean}/{n_total} beat Round-1 `mean`, "
            f"{n_beat_both}/{n_total} beat BOTH. "
            "Inspect which variants moved the needle (pv0/pv1/pv2/pv3) and which "
            "layer they peak at to decide next experiments."
        )

    report_path = out_dir / "phase1b_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] -> {report_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1b re-probing wrapper")
    p.add_argument(
        "--acts_root",
        required=True,
        help="Dir containing {variant}/L{NN}.npz or layer_{NN}.npz (the "
        "prompted_activations/ root produced by extract_prompted_acts.py).",
    )
    p.add_argument(
        "--corpus",
        default=str(_REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"),
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--variants",
        default="pv0,pv1,pv2,pv3",
        help="Comma-separated variant names.",
    )
    p.add_argument(
        "--layers",
        default="0,15,-1",
        help="Comma-separated layer indices. -1 means 'highest layer present'.",
    )
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument(
        "--n_components",
        default="1,2,3,5",
        help="Comma-separated PLS n_components values.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    acts_root = Path(args.acts_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    layer_tokens = [t.strip() for t in args.layers.split(",") if t.strip()]
    n_components_list = tuple(
        int(x.strip()) for x in args.n_components.split(",") if x.strip()
    )

    orcc_df = pd.read_parquet(args.corpus)
    print(f"[corpus] {len(orcc_df)} rows; "
          f"labeled (year not null) = {orcc_df['year'].notna().sum()}", flush=True)

    baseline = load_round1_baseline()

    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = acts_root / variant
        if not variant_dir.exists():
            print(f"  [skip] variant dir missing: {variant_dir}", flush=True)
            continue
        for token in layer_tokens:
            layer_idx = _resolve_layer_token(token, variant_dir)
            if layer_idx is None:
                print(f"  [skip] cannot resolve layer={token} for {variant}", flush=True)
                continue
            try:
                row = reprobe_one(
                    variant=variant,
                    layer_idx=layer_idx,
                    variant_dir=variant_dir,
                    orcc_df=orcc_df,
                    out_dir=out_dir,
                    min_count=args.min_count,
                    n_components_list=n_components_list,
                )
                rows.append(row)
            except FileNotFoundError as e:
                print(f"  [skip] {variant} L{layer_idx}: {e}", flush=True)
            except Exception as e:
                print(f"  [error] {variant} L{layer_idx}: {e}", flush=True)
                raise

    if not rows:
        print("\nNo (variant, layer) pairs were probed. Nothing to summarize.")
        return

    summary = build_summary(rows, baseline, out_dir)
    build_report_md(summary, out_dir)


if __name__ == "__main__":
    main()
