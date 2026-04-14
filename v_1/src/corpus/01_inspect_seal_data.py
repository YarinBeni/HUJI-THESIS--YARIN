#!/usr/bin/env python
"""01_inspect_seal_data.py — Phase 0 inspection of round-4 SEAL/DLL/LBPL CSVs.

Reads:  v_1/data/raw/chungrong/seal_round4/{seal,dll,lbpl}.csv
Writes: v_1/data/raw/chungrong/seal_round4/inspection_report.{md,json}

Implements every check listed in Section 15 of
v_1/justification/seal_round4_pipeline_plan.md.

Idempotent: re-running overwrites both report files. The JSON output is the
data contract for downstream Phase A/B/C/D scripts; if the source CSVs change,
re-running this script + committing the new JSON makes the change visible in
git diff.

Dependencies: stdlib + pandas + numpy only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUND_DIR = REPO_ROOT / "v_1" / "data" / "raw" / "chungrong" / "seal_round4"
CSV_FILES = OrderedDict(
    [
        ("seal", ROUND_DIR / "seal.csv"),
        ("dll", ROUND_DIR / "dll.csv"),
        ("lbpl", ROUND_DIR / "lbpl.csv"),
    ]
)
REPORT_JSON = ROUND_DIR / "inspection_report.json"
REPORT_MD = ROUND_DIR / "inspection_report.md"

# Columns we expect (per Section 3 of the plan).
EXPECTED_COLUMNS = [
    "fragment_id",
    "fragment_line_num",
    "index_in_line",
    "word_language",
    "domain",
    "period",
    "genre",
    "sub_genre",
    "provenance",
    "sub_provenance",
    "place_discovery",
    "place_composition",
    "value",
    "clean_value",
    "lemma",
]

METADATA_COLS = [
    "word_language",
    "domain",
    "period",
    "genre",
    "sub_genre",
    "provenance",
    "sub_provenance",
    "place_discovery",
    "place_composition",
]

LABEL_COLS = ["period", "genre", "sub_genre", "provenance", "sub_provenance", "domain"]
TEXT_COLS = ["value", "clean_value", "lemma"]


# ----------------------------------------------------------------------------
# Section-15 helpers
# ----------------------------------------------------------------------------


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_integrity(name: str, path: Path, df: pd.DataFrame) -> "OrderedDict[str, Any]":
    return OrderedDict(
        [
            ("corpus", name),
            ("path", str(path.relative_to(REPO_ROOT))),
            ("size_bytes", int(os.path.getsize(path))),
            ("md5", md5_of_file(path)),
            ("n_rows", int(len(df))),
            ("n_cols", int(len(df.columns))),
            ("columns", list(df.columns)),
        ]
    )


def profile_column(series: pd.Series, col_name: str) -> "OrderedDict[str, Any]":
    out: "OrderedDict[str, Any]" = OrderedDict()
    out["dtype"] = str(series.dtype)
    n = len(series)
    null_count = int(series.isna().sum())
    out["null_count"] = null_count
    out["null_ratio"] = round(null_count / n, 6) if n else 0.0

    non_null = series.dropna()
    out["unique_count"] = int(non_null.nunique())

    # Top 10 values by frequency.
    vc = non_null.value_counts().head(10)
    out["top_10"] = [(str(k), int(v)) for k, v in vc.items()]

    # Compound-value detection (only for object/string columns).
    if non_null.dtype == object:
        as_str = non_null.astype(str)
        compound_mask = as_str.str.contains(";", regex=False, na=False)
        if compound_mask.any():
            cvc = as_str[compound_mask].value_counts()
            out["compound_values"] = [(str(k), int(v)) for k, v in cvc.items()]

    # Char-length stats for known text columns.
    if col_name in TEXT_COLS and non_null.dtype == object:
        lengths = non_null.astype(str).str.len()
        out["char_length"] = OrderedDict(
            [
                ("min", int(lengths.min())),
                ("max", int(lengths.max())),
                ("mean", round(float(lengths.mean()), 2)),
                ("median", float(lengths.median())),
            ]
        )

    return out


def column_profiles(df: pd.DataFrame) -> "OrderedDict[str, Any]":
    out: "OrderedDict[str, Any]" = OrderedDict()
    for col in df.columns:
        out[col] = profile_column(df[col], col)
    return out


def fragment_aggregation(df: pd.DataFrame) -> "OrderedDict[str, Any]":
    grouped = df.groupby("fragment_id", sort=False, dropna=False)
    word_counts = grouped.size()

    # Metadata consistency: each metadata col should have nunique <= 1 within
    # each fragment (treating NaN as a value via dropna=False).
    inconsistent: "OrderedDict[str, Any]" = OrderedDict()
    for col in METADATA_COLS:
        if col not in df.columns:
            continue
        nu = grouped[col].nunique(dropna=False)
        bad = nu[nu > 1]
        if len(bad) > 0:
            inconsistent[col] = OrderedDict(
                [
                    ("n_inconsistent_fragments", int(len(bad))),
                    ("fragment_ids", [_jsonable(fid) for fid in bad.index][:50]),
                ]
            )

    return OrderedDict(
        [
            ("n_unique_fragments", int(df["fragment_id"].nunique(dropna=False))),
            ("n_words_total", int(len(df))),
            (
                "words_per_fragment",
                OrderedDict(
                    [
                        ("min", int(word_counts.min())),
                        ("max", int(word_counts.max())),
                        ("mean", round(float(word_counts.mean()), 2)),
                        ("median", float(word_counts.median())),
                        ("p25", float(word_counts.quantile(0.25))),
                        ("p75", float(word_counts.quantile(0.75))),
                    ]
                ),
            ),
            ("metadata_inconsistencies", inconsistent),
        ]
    )


def cross_corpus_consistency(raw_dfs: "OrderedDict[str, pd.DataFrame]") -> "OrderedDict[str, Any]":
    out: "OrderedDict[str, Any]" = OrderedDict()

    # Column set comparison.
    col_sets = {name: list(df.columns) for name, df in raw_dfs.items()}
    out["columns_per_corpus"] = col_sets

    union_cols: set[str] = set()
    for cols in col_sets.values():
        union_cols.update(cols)

    columns_missing: "OrderedDict[str, Any]" = OrderedDict()
    for name, cols in col_sets.items():
        missing = sorted(union_cols - set(cols))
        if missing:
            columns_missing[name] = missing
    out["columns_missing_per_corpus"] = columns_missing
    out["all_corpora_share_schema"] = len(columns_missing) == 0

    # Schema vs the EXPECTED_COLUMNS list from the plan.
    schema_diff: "OrderedDict[str, Any]" = OrderedDict()
    for name, cols in col_sets.items():
        unexpected = sorted(set(cols) - set(EXPECTED_COLUMNS))
        missing_expected = sorted(set(EXPECTED_COLUMNS) - set(cols))
        if unexpected or missing_expected:
            schema_diff[name] = OrderedDict(
                [
                    ("unexpected_columns", unexpected),
                    ("missing_expected_columns", missing_expected),
                ]
            )
    out["schema_vs_plan"] = schema_diff
    out["schema_matches_plan"] = len(schema_diff) == 0

    # Per-label-column unique values per corpus + which corpora contain them.
    label_overlap: "OrderedDict[str, Any]" = OrderedDict()
    for col in LABEL_COLS:
        per_corpus: "OrderedDict[str, Any]" = OrderedDict()
        all_values: set[str] = set()
        for name, df in raw_dfs.items():
            if col not in df.columns:
                per_corpus[name] = None  # column absent
                continue
            vals = sorted({str(v) for v in df[col].dropna().unique()})
            per_corpus[name] = vals
            all_values.update(vals)
        # Build value -> [corpora] map.
        value_to_corpora: "OrderedDict[str, list[str]]" = OrderedDict()
        for v in sorted(all_values):
            value_to_corpora[v] = [
                name
                for name, df in raw_dfs.items()
                if col in df.columns and v in {str(x) for x in df[col].dropna().unique()}
            ]
        label_overlap[col] = OrderedDict(
            [
                ("values_per_corpus", per_corpus),
                ("value_to_corpora", value_to_corpora),
            ]
        )
    out["label_value_overlap"] = label_overlap

    # Case-mismatch flags: group raw values by lowercase+strip key.
    case_mismatches: "OrderedDict[str, Any]" = OrderedDict()
    for col in LABEL_COLS:
        all_vals: set[str] = set()
        for df in raw_dfs.values():
            if col in df.columns:
                all_vals.update(str(v) for v in df[col].dropna().unique())
        groups: dict[str, set[str]] = {}
        for v in all_vals:
            key = v.strip().lower()
            groups.setdefault(key, set()).add(v)
        clashes = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
        if clashes:
            case_mismatches[col] = clashes
    out["case_mismatches"] = case_mismatches

    # Compound vs atomic flags: compound values exist alongside their atomic parts.
    compound_atomic: "OrderedDict[str, Any]" = OrderedDict()
    for col in LABEL_COLS:
        all_vals: set[str] = set()
        for df in raw_dfs.values():
            if col in df.columns:
                all_vals.update(str(v) for v in df[col].dropna().unique())
        compound_vals = [v for v in all_vals if ";" in v]
        if not compound_vals:
            continue
        col_clashes: "OrderedDict[str, list[str]]" = OrderedDict()
        for cv in sorted(compound_vals):
            parts = [p.strip() for p in cv.split(";") if p.strip()]
            atomic_present = [p for p in parts if p in all_vals]
            if atomic_present:
                col_clashes[cv] = atomic_present
        if col_clashes:
            compound_atomic[col] = col_clashes
    out["compound_vs_atomic"] = compound_atomic

    # Global fragment_id collisions across corpora.
    fid_sets = {
        name: {_jsonable(v) for v in df["fragment_id"].dropna().unique()}
        for name, df in raw_dfs.items()
    }
    collisions: "OrderedDict[str, list[Any]]" = OrderedDict()
    names = list(fid_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sorted(fid_sets[names[i]] & fid_sets[names[j]], key=str)
            if inter:
                collisions[f"{names[i]}_vs_{names[j]}"] = inter
    out["fragment_id_collisions"] = collisions

    return out


def normalize_label(col_name: str, value: Any) -> Any:
    """Mirror Section 5 normalization rules so per-task counts match the
    registry's eventual behavior."""
    if pd.isna(value):
        return value
    if col_name in ("genre", "sub_genre"):
        return str(value).strip().lower()
    return str(value)


def to_fragment_table(df: pd.DataFrame, corpus_name: str) -> pd.DataFrame:
    """Collapse a word-level CSV to a fragment-level dataframe.

    Picks the first observed value of each metadata column per fragment_id;
    metadata consistency within fragments is verified separately by
    fragment_aggregation()."""
    grouped = df.groupby("fragment_id", sort=False, dropna=False)
    out = pd.DataFrame({"fragment_id": list(grouped.groups.keys())})
    for col in METADATA_COLS:
        if col in df.columns:
            out[col] = grouped[col].first().values
        else:
            out[col] = np.nan
    out["word_count"] = grouped.size().values
    out["corpus"] = corpus_name
    return out


def task_feasibility(
    fragment_tables: "OrderedDict[str, pd.DataFrame]",
) -> "OrderedDict[str, Any]":
    pooled = pd.concat(fragment_tables.values(), ignore_index=True)

    # (task_name, label_col, optional corpus filter)
    tasks = [
        ("period", "period", None),
        ("genre", "genre", None),
        ("sub_genre", "sub_genre", ["seal"]),
        ("provenance", "provenance", None),
        ("sub_provenance", "sub_provenance", None),
        ("domain", "domain", None),
    ]

    out: "OrderedDict[str, Any]" = OrderedDict()
    for task_name, label_col, corpus_filter in tasks:
        sub = pooled.copy()
        applied_filter = corpus_filter or list(fragment_tables.keys())
        if corpus_filter is not None:
            sub = sub[sub["corpus"].isin(corpus_filter)]
        n_before = int(len(sub))
        sub = sub[sub[label_col].notna()].copy()
        n_after_null = int(len(sub))
        sub["_label"] = sub[label_col].apply(lambda v: normalize_label(label_col, v))

        class_counts = sub["_label"].value_counts()
        n_classes_input = int(len(class_counts))
        singletons = class_counts[class_counts == 1].index.tolist()
        non_singleton = class_counts[class_counts >= 2]
        n_classes_after_drop = int(len(non_singleton))
        n_2to4 = int(((class_counts >= 2) & (class_counts <= 4)).sum())
        n_ge5 = int((class_counts >= 5).sum())
        smallest_non_singleton = (
            int(non_singleton.min()) if len(non_singleton) > 0 else None
        )
        k_used = (
            min(5, smallest_non_singleton)
            if smallest_non_singleton is not None
            else None
        )
        fragments_after_drop = int(non_singleton.sum())

        out[task_name] = OrderedDict(
            [
                ("label_col", label_col),
                ("corpora_pooled", applied_filter),
                ("normalization", "lowercase+strip" if label_col in ("genre", "sub_genre") else "raw"),
                ("fragments_total_input", n_before),
                ("fragments_after_null_filter", n_after_null),
                ("n_classes_input", n_classes_input),
                ("n_singletons", len(singletons)),
                ("singletons", [str(s) for s in singletons]),
                ("n_classes_after_drop", n_classes_after_drop),
                ("fragments_after_drop", fragments_after_drop),
                ("n_classes_2to4", n_2to4),
                ("n_classes_ge5", n_ge5),
                ("smallest_non_singleton_class_size", smallest_non_singleton),
                ("k_used", k_used),
                (
                    "top_10_classes",
                    [(str(k), int(v)) for k, v in class_counts.head(10).items()],
                ),
            ]
        )

    return out


def sanity_assertions(raw_dfs: "OrderedDict[str, pd.DataFrame]") -> list[str]:
    """Return a list of human-readable error messages. Empty list = all OK."""
    errors: list[str] = []
    for name, df in raw_dfs.items():
        if "clean_value" not in df.columns:
            errors.append(f"{name}: clean_value column missing")
            continue
        n_null = int(df["clean_value"].isna().sum())
        if n_null > 0:
            errors.append(f"{name}: {n_null} rows have null clean_value")
        n_empty = int(
            (df["clean_value"].fillna("").astype(str).str.strip() == "").sum()
        )
        # n_empty includes nulls; subtract to count only "non-null but blank".
        n_blank_only = n_empty - n_null
        if n_blank_only > 0:
            errors.append(
                f"{name}: {n_blank_only} rows have empty/whitespace clean_value (non-null)"
            )
        if "fragment_id" not in df.columns:
            errors.append(f"{name}: fragment_id column missing")
        elif df["fragment_id"].isna().any():
            errors.append(
                f"{name}: {int(df['fragment_id'].isna().sum())} rows have null fragment_id"
            )
        # "no fragment has fewer than 1 word" is automatic from groupby semantics
        # (a group cannot exist without at least one row), so no explicit check.
    return errors


# ----------------------------------------------------------------------------
# JSON / Markdown rendering
# ----------------------------------------------------------------------------


def _jsonable(v: Any) -> Any:
    """Coerce numpy / pandas scalars into plain Python types for json.dump."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [to_jsonable(v) for v in sorted(obj, key=str)]
    return _jsonable(obj)


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    a = lines.append

    a("# SEAL Round 4 — Inspection Report")
    a("")
    a(f"Generated: `{report['generated_at']}`")
    a(f"Source: `{report['source_dir']}`")
    a(f"Script: `{report['script']}`")
    a(f"Plan: {report['plan_section']}")
    a("")
    a("This report is the data contract for downstream Phase A/B/C/D scripts.")
    a("Re-running the script overwrites this file. The companion JSON is the")
    a("machine-readable form and is checked into git.")
    a("")

    # ---------- 1. File integrity ----------
    a("## 1. File integrity")
    a("")
    a("| Corpus | Path | Size (bytes) | MD5 | Rows | Cols |")
    a("|--------|------|-------------:|-----|-----:|-----:|")
    for fi in report["file_integrity"]:
        a(
            f"| {fi['corpus']} | `{fi['path']}` | {fi['size_bytes']:,} | "
            f"`{fi['md5']}` | {fi['n_rows']:,} | {fi['n_cols']} |"
        )
    a("")

    # ---------- 2. Per-column profiles ----------
    a("## 2. Per-column profiles")
    a("")
    for name, profiles in report["column_profiles"].items():
        a(f"### {name}.csv")
        a("")
        a("| Column | dtype | Nulls | Null % | Unique | Top values |")
        a("|--------|-------|------:|-------:|-------:|------------|")
        for col, p in profiles.items():
            top_repr = ", ".join(f"`{k}`×{v}" for k, v in p["top_10"][:3])
            a(
                f"| `{col}` | {p['dtype']} | {p['null_count']:,} | "
                f"{p['null_ratio'] * 100:.1f}% | {p['unique_count']:,} | {top_repr} |"
            )
        a("")
        # Char length sub-table for text columns.
        text_lines = []
        for col, p in profiles.items():
            if "char_length" in p:
                cl = p["char_length"]
                text_lines.append(
                    f"| `{col}` | {cl['min']} | {cl['max']} | {cl['mean']} | {cl['median']} |"
                )
        if text_lines:
            a("Text-column character lengths:")
            a("")
            a("| Column | min | max | mean | median |")
            a("|--------|----:|----:|-----:|-------:|")
            for ln in text_lines:
                a(ln)
            a("")
        # Compound values.
        compound_lines = []
        for col, p in profiles.items():
            if "compound_values" in p:
                vals = ", ".join(
                    f"`{k}`×{v}" for k, v in p["compound_values"][:5]
                )
                more = (
                    f" (+{len(p['compound_values']) - 5} more)"
                    if len(p["compound_values"]) > 5
                    else ""
                )
                compound_lines.append(f"- **{col}**: {vals}{more}")
        if compound_lines:
            a("Compound (`;`-containing) values:")
            a("")
            for ln in compound_lines:
                a(ln)
            a("")

    # ---------- 3. Fragment-level aggregation ----------
    a("## 3. Fragment-level aggregation")
    a("")
    a("| Corpus | Fragments | Words | Words/frag min | max | mean | median | p25 | p75 |")
    a("|--------|----------:|------:|---------------:|----:|-----:|-------:|----:|----:|")
    for name, fa in report["fragment_aggregation"].items():
        wpf = fa["words_per_fragment"]
        a(
            f"| {name} | {fa['n_unique_fragments']:,} | {fa['n_words_total']:,} | "
            f"{wpf['min']} | {wpf['max']} | {wpf['mean']} | {wpf['median']} | "
            f"{wpf['p25']} | {wpf['p75']} |"
        )
    a("")
    a("Metadata consistency within fragments (each metadata column should have")
    a("exactly one unique value per fragment_id):")
    a("")
    any_inconsistent = False
    for name, fa in report["fragment_aggregation"].items():
        inc = fa["metadata_inconsistencies"]
        if inc:
            any_inconsistent = True
            a(f"- **{name}**:")
            for col, info in inc.items():
                fids = ", ".join(str(x) for x in info["fragment_ids"][:10])
                more = (
                    f" (+{info['n_inconsistent_fragments'] - 10} more)"
                    if info["n_inconsistent_fragments"] > 10
                    else ""
                )
                a(
                    f"  - `{col}`: {info['n_inconsistent_fragments']} inconsistent "
                    f"fragments — {fids}{more}"
                )
    if not any_inconsistent:
        a("- All corpora pass: every fragment has internally consistent metadata.")
    a("")

    # ---------- 4. Cross-corpus consistency ----------
    a("## 4. Cross-corpus consistency")
    a("")
    cc = report["cross_corpus_consistency"]
    a(
        f"- **All corpora share the same column set:** "
        f"`{cc['all_corpora_share_schema']}`"
    )
    a(
        f"- **Schema matches the plan's expected column list:** "
        f"`{cc['schema_matches_plan']}`"
    )
    if cc["columns_missing_per_corpus"]:
        a("- **Missing columns vs union:**")
        for name, cols in cc["columns_missing_per_corpus"].items():
            a(f"  - `{name}`: {cols}")
    if cc["schema_vs_plan"]:
        a("- **Schema vs plan diff:**")
        for name, diff in cc["schema_vs_plan"].items():
            a(f"  - `{name}`: {diff}")
    a("")

    a("### Label value overlap")
    a("")
    a("For each label column, the unique values present in each corpus.")
    a("")
    for col, info in cc["label_value_overlap"].items():
        a(f"#### `{col}`")
        a("")
        for corpus, vals in info["values_per_corpus"].items():
            if vals is None:
                a(f"- **{corpus}**: column absent")
            elif len(vals) == 0:
                a(f"- **{corpus}**: 0 non-null values")
            elif len(vals) <= 20:
                a(f"- **{corpus}** ({len(vals)} unique): {', '.join(f'`{v}`' for v in vals)}")
            else:
                head = ", ".join(f"`{v}`" for v in vals[:20])
                a(f"- **{corpus}** ({len(vals)} unique): {head}, …")
        a("")

    a("### Case-mismatch flags")
    a("")
    if cc["case_mismatches"]:
        for col, clashes in cc["case_mismatches"].items():
            a(f"- **{col}**:")
            for key, vals in clashes.items():
                a(f"  - `{key}` ← {vals}")
    else:
        a("- None detected (no two raw values normalize to the same lowercase form).")
    a("")

    a("### Compound vs atomic clashes")
    a("")
    if cc["compound_vs_atomic"]:
        for col, clashes in cc["compound_vs_atomic"].items():
            a(f"- **{col}**:")
            for cv, parts in clashes.items():
                a(f"  - compound `{cv}` overlaps atomic value(s) {parts}")
    else:
        a("- None detected.")
    a("")

    a("### Fragment ID collisions across corpora")
    a("")
    if cc["fragment_id_collisions"]:
        for pair, ids in cc["fragment_id_collisions"].items():
            a(f"- **{pair}**: {len(ids)} colliding ids → first 10: {ids[:10]}")
    else:
        a("- None detected — fragment_ids are globally unique across corpora.")
    a("")

    # ---------- 5. Per-task feasibility ----------
    a("## 5. Per-task feasibility")
    a("")
    a("Numbers below apply Section-5 normalization rules (lowercase+strip on")
    a("`genre` / `sub_genre`; everything else raw, including compound provenances).")
    a("Singletons (N=1 classes) are dropped per Section 6.")
    a("")
    a("| Task | Pooled | Frags in | After null | Classes | Singletons | Classes left | Frags left | smallest non-singleton | k |")
    a("|------|--------|---------:|-----------:|--------:|-----------:|-------------:|-----------:|----------------------:|--:|")
    for task, t in report["task_feasibility"].items():
        pooled = "+".join(t["corpora_pooled"])
        a(
            f"| `{task}` | {pooled} | {t['fragments_total_input']} | "
            f"{t['fragments_after_null_filter']} | {t['n_classes_input']} | "
            f"{t['n_singletons']} | {t['n_classes_after_drop']} | "
            f"{t['fragments_after_drop']} | "
            f"{t['smallest_non_singleton_class_size']} | {t['k_used']} |"
        )
    a("")
    for task, t in report["task_feasibility"].items():
        a(f"### `{task}`")
        a("")
        a(f"- label column: `{t['label_col']}`")
        a(f"- normalization: {t['normalization']}")
        a(f"- corpora pooled: {t['corpora_pooled']}")
        a(
            f"- class N distribution: {t['n_singletons']} singletons, "
            f"{t['n_classes_2to4']} classes with 2-4 fragments, "
            f"{t['n_classes_ge5']} classes with ≥5 fragments"
        )
        if t["singletons"]:
            sample = t["singletons"][:15]
            more = (
                f" (+{len(t['singletons']) - 15} more)"
                if len(t["singletons"]) > 15
                else ""
            )
            a(f"- singletons (will be dropped): {sample}{more}")
        a("- top 10 classes by N:")
        for k, v in t["top_10_classes"]:
            a(f"  - `{k}`: {v}")
        a("")

    # ---------- 6. Sanity assertions ----------
    a("## 6. Sanity assertions")
    a("")
    if report["sanity_errors"]:
        a("**FAILED** — the following invariants were violated:")
        a("")
        for e in report["sanity_errors"]:
            a(f"- {e}")
    else:
        a("All assertions passed:")
        a("")
        a("- `clean_value` is non-null and non-empty in every row of all 3 corpora")
        a("- `fragment_id` is non-null in every row")
        a("- Every fragment has at least 1 word (automatic from groupby semantics)")
    a("")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    if not ROUND_DIR.exists():
        print(f"ERROR: {ROUND_DIR} does not exist", file=sys.stderr)
        return 2
    for name, path in CSV_FILES.items():
        if not path.exists():
            print(f"ERROR: missing CSV {path}", file=sys.stderr)
            return 2

    print(f"[inspect] reading 3 CSVs from {ROUND_DIR.relative_to(REPO_ROOT)}")
    raw_dfs: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for name, path in CSV_FILES.items():
        df = pd.read_csv(path)
        raw_dfs[name] = df
        print(f"[inspect]   {name}: {len(df):,} rows × {len(df.columns)} cols")

    report: "OrderedDict[str, Any]" = OrderedDict()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["source_dir"] = str(ROUND_DIR.relative_to(REPO_ROOT))
    report["script"] = "v_1/src/corpus/01_inspect_seal_data.py"
    report["plan_section"] = (
        "Section 15 of v_1/justification/seal_round4_pipeline_plan.md"
    )

    print("[inspect] computing file integrity...")
    report["file_integrity"] = [
        file_integrity(name, path, raw_dfs[name])
        for name, path in CSV_FILES.items()
    ]

    print("[inspect] profiling columns...")
    report["column_profiles"] = OrderedDict(
        (name, column_profiles(df)) for name, df in raw_dfs.items()
    )

    print("[inspect] aggregating fragments...")
    report["fragment_aggregation"] = OrderedDict(
        (name, fragment_aggregation(df)) for name, df in raw_dfs.items()
    )

    print("[inspect] checking cross-corpus consistency...")
    report["cross_corpus_consistency"] = cross_corpus_consistency(raw_dfs)

    print("[inspect] computing per-task feasibility...")
    fragment_tables = OrderedDict(
        (name, to_fragment_table(df, name)) for name, df in raw_dfs.items()
    )
    report["task_feasibility"] = task_feasibility(fragment_tables)

    print("[inspect] running sanity assertions...")
    sanity_errors = sanity_assertions(raw_dfs)
    report["sanity_errors"] = sanity_errors

    # Write JSON.
    REPORT_JSON.write_text(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False) + "\n")
    print(f"[inspect] wrote {REPORT_JSON.relative_to(REPO_ROOT)}")

    # Write Markdown.
    REPORT_MD.write_text(render_markdown(report))
    print(f"[inspect] wrote {REPORT_MD.relative_to(REPO_ROOT)}")

    # Console summary.
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for fi in report["file_integrity"]:
        print(f"  {fi['corpus']:>5}: {fi['n_rows']:>6,} rows, md5={fi['md5'][:8]}")
    print()
    print("Per-task fragment counts (post normalization, post-singleton-drop):")
    for task, t in report["task_feasibility"].items():
        print(
            f"  {task:>16}: {t['fragments_after_drop']:>4} frags / "
            f"{t['n_classes_after_drop']:>3} classes (k={t['k_used']}, "
            f"{t['n_singletons']} singletons dropped)"
        )
    print()
    if sanity_errors:
        print("SANITY ASSERTIONS FAILED:")
        for e in sanity_errors:
            print(f"  - {e}")
        return 1

    print("All sanity assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
