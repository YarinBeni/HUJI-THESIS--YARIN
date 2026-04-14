#!/usr/bin/env python
"""seal_tasks.py — Task registry and data loader for SEAL/DLL/LBPL experiments.

Single source of truth for the 6 multi-task labels.  Imported by both the
bias_check (Phase C) and linear_probing (Phase D) pipelines.

Usage in other scripts:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from seal_tasks import load_task_data, TASK_NAMES

When run directly, executes a self-test across all 6 tasks, verifies the
numbers against the Phase 0 inspection contract, and writes the report to
v_1/data/evaluation/corpora/seal_tasks_verification.md.

Plan reference: Section 11 steps 9–11 of
  v_1/justification/seal_round4_pipeline_plan.md

Dependencies: stdlib + pandas + numpy only.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[2]   # v_1/src/bias_check/ → repo root

CORPUS_PARQUET = (
    REPO_ROOT / "v_1" / "data" / "evaluation" / "corpora" / "seal_corpus.parquet"
)
INSPECTION_REPORT = (
    REPO_ROOT / "v_1" / "data" / "raw" / "chungrong" / "seal_round4"
    / "inspection_report.json"
)
VERIFICATION_OUT = (
    REPO_ROOT / "v_1" / "data" / "evaluation" / "corpora"
    / "seal_tasks_verification.md"
)

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

# Each entry: label_col, corpora to pool (None = all three).
# Label normalization (Section 5) is already applied in the corpus parquet for
# genre / sub_genre; all other labels are stored raw.
_TASK_REGISTRY: "dict[str, dict[str, Any]]" = {
    "period": {
        "label_col":  "period",
        "corpora":    None,     # SEAL + DLL + LBPL
        "norm":       "raw",
    },
    "genre": {
        "label_col":  "genre",
        "corpora":    None,
        "norm":       "lowercase+strip",   # applied in parquet
    },
    "sub_genre": {
        "label_col":  "sub_genre",
        "corpora":    ["seal"],  # DLL / LBPL have no sub_genre
        "norm":       "lowercase+strip",   # applied in parquet
    },
    "provenance": {
        "label_col":  "provenance",
        "corpora":    None,
        "norm":       "raw",
    },
    "sub_provenance": {
        "label_col":  "sub_provenance",
        "corpora":    None,
        "norm":       "raw",
    },
    "domain": {
        "label_col":  "domain",
        "corpora":    None,
        "norm":       "raw",    # SEAL / DLL / LBPL corpus-membership tag
    },
}

TASK_NAMES: list[str] = list(_TASK_REGISTRY.keys())

# Text column mapping.
_CLEANING_TO_COL: dict[str, str] = {
    "raw":     "text",
    "tier0":   "text_tier0",
    "maximal": "text_maximal",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_task_data(
    task_name: str,
    cleaning: str = "tier0",
) -> "tuple[pd.DataFrame, dict[str, Any]]":
    """Load, filter, normalize, and return data for one task.

    Parameters
    ----------
    task_name : str
        One of the 6 keys in TASK_NAMES.
    cleaning : str
        Which text column to expose as ``text``:
        "raw" → ``text``, "tier0" → ``text_tier0``, "maximal" → ``text_maximal``.

    Returns
    -------
    df : pd.DataFrame
        One row per fragment.  Columns:
          text        – selected cleaned text
          label_raw   – the label string (normalized per Section 5 for genre/sub_genre)
          label_idx   – integer class index (alphabetically sorted over surviving classes)
          fragment_id, corpus, word_count, domain, period, genre, sub_genre,
          provenance, sub_provenance, word_language
    task_summary : dict
        Statistics matching the structure of the per-task task_summary.json
        written by Phase C / D pipeline runs.  Cross-checked in self_test()
        against the Phase 0 inspection contract.
    """
    if task_name not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Choose from: {TASK_NAMES}"
        )
    if cleaning not in _CLEANING_TO_COL:
        raise ValueError(
            f"Unknown cleaning '{cleaning}'. Choose from: {list(_CLEANING_TO_COL)}"
        )

    task = _TASK_REGISTRY[task_name]
    label_col = task["label_col"]
    text_col = _CLEANING_TO_COL[cleaning]

    # ── Load parquet ─────────────────────────────────────────────────────────
    if not CORPUS_PARQUET.exists():
        raise FileNotFoundError(
            f"Corpus parquet not found: {CORPUS_PARQUET}\n"
            f"Run v_1/src/corpus/02_build_seal_corpus.py first."
        )
    df = pd.read_parquet(CORPUS_PARQUET)

    # ── Filter by corpora ─────────────────────────────────────────────────────
    all_corpora = ["seal", "dll", "lbpl"]
    corpora = task["corpora"] if task["corpora"] is not None else all_corpora
    df = df[df["corpus"].isin(corpora)].copy()
    n_input = len(df)

    # ── Filter out null labels ────────────────────────────────────────────────
    df = df[df[label_col].notna()].copy()
    n_after_null = len(df)

    # ── Drop singleton classes (N=1) ──────────────────────────────────────────
    # Singletons are mathematically untestable with stratified k-fold CV.
    # Section 6 of the pipeline plan.
    class_counts = df[label_col].value_counts()
    singletons = sorted(class_counts[class_counts == 1].index.tolist())
    n_classes_input = int(len(class_counts))

    df = df[~df[label_col].isin(singletons)].copy()
    n_classes_after_drop = int(df[label_col].nunique())
    n_after_drop = int(len(df))

    # ── Effective k ───────────────────────────────────────────────────────────
    surviving_counts = df[label_col].value_counts()
    smallest_class = int(surviving_counts.min()) if len(surviving_counts) > 0 else 0
    k_used = min(5, smallest_class) if smallest_class > 0 else 0

    # ── Label encoding (alphabetically sorted for reproducibility) ────────────
    class_names = sorted(df[label_col].unique())
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    df["label_raw"] = df[label_col].astype(str)
    df["label_idx"] = df[label_col].map(label_to_idx).astype(int)
    df["text"] = df[text_col].astype(str)

    # ── Select output columns ─────────────────────────────────────────────────
    keep_cols = [
        "text", "label_raw", "label_idx",
        "fragment_id", "corpus", "word_count",
        "domain", "period", "genre", "sub_genre",
        "provenance", "sub_provenance", "word_language",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

    # ── Build task summary ────────────────────────────────────────────────────
    top5 = [
        (str(k), int(v))
        for k, v in surviving_counts.head(5).items()
    ]
    task_summary: "dict[str, Any]" = OrderedDict(
        [
            ("task_name",                task_name),
            ("label_col",                label_col),
            ("corpora_pooled",           corpora),
            ("cleaning",                 cleaning),
            ("text_col_used",            text_col),
            ("fragments_total_input",    n_input),
            ("fragments_after_null_filter", n_after_null),
            ("n_classes_input",          n_classes_input),
            ("singletons_dropped",       singletons),
            ("n_classes_after_drop",     n_classes_after_drop),
            ("fragments_after_drop",     n_after_drop),
            ("smallest_class_size",      smallest_class),
            ("k_used",                   k_used),
            ("top5_classes",             top5),
            ("label_to_idx",             label_to_idx),
        ]
    )
    return df, task_summary


# ---------------------------------------------------------------------------
# Self-test (run as __main__)
# ---------------------------------------------------------------------------


def _load_inspection_feasibility() -> "dict[str, Any]":
    if not INSPECTION_REPORT.exists():
        raise FileNotFoundError(
            f"Inspection report not found: {INSPECTION_REPORT}\n"
            f"Run v_1/src/corpus/01_inspect_seal_data.py first."
        )
    with open(INSPECTION_REPORT, encoding="utf-8") as f:
        report = json.load(f)
    return report["task_feasibility"]


def self_test() -> int:
    """Iterate over all 6 tasks, verify counts against Phase 0 contract.

    Writes v_1/data/evaluation/corpora/seal_tasks_verification.md.
    Returns exit code 0 on success, 1 on any mismatch.
    """
    feasibility = _load_inspection_feasibility()

    md_lines: list[str] = []
    md_lines.append("# SEAL Task Registry — Self-Test Verification")
    md_lines.append("")
    md_lines.append(f"Generated: `{datetime.now(timezone.utc).isoformat()[:19]}Z`")
    md_lines.append(
        "Source: `v_1/src/bias_check/seal_tasks.py` (Phase B self-test)"
    )
    md_lines.append(
        "Contract: `v_1/data/raw/chungrong/seal_round4/inspection_report.json`"
    )
    md_lines.append("")
    md_lines.append(
        "This report confirms that `load_task_data()` reproduces the fragment and "
        "class counts predicted by the Phase 0 inspection script.  A mismatch here "
        "means the parquet or the registry has drifted from the agreed contract."
    )
    md_lines.append("")

    # Summary table header.
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append(
        "| Task | Corpora | N in | After null | Classes | Singletons "
        "| Classes left | N left | k | Status |"
    )
    md_lines.append(
        "|------|---------|-----:|-----------:|--------:|-----------:|"
        "-------------:|-------:|--:|--------|"
    )

    all_passed = True
    task_details: list[str] = []

    for task_name in TASK_NAMES:
        df, summary = load_task_data(task_name, cleaning="tier0")
        ref = feasibility[task_name]

        # Fields to compare (key → (actual, expected)).
        checks: "list[tuple[str, Any, Any]]" = [
            ("fragments_total_input",       summary["fragments_total_input"],    ref["fragments_total_input"]),
            ("fragments_after_null_filter", summary["fragments_after_null_filter"], ref["fragments_after_null_filter"]),
            ("n_classes_input",             summary["n_classes_input"],          ref["n_classes_input"]),
            ("n_singletons",                len(summary["singletons_dropped"]),  ref["n_singletons"]),
            ("n_classes_after_drop",        summary["n_classes_after_drop"],     ref["n_classes_after_drop"]),
            ("fragments_after_drop",        summary["fragments_after_drop"],     ref["fragments_after_drop"]),
            ("k_used",                      summary["k_used"],                   ref["k_used"]),
        ]

        mismatches = [
            (field, actual, expected)
            for field, actual, expected in checks
            if actual != expected
        ]
        status = "✓ PASS" if not mismatches else "✗ FAIL"
        if mismatches:
            all_passed = False

        # Summary row.
        corpora_str = "+".join(summary["corpora_pooled"])
        md_lines.append(
            f"| `{task_name}` | {corpora_str} "
            f"| {summary['fragments_total_input']} "
            f"| {summary['fragments_after_null_filter']} "
            f"| {summary['n_classes_input']} "
            f"| {len(summary['singletons_dropped'])} "
            f"| {summary['n_classes_after_drop']} "
            f"| {summary['fragments_after_drop']} "
            f"| {summary['k_used']} "
            f"| {status} |"
        )

        # Per-task detail block.
        block: list[str] = []
        block.append(f"## Task: `{task_name}`")
        block.append("")
        block.append(f"- label column: `{summary['label_col']}`")
        block.append(f"- corpora pooled: {summary['corpora_pooled']}")
        block.append(f"- N fragments (input): {summary['fragments_total_input']}")
        block.append(
            f"- N fragments (after null filter): {summary['fragments_after_null_filter']}"
        )
        block.append(
            f"- N classes (input): {summary['n_classes_input']}; "
            f"singletons: {len(summary['singletons_dropped'])}"
        )
        block.append(
            f"- N classes (after singleton drop): {summary['n_classes_after_drop']}"
        )
        block.append(
            f"- N fragments (after singleton drop): {summary['fragments_after_drop']}"
        )
        block.append(
            f"- smallest surviving class size: {summary['smallest_class_size']}"
        )
        block.append(f"- effective k: {summary['k_used']}")
        block.append("")
        if summary["singletons_dropped"]:
            sample = summary["singletons_dropped"][:15]
            more = (
                f" (+{len(summary['singletons_dropped']) - 15} more)"
                if len(summary["singletons_dropped"]) > 15
                else ""
            )
            block.append(f"- singletons dropped: {sample}{more}")
            block.append("")
        block.append("Top 5 classes by N:")
        block.append("")
        for cls, n in summary["top5_classes"]:
            block.append(f"  - `{cls}`: {n}")
        block.append("")

        if mismatches:
            block.append("**MISMATCHES vs inspection contract:**")
            block.append("")
            for field, actual, expected in mismatches:
                block.append(f"- `{field}`: got {actual}, expected {expected}")
            block.append("")
        else:
            block.append("All counts match the Phase 0 inspection contract. ✓")
            block.append("")

        # Console output.
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")
        print(f"  label_col         : {summary['label_col']}")
        print(f"  corpora_pooled    : {summary['corpora_pooled']}")
        print(f"  N input           : {summary['fragments_total_input']}")
        print(f"  N after null      : {summary['fragments_after_null_filter']}")
        print(f"  classes input     : {summary['n_classes_input']}")
        print(f"  singletons        : {len(summary['singletons_dropped'])}")
        print(f"  classes surviving : {summary['n_classes_after_drop']}")
        print(f"  N surviving       : {summary['fragments_after_drop']}")
        print(f"  k_used            : {summary['k_used']}")
        print(f"  top-5 classes     : {summary['top5_classes']}")
        print(f"  status            : {status}")
        if mismatches:
            for field, actual, expected in mismatches:
                print(f"  MISMATCH  {field}: got={actual} expected={expected}")

        task_details.extend(block)

    md_lines.append("")
    md_lines.extend(task_details)

    # Write verification file.
    md_text = "\n".join(md_lines) + "\n"
    VERIFICATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    VERIFICATION_OUT.write_text(md_text, encoding="utf-8")

    print(f"\n[seal_tasks] wrote {VERIFICATION_OUT.relative_to(REPO_ROOT)}")
    print(f"[seal_tasks] self-test {'PASSED' if all_passed else 'FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(self_test())
