#!/usr/bin/env python3
"""Aggregate all Phase B scan results into a scoreboard.

Reads all phase_b_*.json from results/phase_b/, builds:
  - geodesic_layer_scoreboard.json   (one row per method×cleaning×pool×layer)
  - geodesic_best_layers.json        (best layer per combo by isomap pairwise acc)

Usage:
    python v_1/src/geodesic/phase_b/aggregate.py
"""

import json
import sys
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = ROOT / "v_1/src/geodesic/results/phase_b"
OUT_DIR     = ROOT / "v_1/src/geodesic/results"


def main():
    files = sorted(RESULTS_DIR.glob("phase_b_*.json"))
    if not files:
        print(f"No phase_b results found in {RESULTS_DIR}")
        sys.exit(1)
    print(f"Found {len(files)} result files")

    rows = []
    for jf in files:
        data = json.loads(jf.read_text())
        method   = data["method"]
        cleaning = data["cleaning"]
        pool     = data["pool"]

        for layer_str, lr in data["layers"].items():
            layer = int(layer_str)
            if "error" in lr:
                continue

            iso  = lr.get("isomap", {})
            ebin = lr.get("earliest_bin", {})

            rows.append({
                "method":   method,
                "cleaning": cleaning,
                "pool":     pool,
                "layer":    layer,
                "k_used":   lr.get("k_used"),
                "isomap_spearman":        iso.get("spearman"),
                "isomap_pairwise_acc":    iso.get("pairwise_order_acc"),
                "isomap_neighbor_purity": iso.get("neighbor_purity"),
                "isomap_neighbor_sigma":  iso.get("neighbor_purity_sigma"),
                "ebin_spearman":          ebin.get("spearman"),
                "ebin_pairwise_acc":      ebin.get("pairwise_order_acc"),
                "ebin_neighbor_purity":   ebin.get("neighbor_purity"),
                "ebin_neighbor_sigma":    ebin.get("neighbor_purity_sigma"),
            })

    scoreboard_path = OUT_DIR / "geodesic_layer_scoreboard.json"
    scoreboard_path.write_text(json.dumps(rows, indent=2))
    print(f"Scoreboard → {scoreboard_path}  ({len(rows)} rows)")

    # Best layer per (method, cleaning, pool) by isomap pairwise acc
    key_fn = lambda r: (r["method"], r["cleaning"], r["pool"])
    rows_sorted = sorted(rows, key=key_fn)
    best = {}
    for combo, grp in groupby(rows_sorted, key=key_fn):
        valid = [r for r in grp if r["isomap_pairwise_acc"] is not None]
        if not valid:
            continue
        b = max(valid, key=lambda r: r["isomap_pairwise_acc"] or 0)
        best[f"{combo[0]}__{combo[1]}__{combo[2]}"] = {
            "method":            combo[0],
            "cleaning":          combo[1],
            "pool":              combo[2],
            "best_layer":        b["layer"],
            "isomap_pairwise_acc": b["isomap_pairwise_acc"],
            "isomap_spearman":   b["isomap_spearman"],
            "ebin_pairwise_acc": b["ebin_pairwise_acc"],
            "ebin_spearman":     b["ebin_spearman"],
        }

    best_path = OUT_DIR / "geodesic_best_layers.json"
    best_path.write_text(json.dumps(best, indent=2))
    print(f"Best layers → {best_path}  ({len(best)} combos)")

    # Summary table
    print("\n=== Phase B: best layer per (method, cleaning, pool) — sorted by isomap pairwise acc ===")
    fmt = "{:<30} {:<10} {:<6} {:>5}  {:>10} {:>8}  {:>10} {:>8}"
    print(fmt.format("Method", "Cleaning", "Pool", "Layer",
                     "Iso Pacc", "Iso Sp", "Ebin Pacc", "Ebin Sp"))
    print("-" * 95)
    sorted_best = sorted(best.values(), key=lambda r: r["isomap_pairwise_acc"] or 0, reverse=True)
    for v in sorted_best:
        print(fmt.format(
            v["method"], v["cleaning"], v["pool"], v["best_layer"],
            f"{v['isomap_pairwise_acc']:.4f}", f"{v['isomap_spearman']:.4f}",
            f"{v['ebin_pairwise_acc']:.4f}" if v["ebin_pairwise_acc"] is not None else "N/A",
            f"{v['ebin_spearman']:.4f}"     if v["ebin_spearman"]     is not None else "N/A",
        ))


if __name__ == "__main__":
    main()
