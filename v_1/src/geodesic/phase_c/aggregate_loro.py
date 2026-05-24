#!/usr/bin/env python3
"""Aggregate Phase C LORO results into a drop table."""

import json
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[4]
RESULTS    = ROOT / "v_1/src/geodesic/results/phase_c"
THEIR_GATE = {"strong": 0.10, "hedge": 0.20}


def main():
    files = sorted(RESULTS.glob("loro_*.json"))
    if not files:
        print("No LORO results found.")
        sys.exit(1)

    rows = [json.loads(f.read_text()) for f in files]

    # Group by (method, cleaning, pool, layer)
    configs = {}
    for r in rows:
        key = (r["method"], r["cleaning"], r["pool"], r["layer"])
        configs.setdefault(key, []).append(r)

    print(f"=== Phase C LORO summary ({len(rows)} ruler×config results) ===\n")
    fmt = "{:<30} {:<9} {:<5} {:>3}  {:>12} {:>12} {:>10}  {:>8}"
    print(fmt.format("Method", "Cleaning", "Pool", "L",
                     "pacc_full", "pacc_loro", "drop_pacc", "Verdict"))
    print("-" * 100)

    for key, group in sorted(configs.items()):
        method, cleaning, pool, layer = key
        full_acc  = group[0]["pacc_full"]
        loro_accs = [g["pacc_loro"] for g in group]
        loro_mean = sum(loro_accs) / len(loro_accs)
        drop      = full_acc - loro_mean

        if drop < THEIR_GATE["strong"]:
            verdict = "STRONG (temporal)"
        elif drop < THEIR_GATE["hedge"]:
            verdict = "HEDGED"
        else:
            verdict = "WEAK (ruler-driven)"

        print(fmt.format(method, cleaning, pool, layer,
                         f"{full_acc:.4f}", f"{loro_mean:.4f}",
                         f"{drop:.4f}", verdict))

        # Per-ruler detail
        for g in sorted(group, key=lambda x: -x["n_held_out"]):
            print(f"    {g['ruler']:<35} n={g['n_held_out']:>3}"
                  f"  drop={g['pacc_full']-g['pacc_loro']:+.4f}"
                  f"  cross={g['pacc_loro_cross']:.4f}")

    # Save summary JSON
    summary = []
    for key, group in configs.items():
        method, cleaning, pool, layer = key
        full_acc  = group[0]["pacc_full"]
        loro_mean = sum(g["pacc_loro"] for g in group) / len(group)
        drop      = full_acc - loro_mean
        summary.append({
            "method": method, "cleaning": cleaning, "pool": pool, "layer": layer,
            "pacc_full": full_acc, "pacc_loro_mean": loro_mean, "drop": drop,
            "n_rulers": len(group),
            "per_ruler": [{
                "ruler": g["ruler"], "n": g["n_held_out"],
                "pacc_loro": g["pacc_loro"], "pacc_cross": g["pacc_loro_cross"],
                "drop": g["pacc_full"] - g["pacc_loro"],
            } for g in sorted(group, key=lambda x: x["ruler"])],
        })

    out = ROOT / "v_1/src/geodesic/results/loro_robustness.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {out}")


if __name__ == "__main__":
    main()
