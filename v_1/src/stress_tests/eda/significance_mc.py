"""significance_mc.py — are two balanced-MC results actually different?

The 200 balanced draws are IDENTICAL for every model/arm (draws_matrix.npy),
so two results' per-draw Spearman series are PAIRED samples. The honest test
is therefore on the per-draw DIFFERENCES — mean±std overlap between two rows
massively understates significance, because draw-to-draw variability (the
±0.07 you see on slides) is shared between the paired runs and cancels.

Two modes:
  paired   both results carry `per_draw_spearman` (emitted by mc_probe /
           T11/T12 scorers after 2026-07-12): mean delta, paired SE/t, win
           rate, sign-test p. This is the number to quote.
  naive    only mean/std available (older results): independent-samples z
           with n=200. Reported as APPROXIMATE — it ignores (a) the positive
           pairing covariance (making it conservative) and (b) fragment
           overlap between draws (making it optimistic). Use it for triage,
           rerun the probe for the paired test where it matters.

Usage:
  python significance_mc.py A.json B.json [--path mc_balanced]
  python significance_mc.py --naive 0.319 0.07 0.287 0.07 [--n 200]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def paired_stats(a: list[float], b: list[float]) -> dict:
    n = min(len(a), len(b))
    assert n >= 10, "too few paired draws"
    d = [x - y for x, y in zip(a[:n], b[:n])]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else float("inf")
    wins = sum(1 for x in d if x > 0)
    # normal approximations (n=200 is comfortably large)
    p_t = 2 * (1 - _phi(abs(t)))
    zs = (wins - n / 2) / math.sqrt(n / 4)
    p_sign = 2 * (1 - _phi(abs(zs)))
    return {"n_draws": n, "mean_delta": mean, "sd_delta": math.sqrt(var),
            "se": se, "t": t, "p_approx": p_t,
            "win_rate": wins / n, "sign_test_p": p_sign}


def naive_stats(m1: float, s1: float, m2: float, s2: float, n: int = 200) -> dict:
    se = math.sqrt((s1 ** 2 + s2 ** 2) / n)
    z = (m1 - m2) / se if se > 0 else float("inf")
    return {"mean_delta": m1 - m2, "se_independent": se, "z_approx": z,
            "p_approx": 2 * (1 - _phi(abs(z))),
            "caveat": "independent-draws approximation — see module docstring"}


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _extract(fp: Path, path: str):
    d = json.loads(fp.read_text())
    node = d
    for key in path.split("."):
        if key:
            node = node[key]
    if isinstance(node, dict):
        if "per_draw_spearman" in node:
            return node["per_draw_spearman"], node.get("spearman_mean"), node.get("spearman_std")
        return None, node.get("spearman_mean"), node.get("spearman_std")
    return node, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", help="two result JSONs to compare (A - B)")
    p.add_argument("--path", default="mc_balanced",
                   help="dotted path to the block holding per_draw_spearman / mean / std")
    p.add_argument("--naive", nargs=4, type=float, metavar=("M1", "S1", "M2", "S2"),
                   help="no files: approximate z from two mean/std pairs")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()

    if a.naive:
        r = naive_stats(*a.naive, n=a.n)
        print(json.dumps(r, indent=2))
        return
    assert len(a.files) == 2, "give two JSON files (or --naive M1 S1 M2 S2)"
    pa, ma, sa = _extract(Path(a.files[0]), a.path)
    pb, mb, sb = _extract(Path(a.files[1]), a.path)
    if pa and pb:
        r = paired_stats(pa, pb)
        print(f"PAIRED over {r['n_draws']} shared draws:  "
              f"delta = {r['mean_delta']:+.4f} +- {r['sd_delta']:.4f} (SD of per-draw delta)")
        print(f"  paired t = {r['t']:.1f}  (p ~ {r['p_approx']:.2g})   "
              f"win rate = {r['win_rate']:.0%}  (sign-test p ~ {r['sign_test_p']:.2g})")
    else:
        print("per-draw series missing in at least one file -> NAIVE approximation "
              "(rerun the probe with the patched engine for the paired test):")
        r = naive_stats(ma, sa, mb, sb, n=a.n)
        print(f"  delta = {r['mean_delta']:+.4f}   z ~ {r['z_approx']:.1f}   p ~ {r['p_approx']:.2g}")


if __name__ == "__main__":
    main()
