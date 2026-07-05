"""build_site_balanced_subset.py — balanced Monte-Carlo draws BY FIND-SPOT for the
P2 geography probe (the site-side mirror of the 8-ruler / k=21 year subset).

Same recipe as the ruler analysis: look at the per-class fragment counts, drop the
tail classes that are too small for balanced draws, cap k at the smallest retained
class. Two site-specific twists:

  * only fragments whose provenance is geocoded in shared/sites_gazetteer.csv count;
  * provenance strings are MERGED BY COORDINATE (rounded to 0.1 deg ~ 11 km) first —
    "Kuyunjik (Nineveh)" + "Nineveh" are one place, as are "Babylon" +
    "Babylon (Bābili)" + the "Babylonia" region centroid. Without the merge a
    held-out-site split would leak (train on Babylon, test on Bābili = same spot).

Defaults chosen from the observed distribution (mirrors ruler 8x21):
  min_count=18 -> 10 merged sites, k=18 -> 180 fragments/draw, 200 draws, seed 42.
  Retained: Nineveh 507, Babylon(+Bābili+Babylonia) 165, Assur 99, Nimrud 98,
  Khorsabad 59, Sippar 29, Luristan 28, Borsippa 24, Ur 21, Uruk 18.

Writes (committed):
  .../balanced_subset_sites/draws_matrix.npy          (200 x 1202 bool)
  .../balanced_subset_sites/corpus_fragment_order.json
  .../balanced_subset_sites/site_labels.json           per corpus row: merged site key or null
  .../balanced_subset_sites/manifest.json

Usage:  python v_1/src/stress_tests/p2_godey_geography/build_site_balanced_subset.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
GAZ = Path(__file__).resolve().parents[1] / "shared" / "sites_gazetteer.csv"
OUT = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset_sites"


def merged_site_labels(df: pd.DataFrame, gaz: pd.DataFrame):
    """Per corpus row: merged site key ('lat,lon' at 0.1 deg) or None; plus
    key -> (canonical name, lat, lon, region)."""
    gmap = {str(r.provenance): (float(r.lat), float(r.lon), str(r.region))
            for r in gaz.itertuples(index=False)}
    labels, meta, counts = [], {}, {}
    for p in df["provenance"].astype(str):
        if p not in gmap:
            labels.append(None)
            continue
        lat, lon, region = gmap[p]
        key = f"{round(lat,1)},{round(lon,1)}"
        labels.append(key)
        counts[key] = counts.get(key, 0) + 1
        if key not in meta or counts[key] > meta[key]["n_of_name"]:
            # canonical name = the most frequent provenance string for this coord
            pass
        meta.setdefault(key, {"names": {}, "lat": lat, "lon": lon, "region": region,
                              "n_of_name": 0})
        meta[key]["names"][p] = meta[key]["names"].get(p, 0) + 1
    for key, m in meta.items():
        m["canonical"] = max(m["names"], key=m["names"].get)
        m["n_of_name"] = m["names"][m["canonical"]]
    return labels, meta


def build(n_draws: int, min_count: int, seed_base: int, out_dir: Path):
    df = pd.read_parquet(CORPUS)
    gaz = pd.read_csv(GAZ).dropna(subset=["lat", "lon"])
    labels, meta = merged_site_labels(df, gaz)
    lab = np.array([x if x is not None else "" for x in labels])

    counts = pd.Series([x for x in labels if x]).value_counts()
    kept = counts[counts >= min_count]
    k = int(kept.min())
    sites = list(kept.index)
    print(f"retained {len(sites)} merged sites (min_count>={min_count}), k={k}, "
          f"{k*len(sites)} frags/draw")
    for s in sites:
        print(f"  {meta[s]['canonical']:28s} n={int(counts[s]):>4} "
              f"({', '.join(n for n in meta[s]['names'] if n != meta[s]['canonical']) or '—'})")

    n = len(df)
    draws = np.zeros((n_draws, n), dtype=bool)
    for i in range(n_draws):
        rng = np.random.default_rng(seed_base + i)
        for s in sites:
            idx = np.where(lab == s)[0]
            draws[i, rng.choice(idx, size=k, replace=False)] = True

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "draws_matrix.npy", draws)
    (out_dir / "corpus_fragment_order.json").write_text(
        json.dumps(df["fragment_id"].astype(str).tolist()), encoding="utf-8")
    (out_dir / "site_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    manifest = {
        "config": "site-balanced-mc", "n_draws": n_draws, "k": k,
        "min_count": min_count, "n_sites": len(sites),
        "sites": {s: {"canonical": meta[s]["canonical"], "n": int(counts[s]),
                      "lat": meta[s]["lat"], "lon": meta[s]["lon"],
                      "region": meta[s]["region"],
                      "merged_names": sorted(meta[s]["names"])} for s in sites},
        "coordinate_merge": "0.1 deg rounding (~11 km)",
        "total_frags_per_draw": k * len(sites), "seed_base": seed_base,
        "produced_by": "build_site_balanced_subset.py",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"draws_matrix {draws.shape}, per-draw={int(draws[0].sum())}; wrote {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_draws", type=int, default=200)
    p.add_argument("--min_count", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=str(OUT))
    a = p.parse_args()
    build(a.n_draws, a.min_count, a.seed, Path(a.out_dir))
