"""build_maxking_subset.py — balanced Monte-Carlo draws for the "maximal-with-kings"
config. Unlike the original 8-ruler / k=21 subset, this one:

  * keeps only the 5 rulers whose expected king-found count per draw was >= 6
    (Ashurbanipal, Sennacherib, Esarhaddon, Sargon II, Sîn-šarru-iškun);
  * draws ONLY from each ruler's king-FOUND fragments (under maximal_keepking
    cleaning), so all three pooling sites (mean / king_last / king_mean) are defined
    on the same drawn fragments — a true apples-to-apples set;
  * uses k = 9 per ruler (capped by Sîn-šarru-iškun's 9 king-found fragments).

Writes a parallel subset dir so the original 8-ruler subset is untouched.

Usage:
    python v_1/src/stress_tests/p1_gurnee_tegmark/build_maxking_subset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from cleaning import clean_maximal_keepking          # noqa: E402
from king_token import find_name_word, load_spellings  # noqa: E402

RULERS_5 = ["Ashurbanipal", "Sennacherib", "Esarhaddon", "Sargon II", "Sîn-šarru-iškun"]
DEFAULT_CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
DEFAULT_OUT = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset_maxking"


def king_found_mask(df: pd.DataFrame, spell) -> np.ndarray:
    """Per-fragment: is the ruler's name locatable in the maximal_keepking text?
    (matches what the extractor will find at token level, sans tokenizer edge-cases)."""
    def hit(row):
        sp = spell.get(row["ruler"])
        if not sp:
            return False
        cleaned, name = clean_maximal_keepking(str(row["text_tier0"]), sp)
        return name is not None and find_name_word(cleaned, sp) is not None
    return df.apply(hit, axis=1).to_numpy()


def build(corpus, n_draws, k, seed_base, out_dir):
    df = pd.read_parquet(corpus)
    spell = load_spellings()
    found = king_found_mask(df, spell)
    fid = df["fragment_id"].astype(str).tolist()

    # eligible corpus-row indices per retained ruler (king-found only)
    elig = {}
    for r in RULERS_5:
        idx = np.where((df["ruler"].to_numpy() == r) & found)[0]
        elig[r] = idx
        assert len(idx) >= k, f"{r}: only {len(idx)} king-found frags, need >= k={k}"

    n = len(df)
    draws = np.zeros((n_draws, n), dtype=bool)
    per_draw_counts = []
    for i in range(n_draws):
        rng = np.random.default_rng(seed_base + i)
        cnt = {}
        for r in RULERS_5:
            pick = rng.choice(elig[r], size=k, replace=False)
            draws[i, pick] = True
            cnt[r] = int(k)
        per_draw_counts.append(cnt)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "draws_matrix.npy", draws)
    (out_dir / "corpus_fragment_order.json").write_text(json.dumps(fid), encoding="utf-8")
    manifest = {
        "config": "maximal-with-kings",
        "n_draws": n_draws, "k": k, "n_rulers": len(RULERS_5), "rulers": RULERS_5,
        "total_frags_per_draw": k * len(RULERS_5),
        "draw_from": "king_found_only (maximal_keepking)",
        "king_found_per_ruler": {r: int(len(elig[r])) for r in RULERS_5},
        "corpus_path": str(corpus), "seed_base": seed_base,
        "produced_by": "build_maxking_subset.py",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print("=== build_maxking_subset ===")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"draws_matrix: {draws.shape}, per-draw frags = {int(draws[0].sum())}")
    print(f"wrote {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--n_draws", type=int, default=200)
    p.add_argument("--k", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    build(Path(a.corpus), a.n_draws, a.k, a.seed, Path(a.out_dir))
