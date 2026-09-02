"""make_ssl_corpus.py — S0 of PLAN_SCALE_SSL: one text table over EVERY Akkadian
corpus on disk, for self-supervised pretraining and period probing.

Sources and what each contributes
  unified_corpus.parquet   ORACC + eBL + Archibab, word rows -> texts (value_raw
                           joined in (line_num, word_idx) order). Exact duplicate
                           rows are dropped FIRST: ORACC carries 574,744 repeated
                           word rows (DATA-3), so the naive word count is 30% high.
  3-period letters         adds the 'lbl' (Late Babylonian letters) source that
                           the unified corpus lacks, and PERIOD for ~5k letters.
  ORACC 1st-millennium     PERIOD (6 classes) + genre for 3,775 ORACC texts.
  SEAL                     384 literary texts (not in unified), PERIOD (10), genre.
  ORCC royal inscriptions  1,202 texts (only 52% are in unified), YEAR for 1,193,
                           ruler, period, object type, find-spot.

Then: content-hash dedupe across sources (keep the row with the richest labels),
minimum length, harmonised period labels, and tablet-level splits stratified by
source (80/10/10). Output: chrono/artifacts_ssl/corpus_all.parquet + CENSUS.md.

    python chrono/scripts/make_ssl_corpus.py --out-dir chrono/artifacts_ssl
"""
from __future__ import annotations
import argparse, hashlib, os, re, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "v_1", "data")
EV = os.path.join(DATA, "evaluation", "corpora")

PERIOD_MAP = {  # harmonise the different spellings to one coarse label set
    "neo-assyrian": "Neo-Assyrian", "neo-babylonian": "Neo-Babylonian",
    "late babylonian": "Late Babylonian", "neo or late babylonian": "Late Babylonian",
    "old babylonian": "Old Babylonian", "old assyrian": "Old Assyrian",
    "middle babylonian": "Middle Babylonian", "middle assyrian": "Middle Assyrian",
    "middle babylonian/assyrian": "Middle Babylonian",
    "achaemenid": "Achaemenid", "hellenistic": "Hellenistic",
}


def norm_period(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = re.sub(r"\s*\(.*?\)\s*", " ", str(x)).replace("?", "").strip().lower()
    for k, v in PERIOD_MAP.items():
        if s.startswith(k):
            return v
    return None


def sha(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text.strip()).encode()).hexdigest()[:16]


def load_unified():
    d = pd.read_parquet(os.path.join(DATA, "unified", "unified_corpus.parquet"),
                        columns=["source", "fragment_id", "line_num", "word_idx", "value_raw",
                                 "value_signs", "place_discovery", "domain"])
    n0 = len(d)
    d = d.drop_duplicates(["source", "fragment_id", "line_num", "word_idx", "value_raw"])
    print(f"[unified] dropped {n0 - len(d):,} exact duplicate word rows (DATA-3)", flush=True)
    d = d.sort_values(["source", "fragment_id", "line_num", "word_idx"])
    g = d.groupby(["source", "fragment_id"], sort=False)
    out = pd.DataFrame({
        "text": g["value_raw"].apply(lambda s: " ".join(map(str, s.dropna()))),
        "text_signs": g["value_signs"].apply(lambda s: " ".join(map(str, s.dropna()))),
        "provenance": g["place_discovery"].first(),
        "genre_raw": g["domain"].first(),
    }).reset_index()
    out["period"] = None; out["year"] = np.nan; out["ruler"] = None; out["sub_genre"] = None
    out["label_source"] = "unified"
    return out


def load_letters():
    l = pd.read_parquet(os.path.join(EV, "unified_3groups_akkadian_letters.parquet"),
                        columns=["fragment_id", "fragment_line_num", "index_in_line", "value",
                                 "period", "corpus_source", "place_discovery"])
    l = l.sort_values(["fragment_id", "fragment_line_num", "index_in_line"])
    g = l.groupby("fragment_id", sort=False)
    out = pd.DataFrame({
        "text": g["value"].apply(lambda s: " ".join(map(str, s.dropna()))),
        "period": g["period"].first(), "source": g["corpus_source"].first(),
        "provenance": g["place_discovery"].first(),
    }).reset_index()
    out["source"] = out["source"].replace({"lbl": "lbl_letters"})
    out["genre_raw"] = "letter"; out["text_signs"] = None; out["year"] = np.nan
    out["ruler"] = None; out["sub_genre"] = None; out["label_source"] = "letters3"
    return out


def load_oracc_1st():
    o = pd.read_parquet(os.path.join(EV, "corpus_b_oracc_1st_mill.parquet"),
                        columns=["fragment_id", "period", "genre"])
    return o.drop_duplicates("fragment_id").rename(columns={"genre": "genre_1st"})


def load_seal():
    s = pd.read_parquet(os.path.join(EV, "seal_corpus.parquet"))
    return pd.DataFrame({
        "source": "seal", "fragment_id": s["fragment_id"].astype(str),
        "text": s["text_tier0"].fillna(s["text"]).astype(str), "text_signs": None,
        "period": s["period"], "genre_raw": s["genre"], "sub_genre": s["sub_genre"],
        "provenance": s["provenance"], "year": np.nan, "ruler": None, "label_source": "seal",
    })


def load_orcc():
    o = pd.read_parquet(os.path.join(EV, "orcc_corpus.parquet"))
    return pd.DataFrame({
        "source": "orcc", "fragment_id": o["fragment_id"].astype(str),
        "text": o["text_tier0"].fillna("").astype(str), "text_signs": None,
        "period": o["period"], "genre_raw": o["genre"], "sub_genre": o["sub_genre"],
        "provenance": o["provenance"], "year": o["year"].astype(float), "ruler": o["ruler"],
        "label_source": "orcc",
    })


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(REPO, "chrono", "artifacts_ssl"))
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    uni, let, seal, orcc = load_unified(), load_letters(), load_seal(), load_orcc()
    o1 = load_oracc_1st()
    # attach ORACC-1st-mill period/genre onto unified rows by fragment id
    uni = uni.merge(o1, on="fragment_id", how="left")
    uni["period"] = uni["period_y"].where(uni["period_y"].notna(), uni["period_x"])
    uni = uni.drop(columns=["period_x", "period_y"])
    uni.loc[uni["genre_1st"].notna(), "genre_raw"] = uni["genre_1st"]
    uni = uni.drop(columns=["genre_1st"])

    frames = [uni, let, seal, orcc]
    all_ = pd.concat(frames, ignore_index=True, sort=False)
    all_["text"] = all_["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    all_["n_words"] = all_["text"].str.split().str.len().fillna(0).astype(int)
    all_["period_norm"] = all_["period"].map(norm_period)
    all_["hash"] = all_["text"].map(sha)

    n_before = len(all_)
    # richness = how many labels a row carries; keep the richest copy of a duplicate text
    all_["richness"] = (all_["year"].notna().astype(int) * 4 + all_["period_norm"].notna().astype(int) * 2
                        + all_["genre_raw"].notna().astype(int))
    all_ = all_.sort_values(["hash", "richness"], ascending=[True, False])
    dup_groups = all_.groupby("hash")["source"].agg(lambda s: ",".join(sorted(set(s))))
    all_ = all_.drop_duplicates("hash", keep="first")
    cross = dup_groups[dup_groups.str.contains(",")]
    n_dedup = n_before - len(all_)
    # the dated benchmark must stay complete: every dated ORCC document is
    # kept whatever its length (the 40-king protocol is evaluated on all of them)
    keep = (all_["n_words"] >= args.min_words) | all_["year"].notna()
    all_ = all_[keep].copy()

    # tablet-level splits stratified by source
    rng = np.random.default_rng(args.seed)
    all_["split"] = "train"
    for src, idx in all_.groupby("source").groups.items():
        idx = np.array(list(idx)); rng.shuffle(idx)
        n = len(idx); nv, nt = int(0.1 * n), int(0.1 * n)
        all_.loc[idx[:nv], "split"] = "val"; all_.loc[idx[nv:nv + nt], "split"] = "test"
    # every dated ORCC document stays evaluable by the 40-king protocol -> keep them all in 'dated'
    all_.loc[all_["year"].notna(), "split"] = "dated"

    cols = ["source", "fragment_id", "text", "text_signs", "n_words", "period", "period_norm",
            "genre_raw", "sub_genre", "provenance", "year", "ruler", "label_source", "hash", "split"]
    all_ = all_[cols].reset_index(drop=True)
    out = os.path.join(args.out_dir, "corpus_all.parquet"); all_.to_parquet(out, index=False)

    # census
    L = [f"# SSL corpus census — {pd.Timestamp.utcnow():%Y-%m-%d}", "",
         f"rows before dedupe {n_before:,} · content duplicates removed {n_dedup:,} "
         f"(cross-source duplicate groups: {len(cross):,}) · min length {args.min_words} words · **final {len(all_):,} texts, "
         f"{all_['n_words'].sum():,} words**", "",
         "| source | texts | words | median words | with period | with year |", "|---|---|---|---|---|---|"]
    for src, g in all_.groupby("source"):
        L.append(f"| {src} | {len(g):,} | {g.n_words.sum():,} | {int(g.n_words.median())} | "
                 f"{g.period_norm.notna().sum():,} | {g.year.notna().sum():,} |")
    L += ["", "## period labels (harmonised)", "", "| period | texts | sources |", "|---|---|---|"]
    for per, g in all_[all_.period_norm.notna()].groupby("period_norm"):
        L.append(f"| {per} | {len(g):,} | {', '.join(f'{s}:{n}' for s, n in g.source.value_counts().items())} |")
    unm = all_[all_.period.notna() & all_.period_norm.isna()].period.value_counts().head(8)
    if len(unm):
        L += ["", "unmapped period strings (top): " + "; ".join(f"`{k}`×{v}" for k, v in unm.items())]
    piv = all_.groupby(["split", "source"]).size().unstack(fill_value=0)
    L += ["", "## splits", "", "| split | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
    L += [f"| {idx} | " + " | ".join(str(int(v)) for v in row) + " |" for idx, row in piv.iterrows()]
    open(os.path.join(args.out_dir, "CENSUS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
