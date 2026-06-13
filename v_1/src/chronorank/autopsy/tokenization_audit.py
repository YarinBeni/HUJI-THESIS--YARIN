"""
Pillar 1 — 1a Tokenization audit (CPU, fast).

Isolates factor (T): does Thalesian/uMT5's tokenizer chop Akkadian into fewer,
more meaningful units than the decoder-only models (Qwen3, gpt-oss)?

MULTI-CORPUS (Yarin, 2026-06-14): the tokenizer audit is purely descriptive, so we
sample across genres/periods, not just ORCC royal inscriptions, to show whether a
tokenizer advantage is *general* or royal-specific:
  - orcc        royal inscriptions, NA / 1st mill   (fragment-level: text_tier0/maximal)
  - seal        literary                            (fragment-level: text_tier0/maximal)
  - letters     OB / NA / LB letters                (fragment-level: full_text)
  - archibab    Old Babylonian, 2nd mill            (word-level value_clean -> rebuilt)
  - oracc_1mill 1st mill                             (word-level value_clean -> rebuilt)
  - ebl         literary / scholarly                (word-level value_clean -> rebuilt)
(1b, the *probing* ladder, stays ORCC-only: it needs the per-fragment `year` label,
 which only ORCC has, and reuses the on-disk maximal-balanced activations/harness.)

Per tokenizer we report, per corpus:
  - fertility  : tokens per whitespace-delimited Akkadian word
  - unk_rate   : fraction of tokens equal to the unk id
  - byte_rate  : fraction of byte-fallback (SP <0x..>) / replacement-char tokens
And once per tokenizer (corpus-independent / ORCC-only):
  - char_probe : tokens-per-character for isolated Akkadian special characters
  - category   : fertility/byte by orthographic bucket on ORCC tier0
                 (logogram / determinative / diacritic / index-number / plain)
  - tier0-vs-maximal detail on ORCC; qualitative side-by-side samples

Note: Thalesian did NOT expand uMT5's vocab (finetune/eda/TOKENIZER_EDA.md), so the
Thalesian tokenizer IS the uMT5 tokenizer; we audit both ids to *prove* equality —
that is why factor (T) is held constant in the 1b Thalesian-vs-uMT5 comparison.

Output: results/tokenization_audit.json, results/tokenization_audit.csv,
        results/figures/fertility_by_corpus.png, results/figures/fertility_hist.png
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
_CORP = _REPO / "v_1/data/evaluation/corpora"
OUT_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = OUT_DIR / "figures"

# Corpus registry: (name, path, kind, spec)
#   kind="fragment"  -> spec = [primary_col, *extra_cols]
#   kind="wordlevel" -> spec = "value_clean" (rebuilt per fragment)
CORPORA = [
    ("orcc",        _CORP / "orcc_corpus.parquet",                  "fragment",  ["text_tier0", "text_maximal"]),
    ("seal",        _CORP / "seal_corpus.parquet",                  "fragment",  ["text_tier0", "text_maximal"]),
    ("letters",     _CORP / "texts_for_evaluation.parquet",         "fragment",  ["full_text"]),
    ("archibab",    _CORP / "corpus_a_archibab_2nd_mill.parquet",   "wordlevel", "value_clean"),
    ("oracc_1mill", _CORP / "corpus_b_oracc_1st_mill.parquet",      "wordlevel", "value_clean"),
    ("ebl",         _REPO / "v_1/data/processed/ebl/ebl_corpus.parquet", "wordlevel", "value_clean"),
]

DEFAULT_TOKENIZERS = [
    ("thalesian_cunei400m", "Thalesian/cuneiformBase-400m"),
    ("umt5_base",           "google/umt5-base"),
    ("qwen3_8b",            "Qwen/Qwen3-8B"),
    ("gpt_oss_120b",        "openai/gpt-oss-120b"),
]

SPECIAL_CHARS = list("šṣṭḫʾʿāēīūâêîûàèìù")
_DET_PREFIX = re.compile(r"^[dmf]-")
_DET_SUFFIX = re.compile(r"-(ki|me[sš])$", re.IGNORECASE)
_HAS_UPPER = re.compile(r"[A-Z]")
_HAS_DIGIT = re.compile(r"[0-9]")
_HAS_SPECIAL = re.compile("[" + "".join(SPECIAL_CHARS) + "]")


def categorize_word(w: str) -> str:
    if _DET_PREFIX.search(w) or _DET_SUFFIX.search(w):
        return "determinative"
    if _HAS_UPPER.search(w):
        return "logogram"
    if _HAS_SPECIAL.search(w):
        return "diacritic"
    if _HAS_DIGIT.search(w):
        return "index_number"
    return "plain"


def is_byte_fallback(tok: str) -> bool:
    t = tok.lstrip("▁").lstrip("Ġ")
    return t.startswith("<0x") or "�" in tok


def load_corpus_texts(name, path, kind, spec, sample, seed):
    """Return (texts: list[str], primary_label, df_or_None).

    df is returned only for the fragment corpus that carries text_tier0/text_maximal
    (ORCC), for the detailed cleaning + category analysis.
    """
    if not Path(path).exists():
        print(f"  [skip] {name}: missing {path}")
        return None, None, None
    if kind == "fragment":
        df = pd.read_parquet(path)
        primary = spec[0]
        texts = [str(t) for t in df[primary].fillna("") if str(t).strip()]
        return texts, primary, (df if "text_maximal" in df.columns else None)
    # wordlevel: dedup (frag,line,word), join the word column per fragment.
    # Corpora differ: oracc/ebl populate value_clean; archibab only value_raw.
    df = pd.read_parquet(path, columns=["fragment_id", "line_num", "word_idx",
                                        "value_clean", "value_raw", "value_signs"])
    if df[spec].notna().sum() == 0:  # chosen column empty -> fall back
        for alt in ("value_raw", "value_signs"):
            if alt in df.columns and df[alt].notna().sum() > 0:
                print(f"  [{name}] '{spec}' all-NaN; falling back to '{alt}'")
                spec = alt
                break
    df = df.drop_duplicates(["fragment_id", "line_num", "word_idx"])
    df = df.dropna(subset=[spec])
    df = df.sort_values(["fragment_id", "line_num", "word_idx"])
    grouped = df.groupby("fragment_id")[spec].apply(lambda s: " ".join(map(str, s)))
    grouped = grouped[grouped.map(lambda x: bool(str(x).strip()))]
    if sample and len(grouped) > sample:
        grouped = grouped.sample(sample, random_state=seed)
    return grouped.tolist(), spec, None


def fertility_stats(tok, texts, unk_id):
    tot_tok = tot_word = tot_unk = tot_byte = 0
    fert = []
    for txt in texts:
        words = txt.split()
        if not words:
            continue
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)
        tot_tok += len(ids)
        tot_word += len(words)
        if unk_id is not None:
            tot_unk += sum(1 for i in ids if i == unk_id)
        tot_byte += sum(1 for t in toks if is_byte_fallback(t))
        fert.append(len(ids) / len(words))
    return {
        "fertility": tot_tok / tot_word if tot_word else float("nan"),
        "unk_rate": tot_unk / tot_tok if tot_tok else float("nan"),
        "byte_rate": tot_byte / tot_tok if tot_tok else float("nan"),
        "n_texts": len(fert), "total_words": tot_word,
    }, np.array(fert)


def audit_tokenizer(short, model_id, corpus_texts, orcc_df):
    from transformers import AutoTokenizer
    print(f"\n=== {short}  ({model_id}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    unk_id = tok.unk_token_id
    rec = {
        "short": short, "model_id": model_id, "vocab_size": int(tok.vocab_size),
        "unk_token": tok.unk_token, "unk_id": unk_id,
        "per_corpus": {}, "orcc_per_cleaning": {}, "category": {},
        "char_probe": {}, "samples": {},
    }
    fert_arrays = {}

    # --- per-corpus fertility/unk/byte on the primary text ------------------
    for cname, (texts, label) in corpus_texts.items():
        m, arr = fertility_stats(tok, texts, unk_id)
        m["primary_col"] = label
        rec["per_corpus"][cname] = m
        fert_arrays[cname] = arr
        print(f"  {cname:12s} ({label:11s}) fert={m['fertility']:.3f} "
              f"unk={m['unk_rate']:.4f} byte={m['byte_rate']:.4f} n={m['n_texts']}", flush=True)

    # --- ORCC tier0-vs-maximal detail --------------------------------------
    if orcc_df is not None:
        for col in ["text_tier0", "text_maximal"]:
            texts = [str(t) for t in orcc_df[col].fillna("") if str(t).strip()]
            m, _ = fertility_stats(tok, texts, unk_id)
            rec["orcc_per_cleaning"][col] = m

        # --- category split on ORCC tier0 ----------------------------------
        cat_tok, cat_word, cat_byte = defaultdict(int), defaultdict(int), defaultdict(int)
        for txt in orcc_df["text_tier0"].fillna(""):
            for w in str(txt).split():
                c = categorize_word(w)
                ids = tok(w, add_special_tokens=False)["input_ids"]
                toks = tok.convert_ids_to_tokens(ids)
                cat_tok[c] += len(ids)
                cat_word[c] += 1
                cat_byte[c] += sum(1 for t in toks if is_byte_fallback(t))
        for c in ["plain", "diacritic", "logogram", "determinative", "index_number"]:
            if cat_word[c]:
                rec["category"][c] = {
                    "n_words": cat_word[c],
                    "fertility": cat_tok[c] / cat_word[c],
                    "byte_rate": cat_byte[c] / cat_tok[c] if cat_tok[c] else 0.0,
                }

        # --- qualitative samples -------------------------------------------
        for cat, w in {"determinative": "d-aš-šur", "logogram": "E2-GAL",
                       "diacritic": "kiš-ša2-ti", "index_number": "GAL-u2",
                       "plain": "dan-nu"}.items():
            ids = tok(w, add_special_tokens=False)["input_ids"]
            rec["samples"][cat] = {"word": w,
                                   "tokens": tok.convert_ids_to_tokens(ids),
                                   "n_tokens": len(ids)}

    # --- char probe (tokenizer-intrinsic) ----------------------------------
    for ch in SPECIAL_CHARS:
        rec["char_probe"][ch] = len(tok(ch, add_special_tokens=False)["input_ids"])
    rec["char_probe_mean"] = float(np.mean([rec["char_probe"][c] for c in SPECIAL_CHARS]))
    return rec, fert_arrays


def make_figures(records, fert_by_tok_corpus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # grouped bar: fertility per corpus per tokenizer
    corpora = list(next(iter(fert_by_tok_corpus.values())).keys())
    shorts = list(fert_by_tok_corpus.keys())
    x = np.arange(len(corpora)); w = 0.8 / max(len(shorts), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, s in enumerate(shorts):
        vals = [np.mean(fert_by_tok_corpus[s][c]) if len(fert_by_tok_corpus[s][c]) else np.nan
                for c in corpora]
        ax.bar(x + i * w, vals, w, label=s)
    ax.set_xticks(x + w * (len(shorts) - 1) / 2)
    ax.set_xticklabels(corpora, rotation=20)
    ax.set_ylabel("tokens per Akkadian word")
    ax.set_title("Tokenizer fertility by corpus (lower = more efficient)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fertility_by_corpus.png", dpi=130)
    print(f"Saved {FIG_DIR/'fertility_by_corpus.png'}")

    # pooled-across-corpora fertility histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 6, 40)
    for s in shorts:
        pooled = np.concatenate([fert_by_tok_corpus[s][c] for c in corpora
                                 if len(fert_by_tok_corpus[s][c])])
        ax.hist(pooled, bins=bins, histtype="step", linewidth=2, label=s, density=True)
    ax.set_xlabel("tokens per Akkadian word (all corpora pooled)")
    ax.set_ylabel("density")
    ax.set_title("Tokenizer fertility distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fertility_hist.png", dpi=130)
    print(f"Saved {FIG_DIR/'fertility_hist.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", default=None,
                    help="comma list short=model_id; default = the 4 audited models")
    ap.add_argument("--sample", type=int, default=2000,
                    help="max fragments to sample from each word-level corpus (speed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpora", default=None,
                    help="comma list of corpus names to include (default = all available)")
    args = ap.parse_args()

    toks = ([tuple(p.split("=", 1)) for p in args.tokenizers.split(",")]
            if args.tokenizers else DEFAULT_TOKENIZERS)
    keep = set(args.corpora.split(",")) if args.corpora else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load all corpora once (shared across tokenizers).
    corpus_texts, orcc_df = {}, None
    for name, path, kind, spec in CORPORA:
        if keep and name not in keep:
            continue
        texts, label, df = load_corpus_texts(name, path, kind, spec, args.sample, args.seed)
        if texts is None:
            continue
        corpus_texts[name] = (texts, label)
        if name == "orcc":
            orcc_df = df
        print(f"[corpus] {name:12s} n_texts={len(texts)} primary={label}")

    records, fert_by_tok_corpus = [], {}
    for short, mid in toks:
        rec, fert_arrays = audit_tokenizer(short, mid, corpus_texts, orcc_df)
        records.append(rec)
        fert_by_tok_corpus[short] = fert_arrays

    by = {r["short"]: r for r in records}
    if "thalesian_cunei400m" in by and "umt5_base" in by:
        a = by["thalesian_cunei400m"]["per_corpus"].get("orcc", {}).get("fertility")
        b = by["umt5_base"]["per_corpus"].get("orcc", {}).get("fertility")
        if a and b:
            print(f"\n[equality] Thalesian ORCC fert={a:.4f} uMT5 ORCC fert={b:.4f} "
                  f"identical={abs(a-b) < 1e-9}")

    with open(OUT_DIR / "tokenization_audit.json", "w") as f:
        json.dump({"corpora": list(corpus_texts), "sample": args.sample,
                   "tokenizers": records}, f, indent=2, ensure_ascii=False)

    rows = []
    for r in records:
        for cname, m in r["per_corpus"].items():
            rows.append({"tokenizer": r["short"], "vocab_size": r["vocab_size"],
                         "corpus": cname, "primary_col": m["primary_col"],
                         "fertility": round(m["fertility"], 4),
                         "unk_rate": round(m["unk_rate"], 5),
                         "byte_rate": round(m["byte_rate"], 5),
                         "char_probe_mean": round(r["char_probe_mean"], 3)})
    dfc = pd.DataFrame(rows)
    dfc.to_csv(OUT_DIR / "tokenization_audit.csv", index=False)
    print(f"\nSaved {OUT_DIR/'tokenization_audit.csv'}")
    print("\n" + dfc.to_string(index=False))

    make_figures(records, fert_by_tok_corpus)


if __name__ == "__main__":
    main()
