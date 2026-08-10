"""E4.4b — lens SPECTROSCOPY: read the whole vocabulary axis, not just its poles.

F6 was a 2-point measurement: top-k and bottom-k of the 150k-entry logit-lens
vector. This treats l_t = <u_t, gamma (.) w_hat> as a SPECTRUM: sort the whole
vocabulary along the direction, cut into B rank-deciles, and measure the
CATEGORY COMPOSITION of every decile — temporal-ancient, temporal-modern,
year-numeral, numeral, function, capitalized-name-like, junk. A monotone
category gradient across all ten buckets is a far stronger signature of a time
axis than two clean poles (poles can be flukes of a handful of tokens).

Three instruments bundled:
  1. composition-per-bucket curves, for the raw inner product AND the
     cosine variant l_t / ||u_t|| (removes the "loud on every axis" confound
     of large unembedding rows);
  2. the year-token order test: every token parsing as a calendar year is
     scattered value-vs-l_t; Spearman rho(year value, l) is a single-number,
     poles-free test of temporal calibration;
  3. per-bucket nulls: the identical pipeline on N random unit directions
     gives each bucket x category a null mean/sd, so every real-direction cell
     gets a z-score. This is strictly more sensitive than F14's pole check —
     a mild temporal enrichment in buckets 2-3/8-9 that never reaches top-k
     becomes detectable. It is the honest instrument for the open w_doc
     question ("indistinguishable from random AT THE POLES" != meaningless).

The caveat that stays attached to any null result: the lens is one-directional.
Junk for a mid-stack direction means THIS INSTRUMENT CANNOT READ IT, not that
the direction is meaningless.

    python lens_spectroscopy.py --method olmo2_7b
    python lens_spectroscopy.py --selftest        # synthetic vocab, no model

Writes results/spectroscopy.{method}.json + results/figs/spectroscopy.{method}.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from logit_lens import (AKK_ACTS, DIRS_A, DIRS_PAIR, ENTITY,   # noqa: E402
                        load_unembed)

RESULTS = os.path.join(_HERE, "results")
B = 10
N_NULL = 50

ANCIENT = ("bc", "bce", "ancient", "antiqu", "archaic", "medieval",
           "prehistor", "neolithic", "bronze", "pharao", "dynast", "babylon",
           "assyria", "sumer", "rome", "roman", "greek", "athen", "temple",
           "公元前", "古代", "战国", "古人", "王朝", "上古")
MODERN = ("modern", "contemporary", "today", "current", "recent", "digital",
          "internet", "online", "smartphone", "electr", "technolog",
          "现代", "当代", "科技")
FUNCTION = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "was",
            "for", "on", "with", "as", "at", "by", "it", "be", "that", "this"}


def classify(token: str) -> str:
    t = token.replace("Ġ", "").replace("▁", "").strip()
    low = t.lower()
    if not t or not any(c.isalnum() for c in t):
        return "junk"
    if t.isdecimal():        # NOT isdigit(): '²' passes isdigit but breaks int()
        v = int(t)
        if 500 <= v <= 2029 and len(t) in (3, 4):
            return "year_numeral"
        return "numeral"
    if any(k in low for k in ANCIENT):
        return "temporal_ancient"
    if any(k in low for k in MODERN):
        return "temporal_modern"
    if low in FUNCTION:
        return "function"
    if t[0].isupper() and t.isalpha() and len(t) >= 3:
        return "capitalized"
    if len(t) <= 2:
        return "fragment"
    return "other"


CATS = ["temporal_ancient", "temporal_modern", "year_numeral", "numeral",
        "function", "capitalized", "fragment", "junk", "other"]


def spectrum(logits, cats, years_mask, years_vals):
    """Bucket composition + year-order rho for one score vector."""
    order = np.argsort(logits)
    buckets = np.array_split(order, B)
    comp = np.zeros((B, len(CATS)))
    for b, idx in enumerate(buckets):
        c = cats[idx]
        for ci, name in enumerate(CATS):
            comp[b, ci] = (c == ci).mean()
    if years_mask.sum() >= 10:
        rho = float(stats.spearmanr(years_vals[years_mask],
                                    logits[years_mask]).correlation)
    else:
        rho = float("nan")
    return comp, rho


def analyse(v, W_U, norm, cats, years_mask, years_vals, u_norms):
    v = v / (np.linalg.norm(v) + 1e-8)
    raw = W_U @ (norm * v)
    cos = raw / u_norms
    out = {}
    for tag, sc in (("raw", raw), ("cos", cos)):
        comp, rho = spectrum(sc, cats, years_mask, years_vals)
        out[tag] = {"composition": comp, "rho_year_tokens": rho, "scores": sc}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None)
    ap.add_argument("--variant", default="akk_maximal")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        rng = np.random.default_rng(0)
        V, d = 5000, 64
        W_U = rng.standard_normal((V, d)).astype(np.float32)
        norm = np.ones(d, np.float32)
        tokens = [f"tok{i}" for i in range(V)]
        # plant year tokens ordered along axis 0
        for i, y in enumerate(range(1500, 2000, 5)):
            tokens[i] = str(y)
            W_U[i] = 0.02 * (y - 1750) * np.eye(d)[0] + 0.1 * W_U[i]
        dirs = {"planted": np.eye(d)[0].astype(np.float32)}
        method = "selftest"
    else:
        method = args.method
        tok, W_U, norm = load_unembed(method)
        tokens = tok.convert_ids_to_tokens(list(range(W_U.shape[0])))
        # Qwen's vocab has unassigned ids -> None entries; treat as junk
        tokens = ["" if t is None else t for t in tokens]
        dirs = {}
        g = sorted(glob.glob(os.path.join(DIRS_A, method,
                                          f"{ENTITY}.*.layer*.npz")))
        if g:
            dirs["cellA"] = np.load(g[0])["coef"].astype(np.float32).ravel()
        for p in sorted(glob.glob(os.path.join(
                DIRS_PAIR, f"{method}.{args.variant}.mean.layer*.npz"))):
            L = int(re.search(r"layer(\d+)\.npz$", p).group(1))
            lay = os.path.join(AKK_ACTS, method, args.variant,
                               f"mean.layer{L}.npz")
            if os.path.exists(lay):
                sd = np.load(lay)["acts"].astype(np.float32).std(0) + 1e-8
                dirs["pairwise_doc"] = np.load(p)["w"].astype(np.float32) / sd
                break
        if not dirs:
            sys.exit("no directions found")

    cats = np.array([CATS.index(classify(t)) for t in tokens])
    yvals = np.full(len(tokens), np.nan)
    for i, t in enumerate(tokens):
        s = t.replace("Ġ", "").replace("▁", "").strip()
        if s.isdecimal() and len(s) == 4 and 1000 <= int(s) <= 2029:
            yvals[i] = int(s)
    ymask = np.isfinite(yvals)
    u_norms = np.linalg.norm(W_U * norm, axis=1) + 1e-8
    print(f"[vocab] {len(tokens)} tokens | year tokens: {int(ymask.sum())} | "
          f"ancient-lex hits: {(cats == 0).sum()}", flush=True)

    # null band: N random unit directions through the identical pipeline
    rng = np.random.default_rng(1)
    null_comp = {t: [] for t in ("raw", "cos")}
    null_rho = {t: [] for t in ("raw", "cos")}
    for _ in range(N_NULL):
        v = rng.standard_normal(W_U.shape[1]).astype(np.float32)
        r = analyse(v, W_U, norm, cats, ymask, yvals, u_norms)
        for t in ("raw", "cos"):
            null_comp[t].append(r[t]["composition"])
            null_rho[t].append(r[t]["rho_year_tokens"])
    null_mu = {t: np.mean(null_comp[t], 0) for t in null_comp}
    null_sd = {t: np.std(null_comp[t], 0) + 1e-9 for t in null_comp}

    out = {"method": method, "buckets": B, "n_null": N_NULL, "cats": CATS,
           "directions": {}}
    figs_data = {}
    for name, w in dirs.items():
        r = analyse(w, W_U, norm, cats, ymask, yvals, u_norms)
        rec = {}
        for t in ("raw", "cos"):
            comp = r[t]["composition"]
            z = (comp - null_mu[t]) / null_sd[t]
            nr = np.array(null_rho[t], float)
            rho = r[t]["rho_year_tokens"]
            rec[t] = {
                "composition": comp.round(5).tolist(),
                "z_scores": z.round(2).tolist(),
                "max_abs_z": float(np.abs(z).max()),
                "rho_year_tokens": rho,
                "rho_year_null_sd": float(np.nanstd(nr)),
                "rho_year_z": float((rho - np.nanmean(nr)) / (np.nanstd(nr)
                                                              + 1e-9)),
            }
            print(f"[{name}/{t}] year-token rho={rho:+.3f} "
                  f"(z={rec[t]['rho_year_z']:+.1f}) | max |z| over "
                  f"bucket x category = {rec[t]['max_abs_z']:.1f}", flush=True)
        # qualitative: 12 random tokens per bucket (raw ranking)
        order = np.argsort(r["raw"]["scores"])
        rec["bucket_samples"] = [
            [tokens[i] for i in rng.choice(bk, size=min(12, len(bk)),
                                           replace=False)]
            for bk in np.array_split(order, B)]
        out["directions"][name] = rec
        figs_data[name] = r

    os.makedirs(os.path.join(RESULTS, "figs"), exist_ok=True)
    pth = os.path.join(RESULTS, f"spectroscopy.{method}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # the figure: composition curves per direction (cos variant), temporal +
    # year categories highlighted, null band shaded
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["temporal_ancient", "temporal_modern", "year_numeral", "numeral"]
    fig, axes = plt.subplots(1, len(figs_data), figsize=(6.5 * len(figs_data), 5),
                             squeeze=False)
    for ax, (name, r) in zip(axes[0], figs_data.items()):
        comp = r["cos"]["composition"]
        for cat in show:
            ci = CATS.index(cat)
            mu, sd = null_mu["cos"][:, ci], null_sd["cos"][:, ci]
            ax.fill_between(range(1, B + 1), mu - 2 * sd, mu + 2 * sd,
                            alpha=0.12, color="#888")
            ax.plot(range(1, B + 1), comp[:, ci], marker="o", label=cat)
        ax.set_title(f"{name} (cos ranking)")
        ax.set_xlabel("rank bucket (1 = most negative)")
        ax.set_ylabel("category share")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"lens spectroscopy — {method} · grey = ±2sd random-direction "
                 "null", fontsize=11)
    fig.tight_layout()
    fp = os.path.join(RESULTS, "figs", f"spectroscopy.{method}.png")
    fig.savefig(fp, dpi=180)
    print(f"[done] -> {pth} + {fp}", flush=True)


if __name__ == "__main__":
    main()
