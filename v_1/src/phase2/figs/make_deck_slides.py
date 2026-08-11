# -*- coding: utf-8 -*-
"""Append the phase-2 story as slides at the end of the thesis deck
(v_1/src/stress_tests/results/thesis_story_9.html), in the deck's own
design system (green/paper, Iowan serif, gold activation highlights).
Idempotent: reruns replace the marker-delimited block. Every number and
token is read from committed result files.

    python make_deck_slides.py
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from make_story_html import (J, _byte_decode, circle, esc, line, path, rect,
                             svg_open, txt)

_HERE = os.path.dirname(os.path.abspath(__file__))
_P2 = os.path.abspath(os.path.join(_HERE, ".."))
DECK = os.path.abspath(os.path.join(
    _P2, "..", "stress_tests", "results", "thesis_story_9.html"))

GREEN = "#1a5c3a"
GREEN2 = "#2ea86b"
RED = "#8b1a10"
GOLD = "#c3a94e"
GRAYC = "#8a8f99"

FEAT_LABEL = {44713: "German surnames", 22835: "Western name endings",
              17433: "“X of PLACE” nobility",
              53704: "ancient genealogy", 56768: "Chinese names",
              9763: "Chinese imperial names"}


# --------------------------- charts (deck-toned) ---------------------------
def chart_dissociation():
    sp = pd.read_csv(os.path.join(_P2, "pairs", "results",
                                  "summary_probes.csv"))
    sp = sp[(sp.site == "mean") & (sp.m == 21)]
    inf = {v: J("pairs", "results", "inference", f"{v}.json")["arms"]
           for v in ("akk_maximal", "eng_tier0")}
    LBL = {"olmo2_7b": "OLMo-2 7B", "qwen3_8b": "Qwen3 8B",
           "llama2_7b": "Llama-2 7B", "llama2_13b": "Llama-2 13B",
           "llama2_70b": "Llama-2 70B", "qwen3_1b7": "Qwen3 1.7B",
           "qwen3_32b": "Qwen3 32B", "gpt_oss_120b": "gpt-oss 120B",
           "olmo2_7b_random": "OLMo twin (random)",
           "llama2_13b_random": "13B twin", "llama2_70b_random": "70B twin",
           "llama2_7b_random": "7B twin", "random": "random (qwen init)",
           "tfidf_char": "char n-gram floor"}
    W, PH, ROW = 1040, 54, 21
    n = sp[sp.variant == "akk_maximal"].shape[0]
    H = PH + n * ROW + 40
    s = [svg_open(W, H)]
    for pi, (var, title) in enumerate(
            (("akk_maximal", "Akkadian (raw transliteration)"),
             ("eng_tier0", "English gloss"))):
        X0, XW = 170 + pi * 520, 330
        d = sp[sp.variant == var].sort_values("macro_acc", ascending=False)
        lo, hi = .46, .74
        xm = lambda v: X0 + (v - lo) / (hi - lo) * XW      # noqa: E731
        s.append(txt(X0 + XW / 2, 16, title, 14, "var(--ink)", "middle",
                     700))
        for g in (.5, .55, .6, .65, .7):
            s.append(line(xm(g), PH - 14, xm(g), H - 34,
                          "var(--border-light)"))
            s.append(txt(xm(g), H - 20, f"{g:.2f}", 10, "var(--ink-light)",
                         "middle"))
        floor = float(d[d.method == "tfidf_char"].macro_acc.iloc[0])
        s.append(line(xm(floor), PH - 14, xm(floor), H - 34, RED, 1.5,
                      "5 4"))
        s.append(txt(xm(floor), PH - 20, "surface floor", 10.5, RED,
                     "middle", 700))
        for i, (_, r) in enumerate(d.iterrows()):
            y = PH + i * ROW
            twin = "random" in r.method
            c = GREEN if r.method in ("olmo2_7b", "qwen3_8b") else \
                RED if r.method == "tfidf_char" else GRAYC
            s.append(line(xm(r.macro_acc - r.macro_sd), y,
                          xm(r.macro_acc + r.macro_sd), y, c, 1.3))
            s.append(circle(xm(r.macro_acc), y, 4.4,
                            "#ffffff" if twin else c, c, 1.6))
            s.append(txt(X0 - 8, y + 3.5, LBL.get(r.method, r.method), 11,
                         "var(--ink)" if r.method in
                         ("olmo2_7b", "qwen3_8b", "tfidf_char")
                         else "var(--ink-light)", "end",
                         700 if r.method in ("olmo2_7b", "qwen3_8b")
                         else 400))
            a = inf[var].get(r.method)
            if a and r.method in ("olmo2_7b", "qwen3_8b", "tfidf_char"):
                s.append(txt(xm(r.macro_acc + r.macro_sd) + 5, y + 3.5,
                             f"p={a['permutation']['p_value']:.3g}", 10, c,
                             "start", 700))
    s.append("</svg>")
    return "".join(s)


def chart_lens_tokens():
    TEMPORAL = ("bc", "bce", "ancient", "athen", "公元前", "古代", "战国",
                "古人")
    out = ['<div class="p2lens">']
    for m, name in (("olmo2_7b", "OLMo-2 7B"), ("llama2_7b", "Llama-2 7B"),
                    ("qwen3_8b", "Qwen3 8B")):
        d = J("traces", "results", f"{m}.json")["directions"]
        ck = [k for k in d if k.startswith("cellA")][0]
        pk = [k for k in d if k.startswith("pairwise")][0]
        cell = [_byte_decode(e["token"]).strip() or "␣"
                for e in d[ck]["negative_end"][:9]]
        doc = [_byte_decode(e["token"]).strip() or "␣"
               for e in d[pk]["negative_end"][:4]
               + d[pk]["positive_end"][:4]]
        chips_c = "".join(
            f'<span class="p2chip{" hot" if any(k in t.lower() for k in TEMPORAL) else ""}">{esc(t)}</span>'
            for t in cell)
        chips_d = "".join(f'<span class="p2chip dim">{esc(t)}</span>'
                          for t in doc)
        out.append(
            f'<div><div class="p2lensname">{name}</div>'
            f'<div class="p2lenslab" style="color:{GREEN}">entity year axis '
            f'&middot; early end</div><div class="p2chips">{chips_c}</div>'
            f'<div class="p2lenslab" style="color:var(--ink-light)">document '
            f'order axis &middot; both ends</div>'
            f'<div class="p2chips">{chips_d}</div></div>')
    out.append("</div>")
    return "".join(out)


def chart_spectrum():
    W, H = 1040, 290
    s = [svg_open(W, H)]
    for j, (m, name) in enumerate((("olmo2_7b", "OLMo-2 7B"),
                                   ("llama2_7b", "Llama-2 7B"),
                                   ("qwen3_8b", "Qwen3 8B"))):
        d = J("traces", "results", f"spectroscopy.{m}.json")
        ci = d["cats"].index("temporal_ancient")
        X0, XW, Y0, YH = 70 + j * 340, 270, 40, 180
        xm = lambda b: X0 + (b - 1) / 9 * XW               # noqa: E731
        ym = lambda v: Y0 + YH - v / .55 * YH              # noqa: E731
        s.append(txt(X0 + XW / 2, 22, name, 13.5, "var(--ink)", "middle",
                     700))
        for g in (0, .2, .4):
            s.append(line(X0, ym(g), X0 + XW, ym(g), "var(--border-light)"))
            s.append(txt(X0 - 6, ym(g) + 3.5, f"{g:.1f}", 9.5,
                         "var(--ink-light)", "end"))
        for dname, c in (("pairwise_doc", GRAYC), ("cellA", GREEN)):
            rec = d["directions"][dname]["cos"]
            comp = 100 * np.array(rec["composition"])[:, ci]
            z = np.array(rec["z_scores"])[:, ci]
            pts = [(xm(b + 1), ym(v)) for b, v in enumerate(comp)]
            s.append(path(pts, c, 2))
            for (x, y), zi in zip(pts, z):
                s.append(circle(x, y, 2.8, c))
                if abs(zi) >= 3.35:
                    s.append(circle(x, y, 6, "none", c, 1.5))
                    s.append(txt(x + 9, y + 3, f"z={zi:+.1f}", 10.5, c,
                                 "start", 700))
        s.append(txt(X0, H - 26, "early end", 10, "var(--ink-light)"))
        s.append(txt(X0 + XW, H - 26, "late end", 10, "var(--ink-light)",
                     "end"))
        s.append(txt(X0 + XW / 2, H - 10,
                     "rank decile along the direction", 10,
                     "var(--ink-light)", "middle"))
    s.append("</svg>")
    return "".join(s)


def chart_feature_cards():
    interp = J("sae2", "results", "feature_interp.layer9.json")
    hunt = pd.read_csv(sorted(glob.glob(os.path.join(
        _P2, "sae2", "results", "feature_hunt2.layer*.csv")))[-1])
    hunt = hunt.set_index(hunt.feature.astype(int))
    fh1 = pd.read_csv(os.path.join(_P2, "sae", "results",
                                   "feature_hunt.layer24.csv"))
    f1 = fh1[fh1.feature == 38678].iloc[0]

    def ctx(e):
        c = e["context"].split("<|endoftext|>")[0].strip()
        if ">>" in c and "<<" in c:
            pre, rest = c.split(">>", 1)
            tok, post = rest.split("<<", 1)
            return (f'<div class="p2ctx">{esc(pre)}'
                    f'<span class="p2fire">{esc(tok)}</span>'
                    f'{esc(post)}</div>')
        return f'<div class="p2ctx">{esc(c)}</div>'

    head = (f'<div class="p2feathead"><strong>Feature 38678 &middot; '
            f'Qwen-Scope layer 24</strong> &mdash; the headline entity-time '
            f'feature (F8): fires on <strong>{100*f1.fire_cellA:.0f}%</strong>'
            f' of entity prompts, ρ(strength, death year) = '
            f'<strong>+{f1.rho_year:.2f}</strong>; on documents almost '
            f'never ({100*f1.fire_eng_tier0:.2f}% eng, '
            f'{100*f1.fire_akk_maximal:.2f}% akk).</div>')
    cards = []
    for f in (44713, 17433, 53704, 56768):
        rec = interp["features"][str(f)]
        exs = [e for e in rec["max_activating"]
               if e["context"].split("<|endoftext|>")[0].strip()][:2]
        rho = float(hunt.loc[f, "rho_year"])
        fires = (f'{100*float(hunt.loc[f, "fire_cellA"]):.1f}% ent &middot; '
                 f'{100*float(hunt.loc[f, "fire_eng_tier0_frags"]):.1f}% '
                 f'eng &middot; '
                 f'{100*float(hunt.loc[f, "fire_akk_maximal_frags"]):.1f}% '
                 f'akk')
        cards.append(
            f'<div class="p2card"><div class="p2cardt">{f} &middot; '
            f'“{FEAT_LABEL[f]}” <span style="color:'
            f'{GREEN if rho > 0 else RED}">ρ={rho:+.2f}</span></div>'
            f'{"".join(ctx(e) for e in exs)}'
            f'<div class="p2fires">{fires}</div></div>')
    return head + '<div class="p2cards">' + "".join(cards) + "</div>"


def chart_causality():
    st = J("sae2", "results", "steer.layer9.json")
    hunt = pd.read_csv(sorted(glob.glob(os.path.join(
        _P2, "sae2", "results", "feature_hunt2.layer*.csv")))[-1])
    rho = dict(zip(hunt.feature.astype(int), hunt.rho_year))
    alphas = [-8, -4, -2, 0, 2, 4, 8]
    W, H = 1040, 340
    X0, XW, Y0, YH = 70, 560, 34, 240
    lo, hi = -.75, 1.05
    xm = lambda a: X0 + (a + 8) / 16 * XW                  # noqa: E731
    ym = lambda v: Y0 + YH - (v - lo) / (hi - lo) * YH     # noqa: E731
    s = [svg_open(W, H)]
    for g in (-.5, 0, .5, 1):
        s.append(line(X0, ym(g), X0 + XW, ym(g), "var(--border-light)"))
        s.append(txt(X0 - 8, ym(g) + 3.5, f"{g:+.1f}", 10,
                     "var(--ink-light)", "end"))
    ctrl = np.array([[st["runs"][f"ctrl:{f}"]["amplify"][str(a)]
                      for a in alphas] for f in st["ctrl"]])
    band = "M " + " L ".join(
        f"{xm(a):.1f} {ym(v):.1f}" for a, v in zip(alphas, ctrl.min(0)))
    band += " L " + " L ".join(
        f"{xm(a):.1f} {ym(v):.1f}"
        for a, v in zip(alphas[::-1], ctrl.max(0)[::-1])) + " Z"
    s.append(f'<path d="{band}" fill="{GRAYC}" opacity="0.18"/>')
    s.append(txt(xm(5.0), ym(float(ctrl.max(0)[-1])) - 8,
                 "band of 5 rate-matched control features", 11,
                 "var(--ink-light)"))
    for f in st["treat"]:
        cur = [st["runs"][f"treat:{f}"]["amplify"][str(a)] for a in alphas]
        r = rho.get(int(f), 0)
        c = GREEN if r > 0 else RED
        s.append(path([(xm(a), ym(v)) for a, v in zip(alphas, cur)], c, 2))
        for a, v in zip(alphas, cur):
            s.append(circle(xm(a), ym(v), 3, c))
        dodge = {17433: 18, 53704: -6, 22835: 12}.get(int(f), 0)
        s.append(txt(xm(8) + 10, ym(cur[-1]) + 4 + dodge,
                     f"{FEAT_LABEL.get(int(f), f)}  (ρ={r:+.2f})",
                     11, c, "start", 700))
    for a in alphas:
        s.append(txt(xm(a), Y0 + YH + 16, f"{a:+d}" if a else "0", 10,
                     "var(--ink-light)", "middle"))
    s.append(txt(X0 + XW / 2, H - 6,
                 "clamp strength α (× the feature's own act95), "
                 "at an entity prompt", 11, "var(--ink-light)", "middle"))
    s.append(txt(22, 155, "frozen year read-out (sd)", 11,
                 "var(--ink-light)", "middle")
             .replace('<text', '<text transform="rotate(-90 22 155)"'))
    s.append("</svg>")
    return "".join(s)


def chart_bridge():
    st = J("sae2", "results", "steer.layer9.json")
    rows = []
    for g in ("treat", "ctrl"):
        for f in st[g]:
            b = st["runs"][f"{g}:{f}"]["bridge"]
            rows.append((g, int(f), 100 * b["0"]["fire_last"],
                         100 * b["4"]["fire_last"]))
    W = 1040
    ROW = 27
    H = 46 + len(rows) * ROW + 26
    X0, XW = 320, 620
    xm = lambda v: X0 + v / 38 * XW                        # noqa: E731
    s = [svg_open(W, H)]
    s.append(txt(X0, 20, "% of glosses where the feature fires at the "
                 "READ-OUT token", 11.5, "var(--ink-light)"))
    for i, (g, f, b0, b4) in enumerate(rows):
        y = 38 + i * ROW
        c = GREEN if g == "treat" else GRAYC
        name = FEAT_LABEL.get(f, str(f)) if g == "treat" \
            else f"control {f}"
        s.append(txt(X0 - 10, y + 10, name, 11,
                     "var(--ink)" if g == "treat" else "var(--ink-light)",
                     "end", 700 if g == "treat" else 400))
        s.append(rect(X0, y, max(xm(b0) - X0, 1.2), 7, c, 2, .4))
        s.append(rect(X0, y + 8.5, max(xm(b4) - X0, 1.2), 7, c, 2, 1))
        s.append(txt(xm(max(b0, b4)) + 8, y + 11,
                     f"{b0:.1f} → {b4:.1f}", 10, "var(--ink-light)"))
    s.append(txt(X0, H - 8, "light bar: no clamp · solid bar: feature "
                 "forced ON across the whole mid-document (α=4)", 10.5,
                 "var(--ink-light)"))
    s.append("</svg>")
    return "".join(s)




def chart_orthogonal():
    import json as _j
    rows = []
    for f in glob.glob(os.path.join(_P2, "transfer", "results",
                                    "*.mean.json")):
        d = _j.load(open(f))
        cos = d.get("cosine_vs_pairwise_direction", {})
        cosv = [abs(v["cosine"]) for v in cos.values()
                if isinstance(v, dict) and "cosine" in v]
        rows.append({"m": d["method"], "v": d["variant"],
                     "cos": max(cosv) if cosv else np.nan,
                     "rho": d["frozen"]["spearman"]})
    t = pd.DataFrame(rows)
    t = t[t.m != "olmo2_7b_random"].sort_values(["m", "v"]).reset_index(
        drop=True)
    chance = 1 / np.sqrt(4096)
    W, H = 1040, 300
    s = [svg_open(W, H)]
    X0, XW = 250, 360
    ROW = 38
    xm = lambda v: X0 + v / .03 * XW                       # noqa: E731
    s.append(txt(X0 + XW / 2, 20,
                 "|cos(entity year axis, document order axis)|", 12,
                 "var(--ink)", "middle", 700))
    s.append(rect(X0, 34, xm(chance) - X0, len(t) * ROW, "#eef0f5", 0))
    s.append(txt(xm(chance) + 6, 46,
                 f"chance in d=4096  (1/√d = {chance:.3f})", 10.5,
                 "var(--ink-light)"))
    for i, (_, r) in enumerate(t.iterrows()):
        y = 52 + i * ROW
        c = GREEN if r.m in ("olmo2_7b", "qwen3_8b") else GRAYC
        s.append(txt(X0 - 10, y + 4,
                     f"{r.m.split('_')[0]} · "
                     f"{'akk' if 'akk' in r.v else 'eng'}", 11.5,
                     "var(--ink)", "end", 600))
        s.append(line(X0, y, xm(r.cos), y, c, 2.4))
        s.append(circle(xm(r.cos), y, 5, c))
    X1, X1W = 760, 240
    s.append(txt(X1 + X1W / 2, 20, "frozen-transfer Spearman ρ", 12,
                 "var(--ink)", "middle", 700))
    y0 = 52 + (len(t) - 1) * ROW / 2
    ym2 = lambda v: y0 - v * 90                            # noqa: E731
    s.append(line(X1, ym2(0), X1 + X1W, ym2(0), "var(--ink-light)", 1))
    s.append(txt(X1 - 8, ym2(0) + 4, "0", 10, "var(--ink-light)", "end"))
    s.append(txt(X1 - 8, ym2(1) + 4, "+1", 10, "var(--ink-light)", "end"))
    s.append(txt(X1 - 8, ym2(-1) + 4, "−1", 10, "var(--ink-light)", "end"))
    bw = X1W / len(t) - 8
    for i, (_, r) in enumerate(t.iterrows()):
        c = GREEN if r.m in ("olmo2_7b", "qwen3_8b") else GRAYC
        x = X1 + i * (bw + 8)
        h = abs(r.rho) * 90
        yb = ym2(0) - h if r.rho > 0 else ym2(0)
        s.append(rect(x, yb, bw, max(h, 1), c, 2))
    s.append(txt(X1 + X1W / 2, 52 + len(t) * ROW,
                 "the cell-A year probe, applied frozen to fragments", 10.5,
                 "var(--ink-light)", "middle"))
    s.append("</svg>")
    return "".join(s)


def chart_gating():
    tf1 = J("sae", "results", "token_firing.layer24.json")[
        "median_fired_anywhere"]
    fh1 = pd.read_csv(os.path.join(_P2, "sae", "results",
                                   "feature_hunt.layer24.csv"))
    p2 = J("sae2", "results", "pipeline.json")
    tf2 = {k: v["median_fired_anywhere"] for k, v in p2["step4"].items()}
    last1 = {"cellA_entities": float(fh1.fire_cellA.median()),
             "eng_tier0_frags": float(fh1.fire_eng_tier0.median()),
             "akk_maximal_frags": float(fh1.fire_akk_maximal.median())}
    pops = ["cellA_entities", "eng_tier0_frags", "akk_maximal_frags"]
    plabels = ["entity prompts", "English glosses", "Akkadian"]
    panels = [("last-token firing (read-out position)  ·  Qwen-Scope L24",
               last1),
              ("fired anywhere in the text  ·  Qwen-Scope L24", tf1),
              ("fired anywhere  ·  second dictionary (Karvonen 65k, L9)",
               tf2)]
    W, H = 1040, 300
    s = [svg_open(W, H)]
    for pi, (title, data) in enumerate(panels):
        X0, XW, Y0, YH = 60 + pi * 340, 280, 56, 180
        ym = lambda v: Y0 + YH - v / 100 * YH              # noqa: E731
        s.append(txt(X0 + XW / 2, 24, title, 11, "var(--ink)", "middle",
                     700))
        for g in (0, 50, 100):
            s.append(line(X0, ym(g), X0 + XW, ym(g), "var(--border-light)"))
            s.append(txt(X0 - 6, ym(g) + 3.5, f"{g}", 9.5,
                         "var(--ink-light)", "end"))
        bw = XW / 3 - 26
        cols = [GOLD, GREEN, RED]
        for k, p in enumerate(pops):
            v = 100 * float(data.get(p, np.nan))
            x = X0 + 14 + k * (bw + 26)
            s.append(rect(x, ym(v), bw, max(YH * v / 100, 1.2), cols[k], 3))
            s.append(txt(x + bw / 2, ym(v) - 6,
                         f"{v:.2f}%" if v < 1 else f"{v:.1f}%", 10.5,
                         "var(--ink)", "middle", 700))
            s.append(txt(x + bw / 2, Y0 + YH + 16, plabels[k], 9.5,
                         "var(--ink-light)", "middle"))
    s.append("</svg>")
    return "".join(s)


def chart_decomposition():
    fh1 = pd.read_csv(os.path.join(_P2, "sae", "results",
                                   "feature_hunt.layer24.csv"))
    fh2 = pd.read_csv(sorted(glob.glob(os.path.join(
        _P2, "sae2", "results", "feature_hunt2.layer*.csv")))[-1])
    W, H = 1040, 330
    s = [svg_open(W, H)]
    # (a) cos strip
    X0, XW, Y0, YH = 90, 220, 50, 220
    ym = lambda v: Y0 + YH - v * YH                        # noqa: E731
    s.append(txt(X0 + XW / 2, 22, "no single “year neuron”", 12.5,
                 "var(--ink)", "middle", 700))
    for g in (0, .5, 1):
        s.append(line(X0, ym(g), X0 + XW, ym(g), "var(--border-light)"))
        s.append(txt(X0 - 6, ym(g) + 3.5, f"{g:.1f}", 9.5,
                     "var(--ink-light)", "end"))
    s.append(txt(X0 + XW / 2, ym(1) - 8,
                 "|cos| = 1 would be a single dedicated neuron", 10,
                 "var(--ink-light)", "middle"))
    rng = np.random.default_rng(0)
    for k, (fh, c) in enumerate(((fh1, GREEN), (fh2, GOLD))):
        for v in fh.cos_ridge.abs():
            x = X0 + 55 + k * 110 + rng.uniform(-22, 22)
            s.append(circle(x, ym(float(v)), 3.2, c, "none", 0))
    s.append(txt(X0 + 55, Y0 + YH + 16, "Qwen-Scope L24", 10,
                 "var(--ink-light)", "middle"))
    s.append(txt(X0 + 165, Y0 + YH + 16, "Karvonen 65k L9", 10,
                 "var(--ink-light)", "middle"))
    s.append(txt(30, 160, "|cos(feature decoder, year direction)|", 10.5,
                 "var(--ink-light)", "middle")
             .replace('<text', '<text transform="rotate(-90 30 160)"'))
    # (b) hunt scatter
    X1, X1W, Y1, Y1H = 430, 540, 50, 220
    xm2 = lambda v: X1 + v * X1W                           # noqa: E731
    ym2 = lambda v: Y1 + Y1H / 2 - v / .7 * (Y1H / 2)      # noqa: E731
    s.append(txt(X1 + X1W / 2, 22,
                 "the features the hunt found on the year probe's "
                 "population", 12.5, "var(--ink)", "middle", 700))
    s.append(line(X1, ym2(0), X1 + X1W, ym2(0), "var(--ink-light)", 1))
    for g in (-.4, -.2, .2, .4, .6):
        s.append(line(X1, ym2(g), X1 + X1W, ym2(g), "var(--border-light)"))
        s.append(txt(X1 - 6, ym2(g) + 3.5, f"{g:+.1f}", 9.5,
                     "var(--ink-light)", "end"))
    for fh, c, mk in ((fh1, GREEN, "c"), (fh2, GOLD, "r")):
        for _, r in fh.iterrows():
            x, y = xm2(float(r.fire_cellA)), ym2(float(r.rho_year))
            if mk == "c":
                s.append(circle(x, y, 3.6, c, "#ffffff", .8))
            else:
                s.append(rect(x - 3, y - 3, 6, 6, c, 1))
    f38 = fh1[fh1.feature == 38678].iloc[0]
    s.append(circle(xm2(float(f38.fire_cellA)), ym2(float(f38.rho_year)),
                    7.5, "none", "var(--ink)", 1.6))
    s.append(txt(xm2(float(f38.fire_cellA)) + 12,
                 ym2(float(f38.rho_year)) - 8,
                 "38678 — the entity-time feature (62% fire, ρ=+.57)", 11,
                 "var(--ink)", "start", 700))
    for f, lab, dx, dy in ((44713, "German surnames", 8, -10),
                           (17433, "“X of PLACE” nobility", 8, 14),
                           (9763, "Chinese imperial names", 10, 16)):
        r = fh2[fh2.feature == f]
        if len(r):
            s.append(txt(xm2(float(r.fire_cellA.iloc[0])) + dx,
                         ym2(float(r.rho_year.iloc[0])) + dy, lab, 10.5,
                         "#7c5e00", "start", 700))
    s.append(txt(X1 + X1W / 2, Y1 + Y1H + 26,
                 "firing rate on entity prompts", 10.5, "var(--ink-light)",
                 "middle"))
    s.append(txt(392, 160, "ρ(feature strength, death year)", 10.5,
                 "var(--ink-light)", "middle")
             .replace('<text', '<text transform="rotate(-90 392 160)"'))
    s.append("</svg>")
    return "".join(s)


# ----------------------------- slide assembly ------------------------------
STYLE = '''<style>/* phase-2 additions */
.p2chart{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden;margin:2px 0 10px;}
.p2chart svg{max-height:100%;}
.p2lens{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;flex:1;min-height:0;align-content:start;}
.p2lensname{font-family:var(--serif);font-size:16px;font-weight:700;color:var(--ink);margin-bottom:6px;}
.p2lenslab{font-size:10.5px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin:8px 0 5px;}
.p2chips{display:flex;flex-wrap:wrap;gap:4px;}
.p2chip{background:#f3f4f7;border-radius:5px;padding:1px 7px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;color:var(--ink-light);}
.p2chip.hot{background:#faf1d4;color:#7c5e00;font-weight:800;border:1px solid #e3d7a8;}
.p2chip.dim{opacity:.8;}
.p2feathead{background:var(--green-bg);border-left:4px solid var(--green);border-radius:7px;padding:10px 16px;font-size:14.5px;line-height:1.5;color:var(--ink);margin-bottom:12px;}
.p2cards{display:grid;grid-template-columns:1fr 1fr;gap:11px;flex:1;min-height:0;align-content:start;}
.p2card{background:#f8f9fb;border:1px solid var(--border);border-radius:9px;padding:11px 14px;}
.p2cardt{font-family:var(--serif);font-size:15px;font-weight:700;color:var(--ink);margin-bottom:6px;}
.p2ctx{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;background:#fff;border:1px solid var(--border-light);border-radius:5px;padding:3px 8px;margin:4px 0;color:var(--ink-light);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.p2fire{background:#c3a94e;color:#fff;border-radius:3px;padding:0 4px;font-weight:800;}
.p2fires{font-size:11.5px;color:var(--ink-light);margin-top:5px;font-variant-numeric:tabular-nums;}
</style>'''


def slide(idx, eyebrow, headline, body, takeaway=None, note=None):
    tk = (f'<div class="takeaway"><span class="tk-label">Takeaway</span>'
          f'{takeaway}</div>') if takeaway else ""
    nt = f'<div class="fig-note">{note}</div>' if note else ""
    return (f'<section class="slide" data-index="{idx}">\n'
            f'  <div class="eyebrow">{eyebrow}</div>\n'
            f'  <h2 class="sh">{headline}</h2>\n'
            f'  {body}\n{nt}{tk}\n</section>\n')


TITLES_NEW = [
    "Phase 2: why does it collapse at the entity-to-document boundary?",
    "Phase 2 method, data and configuration",
    "Ordering fragments: no model beats the surface floor in Akkadian",
    "The entity year axis and the document axis are different, orthogonal directions",
    "The year direction literally reads 'ancient'; the document direction reads nothing",
    "Across the whole vocabulary: ancient words pile up in the entity axis's first decile only",
    "Where the year features fire: entity-gated, alive mid-text, silent at the read-out",
    "No single year neuron: a distributed code, and the features that carry it",
    "Decomposing the year signal into SAE features: name-culture detectors, not a concept of time",
    "Causal test: pushing a name-culture feature drags the year prediction, controls stay flat",
    "And why it never reaches documents: even forced on, the signal dies before the read-out",
]


def main():
    deck = open(DECK, encoding="utf-8").read()
    # idempotent: strip any previous phase-2 block + title patch
    deck = re.sub(r"<!-- PHASE2-BEGIN -->.*?<!-- PHASE2-END -->\n?", "",
                  deck, flags=re.S)
    deck = re.sub(r"/\*P2TITLES\*/.*?/\*P2TITLES-END\*/\n?", "", deck,
                  flags=re.S)
    base_total = len(re.findall(r'<section class="slide', deck))

    s33 = slide(
        base_total, "Phase 2 · the mechanistic program · F1–F27",
        "Phase 2: the deck ends where a linear world model ends — 27"
        " experiments ask <em>why</em> it ends there",
        '''<div class="text-points">
  <div class="tp"><div class="tp-h">Reframe as ordering (E1, E8)</div>
  <div class="tp-b">628k fragment pairs, "which was composed earlier?",
  quota-balanced per ruler pair, permutation tests that shuffle whole
  kings. Kills the "regression format" and "label leakage" explanations.</div></div>
  <div class="tp"><div class="tp-h">Transfer the axis (E3)</div>
  <div class="tp-b">Freeze the entity year direction, apply it to fragments:
  ρ ≈ 0, and cos(entity axis, document axis) ≈ .01 — chance level in
  d=4096. The two "times" are different directions.</div></div>
  <div class="tp"><div class="tp-h">Decompose (F6–F8, F21–F25)</div>
  <div class="tp-b">Logit-lens the directions; split representations into
  sparse-autoencoder features in <strong>two independent dictionaries</strong>;
  interpret the features by their max-activating contexts.</div></div>
  <div class="tp"><div class="tp-h">Intervene (F23, F26)</div>
  <div class="tp-b">Clamp features with firing-rate-matched controls: does the
  year prediction move? Can the signal be forced into a document?</div></div>
  <div class="tp"><div class="tp-h">Exhaust the alternatives (F15–F19, F27)</div>
  <div class="tp-b">Length, find-spot, pooling choice, seriation, kernel and MLP
  probes — every "document time" candidate decomposes into text form.</div></div>
</div>''',
        "Same corpus, same probes, one new question: not <em>whether</em> the"
        " model dates documents (it doesn't), but <em>what exists instead</em>"
        " — mapped down to individual features."
    )
    s33b = slide(
        base_total + 1, "Phase 2 · method, data, configuration",
        "Where each method comes from, and exactly what it runs on",
        '''<div class="cfg tight">
  <div class="cfg-k">Paradigm</div><div class="cfg-v"><strong>Gurnee &amp; Tegmark 2023</strong> (linear probes on residual activations) is the frame the whole deck tests; phase 2 adds <strong>El-Shangiti et al., NAACL 2025</strong> (activation steering at name tokens), <strong>Belrose et al. 2023</strong> (LEACE concept erasure + the tuned-lens caveat), <strong>Atkins et al. 1998</strong> (spectral seriation), <strong>Templeton et al. 2024</strong> (Golden-Gate feature clamping) and two independent SAE dictionaries — <strong>Qwen-Scope</strong> (64k, layer 24) and <strong>Karvonen's batch-TopK 65k</strong> (layer 9, the only layer of that release passing our FVU ≤ .35 reconstruction gate).</div>
  <div class="cfg-k">Data</div><div class="cfg-v"><strong>Cell A:</strong> 7,507 historical figures (death year), name prompts, held-out test split. <strong>Cells B'/C:</strong> the same 1,187 dated Assyrian royal inscriptions as the deck (40 rulers, 777 eligible ruler pairs) in two variants — <code>akk_maximal</code> (raw transliteration, royal names removed) and <code>eng_tier0</code> (literal English gloss). Genre is constant (all royal inscriptions) — checked, so it cannot confound.</div>
  <div class="cfg-k">Pooling</div><div class="cfg-v">Fragments: <strong>mean pooling</strong> over tokens at the layer selected once in F1 (last-token also swept — F19 shows the trained-vs-twin gap is pooling-dependent and reports it). Entities: <strong>last-token</strong> of the name, the probe's own convention. Year targets standardized per train fold; probe read-outs are in sd-of-death-year units.</div>
  <div class="cfg-k">Protocol</div><div class="cfg-v">Pairwise: quota m=21 pairs per ruler pair per draw, weights 1/m, macro over ruler pairs, <strong>both-rulers-held-out</strong> folds, 100 Monte-Carlo draws. Inference: permutation with full refit at ruler level (B=150) + dyadic bootstrap (Snijders &amp; Borgatti 1999). Interventions: every treated feature has a <strong>firing-rate-matched random control</strong>; claims are treated-minus-control.</div>
  <div class="cfg-k">Controls</div><div class="cfg-v">Random-weight twins of each architecture, the char n-gram floor, random directions for every lens/spectroscopy read, pre-registered decision rules written before each job ran.</div>
</div>''',
        None)
    s34 = slide(
        base_total + 2, "E1 + E8 · pairwise ordering · ruler-level permutation",
        "Ordering fragments: untrained twins top the Akkadian board; only"
        " trained models are significant in English",
        f'<div class="p2chart">{chart_dissociation()}</div>',
        "Left: random-weight twins (hollow) sit at the top and everything"
        " hugs the char n-gram floor — the “order” is text form, not"
        " knowledge. Right: only trained OLMo and Qwen are significant"
        " (p=.0066, permuting whole kings) while the floor itself is not"
        " (p=.11). This small English signal is what phase 2 dissects.")
    s34b = slide(
        base_total + 3, "E3 · frozen transfer + direction geometry",
        "The entity year axis and the document order axis are two different"
        " directions — orthogonal at chance level",
        f'<div class="p2chart">{chart_orthogonal()}</div>',
        "Freeze the cell-A year direction and apply it to fragments:"
        " Spearman ρ ≈ 0 in every arm (right). And directly:"
        " |cos| between the two learned directions sits inside the"
        " 1/√d ≈ .016 chance band (left) — the document axis is not a"
        " tilted or diluted copy of the entity axis; it is a different"
        " axis altogether. The positive control (same code path on cell A)"
        " reproduces ρ=.87–.89, so the null is real.")
    s35 = slide(
        base_total + 4, "F6 · logit lens on the probe directions",
        "Project each direction onto the vocabulary: the entity year axis"
        " points at ancient-time words, the document axis at junk",
        f'<div class="p2chart" style="align-items:flex-start">{chart_lens_tokens()}</div>',
        "Gold chips: the entity axis's early end reads Ancient / BCE / BC in"
        " every model — in Qwen even in Chinese (公元前 “BCE”, 古代"
        " “ancient”, 战国 “Warring States”). The direction trained to"
        " order documents projects onto nothing temporal at either end.")
    s36 = slide(
        base_total + 5, "F21 · whole-vocabulary spectroscopy · 50 random-direction nulls",
        "Not just the extremes: rank all ~150k tokens along each direction"
        " — ancient vocabulary concentrates in the entity axis's first"
        " decile only",
        f'<div class="p2chart">{chart_spectrum()}</div>',
        "Green: entity year axis — the share of ancient-temporal tokens"
        " spikes in decile 1 in all three models (○ = |z| ≥ 3.35,"
        " Bonferroni, vs 50 random directions). Gray: the document axis"
        " never leaves the noise in any decile.")
    s36b = slide(
        base_total + 6, "F8 + F11 + F22 · where the year features fire",
        "The year features are entity-gated: alive inside English text,"
        " silent at the read-out, and (in a sparse basis) never engaging"
        " Akkadian",
        f'<div class="p2chart">{chart_gating()}</div>',
        "Median firing of the top-50 year-correlated features. Left: at the"
        " read-out token they fire on entities (11.7%) and essentially"
        " never on documents (0.08% eng). Middle: anywhere in the text they"
        " DO fire inside English glosses (14.9%) — the signal exists"
        " mid-document but does not propagate. Right: an independently"
        " trained second dictionary replicates both halves (35.5% eng,"
        " 2.0% akk fired-anywhere).")
    s36c = slide(
        base_total + 7, "F8 + F22 · decomposing the year probe",
        "No single year neuron: the year direction is a distributed code —"
        " and these are the features that carry it",
        f'<div class="p2chart">{chart_decomposition()}</div>',
        "Left: the overlap of every hunted feature with the ridge year"
        " direction — all |cos| ≤ .12 in both dictionaries; the axis is"
        " spread across many features. Right: each dot is one feature"
        " (firing rate × year correlation); 38678 is the headline"
        " entity-time feature, and the labeled ones are the name-culture"
        " detectors of the next slide.")
    s37 = slide(
        base_total + 8, "F8 + F25 · SAE decomposition · max-activating contexts",
        "The year-correlated features are name-culture detectors — the"
        " “time” the model knows about people is onomastics",
        chart_feature_cards(),
        "Each card: a real SAE feature, the contexts that fire it hardest"
        " (firing token in gold), and its firing rates. German surname"
        " endings, “X of England” nobility, ancient genealogy formulas,"
        " Chinese names — naming culture tracks era, and that is the"
        " correlation the probe reads.")
    s38 = slide(
        base_total + 9, "F23 · causal interventions · rate-matched controls",
        "Clamp one feature at an entity prompt and the frozen year"
        " prediction moves — monotonically, in the sign of that feature's"
        " correlation",
        f'<div class="p2chart">{chart_causality()}</div>',
        "Each colored line is one name-culture feature; the gray band is"
        " five control features matched on firing rate but uncorrelated"
        " with year. Pushing “Chinese names” (ρ&lt;0) drags the read-out"
        " earlier by up to 0.6 sd; “German surnames” (ρ&gt;0) later."
        " Correlation became causation — for the features, not for a time"
        " concept. (Side effect: clamping the German-surname feature during"
        " free generation makes the model write German.)")
    s39 = slide(
        base_total + 10, "F23 bridge + F26 ignition · the causal seal",
        "Force the features ON across a whole document — they still never"
        " fire at the read-out token. The entity–document gap is a"
        " disconnection, not missing knowledge",
        f'<div class="p2chart">{chart_bridge()}</div>',
        "Per feature, two bars: natural state vs clamped ON over the whole"
        " mid-document. Every row reads x → x: nothing propagates to the"
        " token the probe reads — treated and control alike. F26 also"
        " ignited the features exactly at the ruler's name inside glosses:"
        " flat against controls. The model has a real, semantic,"
        " causally-usable entity time axis built from onomastic features"
        " — and no channel that carries any of it into a document"
        " representation, even by force.")

    block = ("<!-- PHASE2-BEGIN -->\n" + STYLE + "\n"
             + s33 + s33b + s34 + s34b + s35 + s36 + s36b + s36c
             + s37 + s38 + s39
             + "<!-- PHASE2-END -->\n")
    deck = deck.replace("<script>\nconst TOTAL", block + "<script>\nconst TOTAL")
    new_total = base_total + 11
    deck = re.sub(r"const TOTAL = \d+;", f"const TOTAL = {new_total};",
                  deck, count=1)
    titles_js = ("/*P2TITLES*/TITLES.push("
                 + ",".join('"' + t.replace('"', '\\"') + '"'
                            for t in TITLES_NEW)
                 + ");/*P2TITLES-END*/\n")
    deck = deck.replace("let cur = 0;", titles_js + "let cur = 0;")
    with open(DECK, "w", encoding="utf-8") as f:
        f.write(deck)
    print(f"[done] {base_total} -> {new_total} slides; deck "
          f"{len(deck)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
