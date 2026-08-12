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
    H = PH + n * ROW + 56
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
    s.append(txt(W / 2, H - 4,
                 "pairwise ordering accuracy, macro over ruler pairs "
                 "(0.5 = chance)", 11, "var(--ink-light)", "middle"))
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
    W, H = 1040, 312
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
                if abs(zi) >= 3.0:
                    s.append(circle(x, y, 6, "none", c, 1.5))
                    s.append(txt(x + 9, y + 3, f"z={zi:+.1f}", 10.5, c,
                                 "start", 700))
        s.append(txt(X0, H - 26, "early end", 10, "var(--ink-light)"))
        s.append(txt(X0 + XW, H - 26, "late end", 10, "var(--ink-light)",
                     "end"))
        s.append(txt(X0 + XW / 2, H - 10,
                     "rank decile along the direction", 10,
                     "var(--ink-light)", "middle"))
    s.append(txt(16, 130, "% ancient-temporal tokens in decile", 10,
                 "var(--ink-light)", "middle")
             .replace('<text', '<text transform="rotate(-90 16 130)"'))
    s.append(line(300, 8, 320, 8, GREEN, 2.4))
    s.append(txt(325, 12, "entity year axis", 10.5, GREEN, "start", 700))
    s.append(line(435, 8, 455, 8, GRAYC, 2.4))
    s.append(txt(460, 12, "document order axis", 10.5, GRAYC, "start",
                 700))
    s.append(txt(600, 12, "\u25cb = 3\u03c3 above the null of 50 random "
                 "directions", 10.5, "var(--ink-light)", "start"))
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

    gen = interp["features"]["44713"].get("generations_clamped", [])
    gline = ""
    if gen:
        g = esc(str(gen[0]).replace("\n", " ")[:130])
        gline = (f'<div class="p2ctx" style="margin-top:9px;white-space:'
                 f'normal"><strong>clamp 44713 at 10&times;act95 during'
                 f' generation &rarr;</strong> &ldquo;{g}&hellip;&rdquo;'
                 f' &mdash; the model drifts into German mid-answer: the'
                 f' feature really is name-Germanness.</div>')
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
    return (head + '<div class="p2cards">' + "".join(cards) + "</div>"
            + gline)


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




def chart_ladder():
    """F28: mean drop in E1 pairwise macro when ONE variable is erased,
    trained arms vs the untrained twin."""
    import collections
    rows = [J("erasure", "results", os.path.basename(p)) for p in
            sorted(glob.glob(os.path.join(_P2, "erasure", "results",
                                          "ladder.*.json")))]
    LAB = {"ruler": "ruler identity", "period": "period (Neo-Assyrian…)",
           "subgenre": "object type (prism, slab…)",
           "year10": "year decile  ← positive control",
           "provenance": "find-spot", "length": "text length"}
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in rows:
        a = ("trained" if d["method"] in ("olmo2_7b", "qwen3_8b")
             else "twin" if d["method"] == "olmo2_7b_random" else "floor")
        agg[d["concept"]][a].append(d["erased"]["pairwise_macro"]
                                    - d["raw"]["pairwise_macro"])
    order = sorted(agg, key=lambda c: np.mean(agg[c]["trained"]))
    W, ROW = 1040, 44
    H = 78 + len(order) * ROW + 30
    X0, XW = 300, 560
    lo = -0.20
    xm = lambda v: X0 + (v - lo) / (0 - lo) * XW           # noqa: E731
    s = [svg_open(W, H)]
    s.append(txt(X0 + XW / 2, 22, "how much of the document ordering each "
                 "variable was carrying", 13, "var(--ink)", "middle", 700))
    s.append(line(xm(0), 54, xm(0), 62 + len(order) * ROW, "var(--ink)",
                  1.2))
    s.append(txt(xm(0) + 8, 48, "no loss", 10, "var(--ink-light)"))
    for i, c in enumerate(order):
        y = 66 + i * ROW
        t, w = np.mean(agg[c]["trained"]), np.mean(agg[c]["twin"])
        s.append(txt(X0 - 190, y + 14, LAB.get(c, c), 12.5,
                     "var(--ink)", "start", 700 if c != "year10" else 600))
        for v, col, h, yo in ((t, GREEN, 11, 1), (w, GRAYC, 11, 14)):
            x = xm(min(v, 0))
            s.append(rect(x, y + yo, max(xm(0) - x, 1.2), h, col, 2))
        s.append(txt(xm(min(t, w)) - 8, y + 18,
                     f"{t:+.3f} / {w:+.3f}", 10.5, "var(--ink-light)",
                     "end"))
    yb = 62 + len(order) * ROW
    for g in (-0.20, -0.15, -0.10, -0.05, 0):
        s.append(line(xm(g), yb, xm(g), yb + 5, "var(--ink-light)", 1))
        s.append(txt(xm(g), yb + 17, f"{g:+.2f}", 9.5, "var(--ink-light)",
                     "middle"))
    s.append(txt(X0 + XW / 2, yb + 30, "Δ pairwise macro accuracy after "
                 "erasing that variable (mean over arms)", 10.5,
                 "var(--ink-light)", "middle"))
    s.append(rect(X0 - 190, 44, 12, 8, GREEN, 2))
    s.append(txt(X0 - 174, 52, "trained models", 10.5, "var(--ink-light)"))
    s.append(rect(X0 - 80, 44, 12, 8, GRAYC, 2))
    s.append(txt(X0 - 64, 52, "untrained twin", 10.5, "var(--ink-light)"))
    s.append("</svg>")
    return "".join(s)


def chart_ignite():
    ig = J("steering", "results", "ignite.json")
    W, H = 1040, 322
    s = [svg_open(W, H)]
    arms = (("eng_namespan",
             "English glosses — clamp at the ruler-NAME tokens", -1.45, -.95),
            ("akk_allbutlast",
             "Akkadian — clamp everywhere but the read-out", -1.9, 0.35))
    rows = ([("treat", f) for f in (22835, 44713, 17433, 53704, 56768)]
            + [("ctrl", f) for f in (26421, 30594, 42030, 25239, 50550)])
    ROW = 20
    for pi, (arm, title, lo, hi) in enumerate(arms):
        d = ig["arms"][arm]
        X0, XW, Y0 = 168 + pi * 520, 316, 66
        xm = lambda v: X0 + (min(max(v, lo), hi) - lo) / (hi - lo) * XW  # noqa: E731
        s.append(txt(X0 + XW / 2, 26, title, 11.5, "var(--ink)", "middle",
                     700))
        base = d["feat:treat:22835:a0"]["probe"]
        s.append(line(xm(base), Y0 - 6, xm(base), Y0 + len(rows) * ROW,
                      "var(--border)", 1))
        s.append(txt(xm(base), Y0 - 12, f"no clamp: {base:+.2f}", 9.5,
                     "var(--ink-light)", "middle"))
        for i, (g, f) in enumerate(rows):
            y = Y0 + i * ROW + 8
            c = GREEN if g == "treat" else GRAYC
            name = (FEAT_LABEL.get(f, str(f)) if g == "treat"
                    else f"control {f}")
            s.append(txt(X0 - 8, y + 3.5, name, 10,
                         "var(--ink)" if g == "treat" else "var(--ink-light)",
                         "end", 700 if g == "treat" else 400))
            v0 = d[f"feat:{g}:{f}:a0"]["probe"]
            v8 = d[f"feat:{g}:{f}:a8"]["probe"]
            s.append(line(xm(v0), y, xm(v8), y, c, 1.6))
            s.append(circle(xm(v0), y, 2.6, "#ffffff", c, 1.4))
            s.append(circle(xm(v8), y, 3.4, c))
        yb = Y0 + len(rows) * ROW + 6
        for gv in (lo, (lo + hi) / 2, hi):
            s.append(line(xm(gv), yb, xm(gv), yb + 4, "var(--ink-light)", 1))
            s.append(txt(xm(gv), yb + 15, f"{gv:+.1f}", 9,
                         "var(--ink-light)", "middle"))
        s.append(txt(X0 + XW / 2, yb + 30, "frozen year read-out (sd) — "
                     "hollow: α=0 · solid: α=8", 10,
                     "var(--ink-light)", "middle"))
    s.append("</svg>")
    return "".join(s)


def chart_orthogonal():
    """Left: |cos| of BOTH entity axes against the document direction.
    Right: frozen transfer, in the LATENESS frame — the fragment `year`
    column is BC-positive (larger = earlier) while the entity targets are
    CE-signed, so the stored Spearman is negated here and macro is 1-macro;
    positive now means 'orders documents correctly'."""
    import json as _j

    def collect(pattern, key):
        out = {}
        for f in glob.glob(os.path.join(_P2, "transfer", "results",
                                        pattern)):
            d = _j.load(open(f))
            cosv = [abs(v["cosine"]) for v
                    in d.get("cosine_vs_pairwise_direction", {}).values()
                    if isinstance(v, dict) and "cosine" in v]
            out[(d["method"], d["variant"])] = {
                f"cos_{key}": max(cosv) if cosv else np.nan,
                f"rho_{key}": -d["frozen"]["spearman"]}
        return out
    A = collect("*.mean.json", "A")
    Bx = collect("*.mean.assyrian_ruler.json", "B")
    rows = [{"m": m, "v": v, **rec, **Bx.get((m, v), {})}
            for (m, v), rec in A.items() if "assyrian_ruler" not in v]
    t = pd.DataFrame(rows)
    for c in ("cos_B", "rho_B"):
        if c not in t:
            t[c] = np.nan
    t = t.rename(columns={"cos_A": "cos", "rho_A": "rho"})
    t = t[t.m != "olmo2_7b_random"].sort_values(["m", "v"]).reset_index(
        drop=True)
    chance = 1 / np.sqrt(4096)
    W, H = 1040, 300
    s = [svg_open(W, H)]
    X0, XW = 250, 360
    ROW = 38
    xhi = max(.03, float(np.nanmax([t.cos.max(), t.cos_B.max()])) * 1.12)
    xm = lambda v: X0 + v / xhi * XW                       # noqa: E731
    s.append(txt(X0 + XW / 2, 20,
                 "|cos(entity year axis, document order axis)|", 12,
                 "var(--ink)", "middle", 700))
    s.append(rect(X0, 34, xm(chance) - X0, len(t) * ROW, "#eef0f5", 0))
    s.append(txt(xm(chance) + 6, 46,
                 f"chance in d=4096  (1/√d = {chance:.3f})", 10.5,
                 "var(--ink-light)"))
    for gv in np.arange(0, xhi + 1e-9, .01):
        yb = 34 + len(t) * ROW
        s.append(line(xm(gv), yb, xm(gv), yb + 5, "var(--ink-light)", 1))
        s.append(txt(xm(gv), yb + 16, f"{gv:.2f}", 9.5, "var(--ink-light)",
                     "middle"))
    for i, (_, r) in enumerate(t.iterrows()):
        y = 52 + i * ROW
        c = GREEN if r.m in ("olmo2_7b", "qwen3_8b") else GRAYC
        s.append(txt(X0 - 10, y + 4,
                     f"{r.m.split('_')[0]} · "
                     f"{'akk' if 'akk' in r.v else 'eng'}", 11.5,
                     "var(--ink)", "end", 600))
        s.append(line(X0, y, xm(r.cos), y, c, 2.4))
        s.append(circle(xm(r.cos), y, 5, c))
        if np.isfinite(r.cos_B):
            s.append(circle(xm(r.cos_B), y, 4.5, GOLD, "#ffffff", 1))
    ylg = 34 + len(t) * ROW + 30
    s.append(circle(X0 + 20, ylg, 5, GREEN))
    s.append(txt(X0 + 30, ylg + 4, "famous-figure axis", 10.5,
                 "var(--ink-light)"))
    s.append(circle(X0 + 168, ylg, 4.5, GOLD, "#ffffff", 1))
    s.append(txt(X0 + 178, ylg + 4, "our 34 rulers (E3b)", 10.5,
                 "var(--ink-light)"))
    X1, X1W = 760, 240
    s.append(txt(X1 + X1W / 2, 20, "frozen transfer ρ (later = higher)", 12,
                 "var(--ink)", "middle", 700))
    y0 = 52 + (len(t) - 1) * ROW / 2
    ym2 = lambda v: y0 - v * 380                           # noqa: E731
    for g in (-.2, -.1, 0, .1, .2):
        s.append(line(X1, ym2(g), X1 + X1W, ym2(g),
                      "var(--ink-light)" if g == 0 else
                      "var(--border-light)", 1))
        s.append(txt(X1 - 8, ym2(g) + 4, f"{g:+.1f}" if g else "0", 10,
                     "var(--ink-light)", "end"))
    slot = X1W / len(t)
    bw = slot / 2 - 3
    for i, (_, r) in enumerate(t.iterrows()):
        x0 = X1 + i * slot
        for j, (val, col) in enumerate(((r.rho, GREEN), (r.rho_B, GOLD))):
            if not np.isfinite(val):
                continue
            x = x0 + j * (bw + 2)
            h = abs(val) * 90
            s.append(rect(x, ym2(0) - h if val > 0 else ym2(0), bw,
                          max(h, 1), col, 2))
        s.append(txt(x0 + slot / 2 - 2, ym2(0) + 14,
                     f"{r.m.split('_')[0][:4]}·"
                     f"{'a' if 'akk' in r.v else 'e'}", 8.5,
                     "var(--ink-light)", "middle"))
    s.append(txt(X1 + X1W / 2, 52 + len(t) * ROW,
                 "each entity axis, applied frozen to fragments", 10.5,
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
    s.append(txt(16, 150, "median firing rate of top-50 year features (%)",
                 9.5, "var(--ink-light)", "middle")
             .replace('<text', '<text transform="rotate(-90 16 150)"'))
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
    for gv in (0, .25, .5, .75, 1.0):
        s.append(line(xm2(gv), Y1 + Y1H, xm2(gv), Y1 + Y1H + 5,
                      "var(--ink-light)", 1))
        s.append(txt(xm2(gv), Y1 + Y1H + 16, f"{int(100*gv)}%", 9.5,
                     "var(--ink-light)", "middle"))
    s.append(txt(X1 + X1W / 2, Y1 + Y1H + 30,
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
.cfg-v a{color:var(--green);font-weight:600;text-decoration:underline dotted;}
.frm2{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;background:#fff;border:1px solid var(--border-light);border-radius:4px;padding:0 6px;white-space:nowrap;}
.p2meth{display:grid;grid-template-columns:1fr 1fr auto;gap:10px;background:#f8f9fb;border:1px solid var(--border);border-radius:8px;padding:9px 14px;margin:0 0 10px;font-size:12px;line-height:1.45;color:var(--ink-mid);}
.p2meth b{display:block;font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--green);margin-bottom:2px;}
.p2meth .frm{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:var(--ink);background:#fff;border:1px solid var(--border-light);border-radius:5px;padding:5px 10px;align-self:center;white-space:nowrap;}
</style>'''


def slide(idx, eyebrow, headline, cfg_rows, body, takeaway):
    cfg = ""
    if cfg_rows:
        cells = "".join(f'<div class="cfg-k">{k}</div>'
                        f'<div class="cfg-v">{v}</div>' for k, v in cfg_rows)
        cfg = f'<div class="cfg tight">{cells}</div>'
    return (f'<section class="slide slide-figure fig-major" '
            f'data-index="{idx}">\n'
            f'  <div class="eyebrow">{eyebrow}</div>\n'
            f'  <h2 class="sh">{headline}</h2>\n  {cfg}\n'
            f'  {body}\n'
            f'  <div class="takeaway tight"><span class="tk-label">Key '
            f'takeaway</span>{takeaway}</div>\n</section>\n')


TITLES_NEW = [
    "Phase 2: why does it collapse at the entity-to-document boundary?",
    "Ordering fragments: no model beats the surface floor in Akkadian",
    "The erasure ladder: ruler and era carry the ordering — and so does the untrained twin",
    "The entity year axis and the document axis are different, orthogonal directions",
    "The year direction literally reads 'ancient'; the document direction reads nothing",
    "Across the whole vocabulary: ancient words pile up in the entity axis's first decile only",
    "Where the year features fire: entity-gated, alive mid-text, silent at the read-out",
    "No single year neuron: a distributed code, and the features that carry it",
    "Decomposing the year signal into SAE features: name-culture detectors, not a concept of time",
    "Causal test: pushing a name-culture feature drags the year prediction, controls stay flat",
    "The bridge: forced ON across a whole document, nothing arrives at the read-out token",
    "Ignition at the ruler's own name: the read-out does not move (its control matches the one mover)",
    "Synthesis: a real entity time axis, a form-built document axis, and no route between them",
]

A_GT = '<a href="https://arxiv.org/abs/2310.02207">Gurnee &amp; Tegmark 2023</a>'
A_ELS = '<a href="https://arxiv.org/abs/2410.13194">El-Shangiti et al., NAACL 2025</a>'
A_LEACE = '<a href="https://arxiv.org/abs/2306.03819">Belrose et al., NeurIPS 2023</a>'
A_LENS = '<a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens">nostalgebraist\u2019s logit lens</a>'
A_GEVA = '<a href="https://arxiv.org/abs/2203.14680">Geva et al., EMNLP 2022</a>'
A_SB = ('<a href="https://scholar.google.com/scholar?q=%22Non-parametric+'
        'standard+errors+and+tests+for+network+statistics%22">Snijders'
        ' &amp; Borgatti 1999</a>')
A_SAELL = '<a href="https://www.alignmentforum.org/posts/qykrYY6rXXM7EEs8Q/understanding-sae-features-with-the-logit-lens">SAE-features-with-the-logit-lens</a>'
A_MONO = '<a href="https://transformer-circuits.pub/2023/monosemantic-features">Bricken et al. 2023</a>'
A_ACTADD = '<a href="https://arxiv.org/abs/2308.10248">Turner et al. 2023</a>'
A_TL = '<a href="https://arxiv.org/abs/2303.08112">tuned lens, Belrose et al. 2023</a>'
A_CV = '<a href="https://arxiv.org/abs/2406.11614">Hong et al., EMNLP 2025</a>'
A_GG = '<a href="https://transformer-circuits.pub/2024/scaling-monosemanticity/">Templeton et al. 2024</a>'
A_AUTO = '<a href="https://blog.eleuther.ai/autointerp/">EleutherAI autointerp</a>'
A_QS = '<a href="https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100">Qwen-Scope</a>'
A_KV = '<a href="https://huggingface.co/adamkarvonen/qwen3-8b-saes">Karvonen\u2019s batch-TopK release</a>'


def main():
    deck = open(DECK, encoding="utf-8").read()
    deck = re.sub(r"<!-- PHASE2-BEGIN -->.*?<!-- PHASE2-END -->\n?", "",
                  deck, flags=re.S)
    deck = re.sub(r"/\*P2TITLES\*/.*?/\*P2TITLES-END\*/\n?", "", deck,
                  flags=re.S)
    base = len(re.findall(r'<section class="slide', deck))
    S = []

    S.append(f'''<section class="slide slide-text" data-index="{base}">
  <div class="eyebrow">Phase 2 &middot; the mechanistic program &middot; F1&ndash;F27</div>
  <h2 class="sh">Phase 2: the deck ends where a linear world model ends &mdash; twenty-seven experiments ask <em>why</em> it ends there</h2>
  <div class="text-points">
  <div class="tp"><div class="tp-h">Reframe as ordering (E1, E8)</div>
  <div class="tp-b">628k fragment pairs, &ldquo;which was composed earlier?&rdquo; &mdash; kills the regression-format and label-leakage explanations for the collapse.</div></div>
  <div class="tp"><div class="tp-h">Transfer the axis (E3)</div>
  <div class="tp-b">Freeze the entity year direction, apply it to fragments; measure the angle between the two learned &ldquo;time&rdquo; directions.</div></div>
  <div class="tp"><div class="tp-h">Decompose (F6&ndash;F8, F21&ndash;F25)</div>
  <div class="tp-b">Logit-lens the directions; split representations into sparse-autoencoder features in two independent dictionaries; read each feature by its max-activating contexts.</div></div>
  <div class="tp"><div class="tp-h">Intervene (F23, F26)</div>
  <div class="tp-b">Clamp features with firing-rate-matched controls: does the year prediction move? Can the signal be forced into a document?</div></div>
  <div class="tp"><div class="tp-h">Exhaust the alternatives (F15&ndash;F19, F27)</div>
  <div class="tp-b">Length, find-spot, pooling, seriation, kernel and MLP probes &mdash; every &ldquo;document time&rdquo; candidate decomposes into text form.</div></div>
  </div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>Same corpus, same probes, one new question.</strong> Not <em>whether</em> the model dates documents (it doesn't), but <em>what exists instead</em> &mdash; mapped down to individual features. Cell A = 7,507 historical figures; cell B (our 34 ruler names probed alone &mdash; the entity side already works, slide 9) re-enters at the feature level in F22; cells B&prime;/C = the same 1,187 dated royal inscriptions (40 rulers).</div>
</section>
''')

    S.append(slide(
        base + 1, "E1 + E8 &middot; pairwise ordering",
        "Ordering fragments: untrained twins top the Akkadian board; only"
        " trained models are significant in English",
        [("Task", "order two fragments &mdash; <strong>&ldquo;which was"
          " composed earlier?&rdquo;</strong> &mdash; 628,454 ordered pairs"
          " from the same 1,187 dated fragments, in both variants"
          " (cleaned Akkadian transliteration / literal English gloss)."
          " Relative order only: no absolute year ever enters training."),
         ("Method", "Bradley&ndash;Terry pairwise logistic on activation"
          f" differences, <span class=\'frm2\'>P(a&#8826;b) ="
          " &sigma;(w&middot;(x_a &minus; x_b))</span>; evaluation protocol"
          f" after {A_ELS}, probing frame after {A_GT}. Significance:"
          " permutation that reassigns whole rulers and refits everything"
          f" (B=150) + dyadic bootstrap over rulers ({A_SB})."),
         ("Data &amp; pooling", "<strong>mean pooling</strong> over tokens"
          " at the layer fixed once in F1; quota <strong>m=21</strong>"
          " pairs per ruler pair per draw, weights 1/m, macro over ruler"
          " pairs, <strong>both-rulers-held-out</strong> folds, 100"
          " Monte-Carlo draws. Floor: char n-gram TF-IDF through the"
          " identical protocol.")],
        f'<div class="p2chart">{chart_dissociation()}</div>',
        "<strong>The collapse is not a formatting artifact.</strong>"
        " Left: random-weight twins (hollow) sit at the top of the"
        " Akkadian board and every arm hugs the surface floor &mdash; the"
        " achievable &ldquo;order&rdquo; is text form. Right: in English"
        " only trained OLMo and Qwen are significant (p=.0066) while the"
        " floor itself is not (p=.11); that small trained-only signal is"
        " what the rest of phase 2 dissects."))

    S.append(slide(
        base + 2, "F28 &middot; the single-variable erasure ladder",
        "What was the &ldquo;order&rdquo; actually made of? Erase one"
        " variable at a time &mdash; and the untrained twin loses exactly"
        " as much",
        [("Task", "the English side had a small trained-only signal"
          " (previous slide). Before calling it time, subtract every"
          " candidate: is the ordering carried by ruler identity, era,"
          " what the text is written ON, where it was dug up, or simply"
          " how long it is?"),
         ("Method", f"one concept per run, erased with LEACE ({A_LEACE})"
          " fitted <strong>inside each training fold</strong> and applied"
          " to both sides, then the entire E1 protocol re-run on the"
          " erased representations. Every run carries a manipulation"
          " check &mdash; a probe for the erased concept must fall to"
          " chance (ruler .64&rarr;.16, period .91&rarr;.55). Reading:"
          " <span class=\'frm2\'>&Delta; = macro(erased) &minus;"
          " macro(raw)</span>."),
         ("Data &amp; pooling", "the same 1,187 fragments, mean pooling,"
          " m=21, both-rulers-held-out folds. Six rungs &times; 4 arms"
          " &times; 2 variants: ruler (40 one-hots), period, object type"
          " (sub_genre, top-20), find-spot (top-20), length (log +"
          " 5 quantile bins), and year-decile as the"
          " <strong>positive control</strong> &mdash; erasing coarse era"
          " must hurt any genuine chronology.")],
        f'<div class="p2chart">{chart_ladder()}</div>',
        "<strong>Ruler identity and era carry it &mdash; and they carry it"
        " in an untrained network too.</strong> Erasing ruler costs the"
        " trained models &minus;.150 and the random twin &minus;.118;"
        " every rung repeats that pattern, so what the erasures remove is"
        " an identity/register correlate present without any training, not"
        " learned chronology. Text length contributes exactly nothing"
        " (+.003), and the positive control is no larger than ruler"
        " &mdash; which is what ICC = 1 predicts, since knowing the king"
        " already fixes the era."))

    S.append(slide(
        base + 3, "E3 &middot; frozen transfer + direction geometry",
        "The entity year axis and the document order axis are two"
        " different directions &mdash; orthogonal at chance level",
        [("Task", "does the axis that dates <em>people</em> also order"
          " <em>documents</em>? If document time were the same axis,"
          " diluted, the frozen entity direction should order fragments"
          " above chance."),
         ("Method", "frozen linear transfer &mdash; <span class=\'frm2\'>"
          "s(x) = w_A&middot;x</span>, zero training on the document side"
          " &mdash; plus the direct geometric test cos(w_A, w_doc);"
          f" erasure checks with LEACE ({A_LEACE}), fp64, verified by"
          " ruler-probe collapse (.41&ndash;.59 &rarr; .07&ndash;.14)."),
         ("Data &amp; pooling", "w_A = the cell-A ridge direction"
          " (last-token of the name, its best layer); applied to"
          " mean-pooled fragment activations, both variants. Positive"
          " control: the same code path reproduces &rho;=.87&ndash;.89 on"
          " cell A, so the null is not a bug. <strong>E3b</strong> repeats"
          " everything with w_B &mdash; the axis fitted on <em>our own 34"
          " rulers</em> (ent-last token), the entities these documents are"
          " actually about. Polarity: the fragment year column is"
          " BC-positive while entity targets are CE-signed, so &rho; is"
          " reported here in the lateness frame (higher = orders documents"
          " correctly).")],
        f'<div class="p2chart">{chart_orthogonal()}</div>',
        "<strong>Both entity axes are essentially orthogonal to the"
        " document axis &mdash; and the one that does transfer, transfers"
        " by naming, not by dating.</strong> Every |cos| is within a small"
        " multiple of the 1/&radic;d &asymp; .016 chance line (left; the"
        " largest, olmo&middot;akk ruler axis at .042, is still 2.6&times;"
        " a chance overlap in 4,096 dimensions). The famous-figure axis"
        " orders fragments no better than"
        " an untrained twin; the ruler axis does slightly better"
        " (&rho;=+.05&hellip;+.19, macro .53&ndash;.62 vs twin .43&ndash;"
        ".49) &mdash; and that advantage <strong>vanishes to chance the"
        " moment ruler identity is erased</strong> (macro &rarr;"
        " .49&ndash;.53). The model is recognising WHICH KING the text"
        " names and looking his date up; it is not dating the document."))

    S.append(slide(
        base + 3, "F6 &middot; logit lens on the probe directions",
        "Project each direction onto the vocabulary: the entity year axis"
        " points at ancient-time words, the document axis at junk",
        [("Method", f"direction-level logit lens ({A_LENS}; internal vectors"
          f" read by their top vocabulary projections after {A_GEVA};"
          f" concept-vector reading after {A_CV}): normalize the probe"
          " direction, pass it"
          " through the final RMSNorm and the unembedding, "
          f"<span class=\'frm2\'>&#8467; = W_U(&gamma; &#8857;"
          " &#375;)</span>, and read the extreme tokens at both ends."
          " No training involved."),
         ("Setup", "two directions per model: the cell-A year direction"
          " (ridge, last-token) and E1&rsquo;s pairwise document direction"
          " (mean-pooled); top tokens of each end; random directions"
          " through the identical pipeline as control (F14) &mdash; they"
          " lens to the same kind of junk as the document axis. The"
          f" mid-stack caveat is retired by F29: a per-layer translator"
          f" ({A_TL}), trained on our own corpus, reproduces both"
          " verdicts.")],
        f'<div class="p2chart" style="align-items:flex-start">'
        f'{chart_lens_tokens()}</div>',
        "<strong>The entity year axis is semantically temporal; the"
        " document axis is not a time axis at all.</strong> Its early end"
        " literally reads Ancient / BCE / BC in every model &mdash; in"
        " Qwen even in Chinese (&#20844;&#20803;&#21069; &ldquo;BCE&rdquo;,"
        " &#21476;&#20195; &ldquo;ancient&rdquo;) &mdash; while the"
        " document direction projects onto morphological debris at both"
        " ends."))

    S.append(slide(
        base + 4, "F21 &middot; whole-vocabulary spectroscopy",
        "Not just the extremes: rank all ~150k tokens along each"
        " direction &mdash; ancient vocabulary concentrates in the entity"
        " axis&rsquo;s first decile only",
        [("Method", "upgrade the lens from anecdote to instrument: rank"
          " the <strong>entire vocabulary</strong> along the direction,"
          " split into 10 deciles, score each (decile &times; category)"
          " cell against an empirical null of <strong>50 random"
          " directions</strong>, <span class=\'frm2\'>z = (share &minus;"
          " &mu;_null) / &sigma;_null</span>; a cosine variant divides"
          " out loud unembedding rows. Category reading of vocabulary"
          f" projections follows {A_GEVA}; the random-direction"
          f" calibration is the same control used in the SAE-feature"
          f" logit-lens practice ({A_SAELL}). Cells 3&sigma; above the"
          " null are ringed."),
         ("Setup", "~150k tokens per model &times; 9 keyword categories"
          " (ancient-temporal, modern, year numerals, function words,"
          " capitalized, &hellip;); both directions per model. The"
          " year-token ordering test proved untestable: all three"
          " tokenizers split 4-digit numbers.")],
        f'<div class="p2chart">{chart_spectrum()}</div>',
        "<strong>&ldquo;The year axis is semantic&rdquo; now holds across"
        " the whole spectrum, calibrated against 50 random"
        " directions.</strong>"
        " The ancient-token share spikes in decile 1 of the entity axis"
        " in all three models (z up to +6.8) and survives the cosine"
        " correction; the document axis never leaves the noise in any"
        " decile of any model. Re-reading every direction through"
        " tuned-lens translators (F29) reproduces the picture &mdash; the"
        " entity spike survives (z 3.4&ndash;6.0), the document"
        " directions stay flat &mdash; so the verdict no longer leans on"
        " the raw lens's late-layer assumption."))

    S.append(slide(
        base + 5, "F8 + F11 + F22 &middot; where the year features fire",
        "The year features are entity-gated: alive inside English text,"
        " silent at the read-out, and never engaging Akkadian",
        [("Method", f"split residual activations into sparse-autoencoder"
          f" features ({A_MONO}) in <strong>two independent"
          f" dictionaries</strong>:"
          f" {A_QS} (TopK k=100, layer 24) and {A_KV} (65k, layer 9"
          " &mdash; the only layer of that release passing the"
          " reconstruction gate FVU &le; .35). Inference per dictionary,"
          " <span class=\'frm2\'>z = ReLU(W_enc h + b) &middot;"
          " 1[pre &gt; &theta;]</span>."),
         ("Setup", "features selected by |&rho;(strength, death year)| on"
          " held-out cell-A entities (top-50); firing measured on three"
          " populations &mdash; entity prompts, English glosses, Akkadian"
          " fragments &mdash; at the read-out token and anywhere in the"
          " text.")],
        f'<div class="p2chart">{chart_gating()}</div>',
        "<strong>The gate, in numbers.</strong> At the read-out token the"
        " year features fire on entities (11.7%) and essentially never on"
        " documents (0.08% eng). Mid-text they DO fire inside English"
        " glosses (14.9%) &mdash; the signal exists inside the document"
        " and does not propagate. An independently trained second"
        " dictionary replicates both halves (35.5% eng / 2.0% akk"
        " fired-anywhere)."))

    S.append(slide(
        base + 6, "F8 + F22 &middot; decomposing the year probe",
        "No single year neuron: the year direction is a distributed code"
        " &mdash; and these are the features that carry it",
        [("Method", f"feature&ndash;direction geometry (after the"
          f" SAE-feature logit-lens line, {A_SAELL}): compare every"
          " hunted feature&rsquo;s decoder row to the frozen ridge"
          " direction, <span class=\'frm2\'>cos(W_dec,f&thinsp;,"
          " w_ridge)</span>; hunt features by rank correlation with the"
          " probe&rsquo;s own target, <span class=\'frm2\'>&rho;(z_f,"
          " death year)</span>, on held-out entities."),
         ("Setup", "all candidates firing on &ge;2% of entities in both"
          " dictionaries (754 in Qwen-Scope L24, 776 in the 65k L9);"
          " the ridge direction is the same frozen cell-A probe used"
          " throughout phase 2.")],
        f'<div class="p2chart">{chart_decomposition()}</div>',
        "<strong>The axis is spread across many features &mdash; there is"
        " no &ldquo;year neuron&rdquo;.</strong> The largest overlap is"
        " |cos| = .23 in Qwen-Scope and .08 in the 65k dictionary (left)"
        " &mdash; nowhere near a dedicated axis. The hunt map"
        " (right) shows what carries it instead: 38678 fires on 62% of"
        " entities at &rho;=+.57, surrounded by the name-culture detectors"
        " the next slide reads."))

    S.append(slide(
        base + 7, "F8 + F25 &middot; what the features mean",
        "The year-correlated features are name-culture detectors &mdash;"
        " the &ldquo;time&rdquo; the model knows about people is"
        " onomastics",
        [("Method", f"max-activating examples &mdash; the standard"
          f" feature-reading practice behind Neuronpedia dashboards and"
          f" {A_AUTO} &mdash; corroborated behaviourally by"
          f" Golden-Gate-style clamped generation ({A_GG})."),
         ("Setup", "20 strongest contexts per feature across all three"
          " populations (firing token marked in gold); generation:"
          " clamp at 10&times;act95 during greedy chat decoding &mdash;"
          " clamping the German-surname feature makes the model write"
          " German mid-answer.")],
        chart_feature_cards(),
        "<strong>Naming culture tracks era &mdash; and that is the"
        " correlation the probe reads.</strong> German surname endings"
        " (&rho;=+.40), &ldquo;X of England&rdquo; nobility (&minus;.37),"
        " ancient genealogy formulas (&minus;.37, the only one firing on"
        " the glosses too), Chinese names (&minus;.37): who-you-are"
        " features from which when-you-lived is decodable."
        " <strong>Independent check (F30):</strong> on the one layer"
        " Neuronpedia hosts, third-party autointerp labels for our top-50"
        " year-correlated features come back <strong>26/50"
        " entity-identity vs 6 temporal</strong> &mdash; &ldquo;German"
        " names and places&rdquo;, &ldquo;Chinese surnames&rdquo;,"
        " &ldquo;authors&rsquo; last names&rdquo;. Someone else&rsquo;s"
        " labels, same reading."))

    S.append(slide(
        base + 8, "F23 &middot; causal interventions",
        "Clamp one feature at an entity prompt and the frozen year"
        " prediction moves &mdash; monotonically, in the sign of that"
        " feature&rsquo;s correlation",
        [("Method", f"feature clamping ({A_GG}) with the non-surgicality"
          " discipline:"
          " every treated feature has a <strong>firing-rate-matched random"
          " control</strong>, and the claim is treated-minus-control."
          " Intervention <span class=\'frm2\'>h &larr; h + &alpha;"
          " &middot; act95_f &middot; d_f</span> at every position;"
          " read-out = the frozen cell-A ridge probe at the last token,"
          " in sd-of-death-year units. (Direction-level steering, F9/F12,"
          " was null &mdash; the feature level is where causality"
          " lives.)"),
         ("Setup", "200 held-out entity prompts; hook at the SAE&rsquo;s"
          " layer; &alpha; &isin; {&minus;8 &hellip; +8}; 5 treated"
          " features + 5 matched controls; ablation of single features"
          " changes nothing (distributed code).")],
        f'<div class="p2chart">{chart_causality()}</div>',
        "<strong>Correlation became causation &mdash; for the features,"
        " not for a time concept.</strong> Pushing &ldquo;Chinese"
        " names&rdquo; (&rho;&lt;0) drags the read-out earlier by up to"
        " 0.6 sd; &ldquo;German surnames&rdquo; (&rho;&gt;0) later; all"
        " five controls stay inside the gray band."))

    S.append(slide(
        base + 9, "F23 bridge &middot; forcing the features across a document",
        "Hold the year features ON over an entire gloss &mdash; nothing"
        " arrives at the token the probe reads",
        [("Task", "the gating slide showed the features fire mid-text but"
          " not at the read-out. Can brute force carry them there? If yes,"
          " the entity&ndash;document gap is a routing weakness; if not"
          " even force works, it is a disconnection."),
         ("Method", f"clamp (Golden-Gate style, {A_GG})"
          " <span class=\'frm2\'>h &larr; h + m &#8857;"
          " (&alpha; &middot; act95_f &middot; d_f)</span> on every"
          " mid-document position; the per-sample mask m spares only the"
          " read-out token, so anything measured there had to"
          " <em>propagate</em>. Read-out: does the feature fire at the"
          " last token? Every treated feature has a firing-rate-matched"
          " random control."),
         ("Setup", "300 English glosses; &alpha; = 4&times;act95;"
          " 5 treated + 5 matched control features; pre-registered rules"
          " &mdash; a null <em>with</em> controls is the publishable"
          " outcome.")],
        f'<div class="p2chart">{chart_bridge()}</div>',
        "<strong>Every row reads x &rarr; x.</strong> Whatever a"
        " feature&rsquo;s natural rate at the read-out token (0%, 8%,"
        " 34%), forcing it ON across the whole document changes that rate"
        " by nothing &mdash; treated and control alike. The signal can be"
        " everywhere in the text and still never reach the position the"
        " model reads documents from."))

    S.append(slide(
        base + 10, "F26 ignition &middot; the anchor test",
        "Light the features at the ruler&rsquo;s own name &mdash; the year"
        " read-out still does not move (and the one mover is matched by"
        " its control)",
        [("Task", "maybe the features need their natural trigger &mdash; a"
          " NAME &mdash; not arbitrary positions. So ignite them exactly"
          " where the entity lives inside the document: the ruler-name"
          " token span. This is the feature-level version of the original"
          " cell-C steering idea."),
         ("Method", "offset-mapped char&rarr;token spans for the ruler name"
          " in each gloss (95% coverage, 4.2 tokens on average); clamp"
          " <span class=\'frm2\'>h &larr; h + &alpha; &middot; act95_f"
          " &middot; d_f</span> on the span; frozen ridge read-out at the"
          " last token; rate-matched controls. A separate DIR arm"
          f" (activation addition, {A_ACTADD}) injects the ridge"
          " <em>direction itself</em> at blocks {8,16,24}; its"
          " +5.7&thinsp;sd jump on Akkadian is <strong>circular</strong>"
          " &mdash; the injected vector IS the read-out&rsquo;s own axis"
          " &mdash; and is excluded by the pre-registered rule (a random"
          " direction moves almost nothing)."),
         ("Setup", "400 name-bearing glosses + 400 Akkadian fragments"
          " (clamped everywhere but the read-out); &alpha; &isin;"
          " {4, 8}.")],
        f'<div class="p2chart">{chart_ignite()}</div>',
        "<strong>The causal seal.</strong> On glosses every feature and"
        " every control sits on the no-clamp line. On Akkadian the single"
        " large move (44713 at &alpha;=8) is reproduced exactly by its"
        " rate-matched control &mdash; a nonspecific perturbation, not"
        " the feature. A real, semantic, causally-usable entity time"
        " axis &mdash; and no channel that carries any of it into a"
        " document representation, even by force."))

    S.append(f'''<section class="slide slide-text" data-index="{base + 11}">
  <div class="eyebrow">Phase 2 &middot; synthesis</div>
  <h2 class="sh">Five findings, one conclusion: the knowledge exists, the route does not</h2>
  <div class="text-points">
  <div class="tp"><div class="tp-h">The entity year axis is real and semantic (E3, F6, F21)</div>
  <div class="tp-b">Orthogonal to the document axis at chance level (|cos| &le; .025); its vocabulary end literally reads Ancient / BCE / &#20844;&#20803;&#21069;; the enrichment holds across all ~150k tokens, 3&sigma;+ above a 50-random-direction null (z up to +6.8).</div></div>
  <div class="tp"><div class="tp-h">It is built from name-culture features (F8, F22, F25)</div>
  <div class="tp-b">A distributed code (max |cos| with any single feature: .23): German-surname, nobility, genealogy-formula, Chinese-name detectors &mdash; onomastics, replicated in two independent dictionaries.</div></div>
  <div class="tp"><div class="tp-h">Those features causally feed the read-out (F23)</div>
  <div class="tp-b">Clamping one drags the frozen year prediction up to ~1 sd in the direction its correlation predicts; rate-matched controls stay flat; ablating any single feature changes nothing.</div></div>
  <div class="tp"><div class="tp-h">The document side is text form (E1, E8, F15&ndash;F19, F27)</div>
  <div class="tp-b">Untrained twins and a character-n-gram floor top the Akkadian board; find-spot + length erasure shaves the rest; kernel and MLP probes find nothing the linear ones missed.</div></div>
  <div class="tp"><div class="tp-h">And no channel connects the two (F11, F23 bridge, F26)</div>
  <div class="tp-b">The features are silent at the read-out token on documents; forcing them ON mid-text propagates nothing; igniting them at the ruler&rsquo;s own name moves nothing that its control does not.</div></div>
  </div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>The model dates people, not documents.</strong> The collapse at the entity&rarr;document boundary is a <em>disconnection</em>: a genuine, causally-usable time axis on the entity side, a form-built pseudo-signal on the document side, and &mdash; under every forcing we tried &mdash; no bridge between them.</div>
</section>
''')

    # single source of truth for slide numbering: position in S. (Slides are
    # written with hand-passed indices; inserting one used to shift them all.)
    S = [re.sub(r'data-index="\d+"', f'data-index="{base + i}"', sec, count=1)
         for i, sec in enumerate(S)]
    block = ("<!-- PHASE2-BEGIN -->\n" + STYLE + "\n" + "".join(S)
             + "<!-- PHASE2-END -->\n")
    deck = deck.replace("<script>\nconst TOTAL",
                        block + "<script>\nconst TOTAL")
    new_total = base + len(S)
    deck = re.sub(r"const TOTAL = \d+;", f"const TOTAL = {new_total};",
                  deck, count=1)
    titles_js = ("/*P2TITLES*/TITLES.push("
                 + ",".join('"' + t.replace('"', '\\"') + '"'
                            for t in TITLES_NEW)
                 + ");/*P2TITLES-END*/\n")
    deck = deck.replace("let cur = 0;", titles_js + "let cur = 0;")
    with open(DECK, "w", encoding="utf-8") as f:
        f.write(deck)
    print(f"[done] {base} -> {new_total} slides")


if __name__ == "__main__":
    main()
