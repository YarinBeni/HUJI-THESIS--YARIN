# -*- coding: utf-8 -*-
"""Blog-grade story page (Anthropic/Goodfire register) for the phase-2
findings. Every number, token and context string is read from committed
result files; the page is static HTML+SVG written to figs/story/.

    python make_story_html.py
"""
from __future__ import annotations

import glob
import html
import json
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_P2 = os.path.abspath(os.path.join(_HERE, ".."))
OUT = os.path.join(_HERE, "story")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2a63b8"      # entity time axis
TEAL = "#1e6f6b"      # document / form
AMBER = "#e8862d"     # feature activation
GRAY = "#8b867c"      # twins / controls
INK = "#1c1a17"


def J(*p):
    with open(os.path.join(_P2, *p)) as f:
        return json.load(f)


def esc(t):
    return html.escape(str(t), quote=True)


def _byte_decode(tok):
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(0xa1, 0xac + 1)) + list(range(0xae, 0xff + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    inv = {chr(c): b for b, c in zip(bs, cs)}
    if tok.startswith("▁"):
        return " " + tok[1:]
    if all(ch in inv for ch in tok):
        try:
            return bytes(inv[ch] for ch in tok).decode("utf-8")
        except (UnicodeDecodeError, KeyError):
            return tok
    return tok


# ------------------------------ SVG helpers --------------------------------
def svg_open(w, h):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:auto;display:block">')


def txt(x, y, s, size=12, fill="var(--mut)", anchor="start", weight=400,
        mono=False, opacity=1):
    fam = "ui-monospace,Consolas,monospace" if mono else "inherit"
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="{fam}" opacity="{opacity}">{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke="var(--grid)", w=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{d}/>')


def circle(x, y, r, fill, stroke="none", sw=0):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')


def rect(x, y, w, h, fill, rx=3, opacity=1):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" fill="{fill}" rx="{rx}" '
            f'opacity="{opacity}"/>')


def path(pts, stroke, w=2.2, fill="none", opacity=1):
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    return (f'<path d="{d}" stroke="{stroke}" stroke-width="{w}" '
            f'fill="{fill}" stroke-linecap="round" '
            f'stroke-linejoin="round" opacity="{opacity}"/>')


# ------------------------ chart 1: the dissociation ------------------------
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
           "olmo2_7b_random": "תאום OLMo (אקראי)",
           "llama2_13b_random": "תאום 13B", "llama2_70b_random": "תאום 70B",
           "llama2_7b_random": "תאום 7B", "random": "אקראי (qwen)",
           "tfidf_char": "רצפת n-gram תווים"}
    W, PH, ROW = 980, 64, 24
    n = sp[sp.variant == "akk_maximal"].shape[0]
    H = PH + n * ROW + 46
    s = [svg_open(W, H)]
    for pi, (var, title) in enumerate(
            (("akk_maximal", "אכדית (תעתיק גולמי)"),
             ("eng_tier0", "גלוסה אנגלית"))):
        X0 = 150 + pi * 490
        XW = 320
        d = sp[sp.variant == var].sort_values("macro_acc",
                                              ascending=False)
        lo, hi = .46, .74
        xm = lambda v: X0 + (v - lo) / (hi - lo) * XW      # noqa: E731
        s.append(txt(X0 + XW / 2, 20, title, 15, "var(--ink)", "middle",
                     700))
        for g in (.5, .55, .6, .65, .7):
            s.append(line(xm(g), PH - 18, xm(g), H - 40))
            s.append(txt(xm(g), H - 24, f"{g:.2f}", 10.5, "var(--mut)",
                         "middle"))
        floor = float(d[d.method == "tfidf_char"].macro_acc.iloc[0])
        s.append(line(xm(floor), PH - 18, xm(floor), H - 40, TEAL, 1.6,
                      "5 4"))
        s.append(txt(xm(floor), PH - 26, "רצפת פני-השטח", 11, TEAL,
                     "middle", 600))
        for i, (_, r) in enumerate(d.iterrows()):
            y = PH + i * ROW
            twin = "random" in r.method
            c = BLUE if r.method == "olmo2_7b" else \
                AMBER if r.method == "qwen3_8b" else \
                TEAL if r.method == "tfidf_char" else GRAY
            s.append(line(xm(r.macro_acc - r.macro_sd), y,
                          xm(r.macro_acc + r.macro_sd), y, c, 1.4))
            s.append(circle(xm(r.macro_acc), y, 5,
                            "var(--bg)" if twin else c, c, 1.8))
            s.append(txt(X0 - 8, y + 4, LBL.get(r.method, r.method), 11.5,
                         "var(--ink)" if r.method in
                         ("olmo2_7b", "qwen3_8b", "tfidf_char")
                         else "var(--mut)", "end",
                         700 if r.method in ("olmo2_7b", "qwen3_8b")
                         else 400))
            a = inf[var].get(r.method)
            if a and r.method in ("olmo2_7b", "qwen3_8b", "tfidf_char"):
                p = a["permutation"]["p_value"]
                s.append(txt(xm(r.macro_acc + r.macro_sd) + 6, y + 4,
                             f"p={p:.3g}", 10.5, c, "start", 600))
    s.append("</svg>")
    return "".join(s)


# ------------------- chart 2: lens tokens (token chips) --------------------
def token_chip(t, hot, val=None):
    cls = "chip hot" if hot else "chip"
    return f'<span class="{cls}">{esc(t)}</span>'


def chart_lens_tokens():
    TEMPORAL = ("bc", "bce", "ancient", "athen", "公元前", "古代", "战国",
                "古人")
    out = ['<div class="lensgrid">']
    for m, name in (("olmo2_7b", "OLMo-2 7B"), ("llama2_7b", "Llama-2 7B"),
                    ("qwen3_8b", "Qwen3 8B")):
        d = J("traces", "results", f"{m}.json")["directions"]
        ck = [k for k in d if k.startswith("cellA")][0]
        pk = [k for k in d if k.startswith("pairwise")][0]
        cell = [_byte_decode(e["token"]).strip() or "␣"
                for e in d[ck]["negative_end"][:10]]
        doc = [_byte_decode(e["token"]).strip() or "␣"
               for e in d[pk]["negative_end"][:5]
               + d[pk]["positive_end"][:5]]
        chips_c = "".join(token_chip(
            t, any(k in t.lower() for k in TEMPORAL)) for t in cell)
        chips_d = "".join(f'<span class="chip dim">{esc(t)}</span>'
                          for t in doc)
        out.append(
            f'<div class="lenscol"><div class="lensmodel">{name}</div>'
            f'<div class="lenslabel" style="color:{BLUE}">ציר-השנה של '
            f'הישויות · הקצה המוקדם</div>'
            f'<div class="chiprow" dir="ltr">{chips_c}</div>'
            f'<div class="lenslabel" style="color:var(--mut)">ציר-המסמכים '
            f'(שני הקצוות)</div>'
            f'<div class="chiprow" dir="ltr">{chips_d}</div></div>')
    out.append("</div>")
    return "".join(out)


# -------------------- chart 3: feature cards (dashboard) -------------------
FEAT_LABEL = {44713: "שמות משפחה גרמניים", 17433: "אצולת “X of PLACE”",
              53704: "גנאלוגיה עתיקה", 56768: "שמות סיניים",
              22835: "סיומות שמות מערביים", 9763: "שמות קיסריים סיניים"}


def chart_feature_cards():
    interp = J("sae2", "results", "feature_interp.layer9.json")
    hunt = pd.read_csv(sorted(glob.glob(os.path.join(
        _P2, "sae2", "results", "feature_hunt2.layer*.csv")))[-1])
    hunt = hunt.set_index(hunt.feature.astype(int))
    fh1 = pd.read_csv(os.path.join(_P2, "sae", "results",
                                   "feature_hunt.layer24.csv"))
    f1 = fh1[fh1.feature == 38678].iloc[0]

    def ctx_html(e):
        c = e["context"].split("<|endoftext|>")[0].strip()
        if ">>" in c and "<<" in c:
            pre, rest = c.split(">>", 1)
            tok, post = rest.split("<<", 1)
            return (f'<div class="ctx" dir="ltr">{esc(pre)}'
                    f'<span class="fire">{esc(tok)}</span>'
                    f'{esc(post)}</div>')
        return f'<div class="ctx" dir="ltr">{esc(c)}</div>'

    def meter(v, color, label):
        pct = max(0.6, 100 * v)
        return (f'<div class="meter"><div class="meterlab">{label}</div>'
                f'<div class="meterbar"><div class="meterfill" '
                f'style="width:{pct:.1f}%;background:{color}"></div></div>'
                f'<div class="meterval" dir="ltr">{100*v:.1f}%</div></div>')

    cards = [f'''<div class="card headline">
      <div class="cardtitle">פיצ'ר 38678 · Qwen-Scope שכבה 24 —
      פיצ'ר-הזמן-האנטיטי הראשי (ניסוי F8)</div>
      <div class="cardsub">נדלק על {100*f1.fire_cellA:.0f}% מפרומפטי-הישויות
      עם ρ(עוצמה, שנת-מוות)=‏+{f1.rho_year:.2f} — ועל מסמכים כמעט אף פעם:
      {100*f1.fire_eng_tier0:.2f}% מהגלוסות, {100*f1.fire_akk_maximal:.2f}%
      מהאכדית. "שער-הישות" בפיצ'ר אחד.</div></div>''']
    for f in (44713, 17433, 53704, 56768):
        rec = interp["features"][str(f)]
        exs = [e for e in rec["max_activating"]
               if e["context"].split("<|endoftext|>")[0].strip()][:3]
        rho = float(hunt.loc[f, "rho_year"])
        cards.append(f'''<div class="card">
      <div class="cardtitle">פיצ'ר {f} · “{FEAT_LABEL[f]}”
        <span class="rho" style="color:{BLUE if rho > 0 else TEAL}"
        dir="ltr">ρ(שנה)={rho:+.2f}</span></div>
      <div class="ctxs">{''.join(ctx_html(e) for e in exs)}</div>
      <div class="meters">
        {meter(float(hunt.loc[f, "fire_cellA"]), AMBER, "ישויות")}
        {meter(float(hunt.loc[f, "fire_eng_tier0_frags"]), BLUE, "גלוסות")}
        {meter(float(hunt.loc[f, "fire_akk_maximal_frags"]), TEAL, "אכדית")}
      </div></div>''')
    return '<div class="cards">' + "".join(cards) + "</div>"


# --------------------- chart 4: spectrum composition -----------------------
def chart_spectrum():
    W, H = 980, 300
    s = [svg_open(W, H)]
    for j, (m, name) in enumerate((("olmo2_7b", "OLMo-2 7B"),
                                   ("llama2_7b", "Llama-2 7B"),
                                   ("qwen3_8b", "Qwen3 8B"))):
        d = J("traces", "results", f"spectroscopy.{m}.json")
        ci = d["cats"].index("temporal_ancient")
        X0, XW, Y0, YH = 60 + j * 320, 250, 40, 190
        xm = lambda b: X0 + (b - 1) / 9 * XW               # noqa: E731
        ym = lambda v: Y0 + YH - v / .55 * YH              # noqa: E731
        s.append(txt(X0 + XW / 2, 24, name, 14, "var(--ink)", "middle",
                     700))
        for g in (0, .2, .4):
            s.append(line(X0, ym(g), X0 + XW, ym(g)))
            s.append(txt(X0 - 6, ym(g) + 4, f"{g:.1f}", 10, "var(--mut)",
                         "end"))
        for dname, c in (("pairwise_doc", GRAY), ("cellA", BLUE)):
            rec = d["directions"][dname]["cos"]
            comp = 100 * np.array(rec["composition"])[:, ci]
            z = np.array(rec["z_scores"])[:, ci]
            pts = [(xm(b + 1), ym(v)) for b, v in enumerate(comp)]
            s.append(path(pts, c, 2.2))
            for (x, y), zi in zip(pts, z):
                s.append(circle(x, y, 3, c))
                if abs(zi) >= 3.35:
                    s.append(circle(x, y, 6.5, "none", c, 1.6))
                    s.append(txt(x + 10, y + 3, f"z={zi:+.1f}", 11, c,
                                 "start", 700))
        s.append(txt(X0, H - 22, "מוקדם", 11, "var(--mut)", "start"))
        s.append(txt(X0 + XW, H - 22, "מאוחר", 11, "var(--mut)", "end"))
        s.append(txt(X0 + XW / 2, H - 22, "עשירוני-דירוג לאורך הכיוון",
                     10.5, "var(--mut)", "middle"))
    s.append(txt(20, 150, "% מילים עתיקות בדלי", 11, "var(--mut)",
                 "middle") .replace('<text', '<text transform="rotate(-90 20 150)"'))
    s.append("</svg>")
    leg = (f'<div class="legend" dir="rtl">'
           f'<span class="key"><i style="background:{BLUE}"></i>'
           f'ציר-השנה של הישויות</span>'
           f'<span class="key"><i style="background:{GRAY}"></i>'
           f'ציר-המסמכים</span>'
           f'<span class="key">○ = מובהק (|z| ≥ 3.35, בונפרוני מול 50 '
           f'כיוונים אקראיים)</span></div>')
    return "".join(s) + leg


# ------------------------ chart 5: causality -------------------------------
def chart_causality():
    st = J("sae2", "results", "steer.layer9.json")
    hunt = pd.read_csv(sorted(glob.glob(os.path.join(
        _P2, "sae2", "results", "feature_hunt2.layer*.csv")))[-1])
    rho = dict(zip(hunt.feature.astype(int), hunt.rho_year))
    alphas = [-8, -4, -2, 0, 2, 4, 8]
    W, H = 980, 360
    X0, XW, Y0, YH = 70, 520, 46, 250
    lo, hi = -.75, 1.05
    xm = lambda a: X0 + (a + 8) / 16 * XW                  # noqa: E731
    ym = lambda v: Y0 + YH - (v - lo) / (hi - lo) * YH     # noqa: E731
    s = [svg_open(W, H)]
    for g in (-.5, 0, .5, 1):
        s.append(line(X0, ym(g), X0 + XW, ym(g)))
        s.append(txt(X0 - 8, ym(g) + 4, f"{g:+.1f}", 10.5, "var(--mut)",
                     "end"))
    ctrl = np.array([[st["runs"][f"ctrl:{f}"]["amplify"][str(a)]
                      for a in alphas] for f in st["ctrl"]])
    band = "M " + " L ".join(
        f"{xm(a):.1f} {ym(v):.1f}" for a, v in zip(alphas, ctrl.min(0)))
    band += " L " + " L ".join(
        f"{xm(a):.1f} {ym(v):.1f}"
        for a, v in zip(alphas[::-1], ctrl.max(0)[::-1])) + " Z"
    s.append(f'<path d="{band}" fill="{GRAY}" opacity="0.16"/>')
    s.append(txt(xm(5.2), ym(float(ctrl.max(0)[-1])) - 8,
                 "רצועת 5 פיצ'רי-הביקורת", 11.5, "var(--mut)"))
    for f in st["treat"]:
        cur = [st["runs"][f"treat:{f}"]["amplify"][str(a)] for a in alphas]
        r = rho.get(int(f), 0)
        c = BLUE if r > 0 else TEAL
        s.append(path([(xm(a), ym(v)) for a, v in zip(alphas, cur)], c))
        for a, v in zip(alphas, cur):
            s.append(circle(xm(a), ym(v), 3.2, c))
        dodge = {17433: 20, 53704: -6, 22835: 14}.get(int(f), 0)
        s.append(txt(xm(8) + 10, ym(cur[-1]) + 4 + dodge,
                     f"{FEAT_LABEL.get(int(f), f)}  (ρ={r:+.2f})", 11.5, c,
                     "start", 600))
    for a in alphas:
        s.append(txt(xm(a), Y0 + YH + 18, f"{a:+d}" if a else "0", 10.5,
                     "var(--mut)", "middle"))
    s.append(txt(X0 + XW / 2, H - 8,
                 "עוצמת ההצמדה α (× act95 של הפיצ'ר) בפרומפט-ישות", 11.5,
                 "var(--mut)", "middle"))
    s.append(txt(24, 170, "קריאת-השנה (סטיות-תקן)", 11.5, "var(--mut)",
                 "middle").replace('<text',
                                   '<text transform="rotate(-90 24 170)"'))
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
    W = 980
    ROW = 34
    H = 60 + len(rows) * ROW + 30
    X0, XW = 300, 560
    xm = lambda v: X0 + v / 38 * XW                        # noqa: E731
    s = [svg_open(W, H)]
    s.append(txt(X0, 24, "% מהגלוסות שבהן הפיצ'ר נדלק בטוקן-הקריאה", 12,
                 "var(--mut)"))
    for i, (g, f, b0, b4) in enumerate(rows):
        y = 52 + i * ROW
        c = AMBER if g == "treat" else GRAY
        name = FEAT_LABEL.get(f, str(f)) if g == "treat" else f"ביקורת {f}"
        s.append(txt(X0 - 10, y + 11, name, 11.5,
                     "var(--ink)" if g == "treat" else "var(--mut)", "end",
                     600 if g == "treat" else 400))
        s.append(rect(X0, y, max(xm(b0) - X0, 1.2), 8, c, 2, .4))
        s.append(rect(X0, y + 10, max(xm(b4) - X0, 1.2), 8, c, 2, 1))
        s.append(txt(xm(max(b0, b4)) + 8, y + 13,
                     f"{b0:.1f} ← {b4:.1f}", 10.5, "var(--mut)"))
    s.append(txt(X0, H - 8, "פס עליון-בהיר: בלי הצמדה · פס תחתון-מלא: "
                 "הפיצ'ר הוצמד בכוח על כל אמצע-המסמך (α=4)", 11,
                 "var(--mut)"))
    s.append("</svg>")
    return "".join(s)


# --------------------------- page assembly ---------------------------------
def section(eyebrow, headline, chart, reading, why):
    return f'''<section>
  <div class="eyebrow">{eyebrow}</div>
  <h2>{headline}</h2>
  <figure class="chart" dir="ltr">{chart}</figure>
  <div class="read"><span class="readlab">איך לקרוא את זה</span>{reading}</div>
  <div class="why"><span class="readlab">למה זה חשוב</span>{why}</div>
</section>'''


def main():
    parts = []
    parts.append(chart_dissociation())
    c_lens = chart_lens_tokens()
    c_cards = chart_feature_cards()
    c_spec = chart_spectrum()
    c_caus = chart_causality()
    c_bridge = chart_bridge()

    page = f'''<title>מדוע מודלים יודעים "מתי" על אנשים — אבל לא על טקסטים</title>
<style>
:root {{
  --bg:#f8f9f8; --card:#ffffff; --ink:#1c1a17; --mut:#6b675f;
  --grid:#e6e6e2; --blue:{BLUE}; --teal:{TEAL}; --amber:{AMBER};
  --chip:#f2f0ea; --fire:#fbe3c5;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#161715; --card:#1e1f1d; --ink:#ece9e2; --mut:#a09b90;
  --grid:#32332f; --blue:#6b9be0; --teal:#57b0a9; --amber:#eda45c;
  --chip:#2a2b28; --fire:#4d3517;
}} }}
:root[data-theme="dark"] {{
  --bg:#161715; --card:#1e1f1d; --ink:#ece9e2; --mut:#a09b90;
  --grid:#32332f; --blue:#6b9be0; --teal:#57b0a9; --amber:#eda45c;
  --chip:#2a2b28; --fire:#4d3517;
}}
* {{ box-sizing:border-box }}
body {{ background:var(--bg); color:var(--ink); margin:0; direction:rtl;
  font-family:"Segoe UI",-apple-system,"Noto Sans Hebrew",Arial,sans-serif;
  line-height:1.65 }}
.wrap {{ max-width:62rem; margin:0 auto; padding:3rem 1.2rem 5rem }}
header h1 {{ font-size:2.1rem; line-height:1.22; margin:.4rem 0 .6rem;
  text-wrap:balance; letter-spacing:-.01em }}
header .sub {{ color:var(--mut); max-width:44rem; font-size:1.02rem }}
.kicker {{ color:var(--blue); font-weight:700; font-size:.8rem;
  letter-spacing:.14em }}
section {{ background:var(--card); border:1px solid var(--grid);
  border-radius:14px; padding:1.8rem 2rem 1.5rem; margin:1.6rem 0 }}
.eyebrow {{ color:var(--mut); font-size:.74rem; font-weight:700;
  letter-spacing:.12em }}
h2 {{ font-size:1.45rem; line-height:1.3; margin:.25rem 0 1rem;
  text-wrap:balance }}
.chart {{ margin:0 0 1.1rem; overflow-x:auto }}
.read,.why {{ font-size:.97rem; max-width:52rem; margin:.6rem 0 }}
.readlab {{ display:inline-block; font-size:.72rem; font-weight:700;
  letter-spacing:.1em; color:var(--blue); margin-inline-end:.6rem }}
.why .readlab {{ color:var(--teal) }}
.legend {{ display:flex; gap:1.2rem; flex-wrap:wrap; color:var(--mut);
  font-size:.85rem; margin-top:.4rem }}
.key i {{ display:inline-block; width:.75em; height:.75em;
  border-radius:3px; margin-inline-end:.35em }}
.lensgrid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.1rem }}
@media (max-width:720px) {{ .lensgrid {{ grid-template-columns:1fr }} }}
.lensmodel {{ font-weight:700; margin-bottom:.4rem }}
.lenslabel {{ font-size:.76rem; font-weight:700; letter-spacing:.05em;
  margin:.5rem 0 .3rem }}
.chiprow {{ display:flex; flex-wrap:wrap; gap:.3rem }}
.chip {{ background:var(--chip); border-radius:6px; padding:.1rem .45rem;
  font-family:ui-monospace,Consolas,monospace; font-size:.82rem;
  color:var(--mut) }}
.chip.hot {{ background:var(--fire); color:var(--ink); font-weight:700 }}
.chip.dim {{ opacity:.75 }}
.cards {{ display:grid; gap:1rem }}
.card {{ border:1px solid var(--grid); border-radius:10px;
  padding:1rem 1.2rem }}
.card.headline {{ background:linear-gradient(0deg,transparent,transparent),
  var(--chip) }}
.cardtitle {{ font-weight:700; font-size:1rem }}
.cardsub {{ color:var(--mut); font-size:.92rem; margin-top:.3rem }}
.rho {{ font-weight:700; font-size:.85rem; margin-inline-start:.6rem }}
.ctxs {{ margin:.6rem 0 }}
.ctx {{ font-family:ui-monospace,Consolas,monospace; font-size:.85rem;
  background:var(--chip); border-radius:6px; padding:.28rem .6rem;
  margin:.3rem 0; color:var(--mut); overflow-x:auto; white-space:nowrap }}
.fire {{ background:var(--amber); color:#fff; border-radius:4px;
  padding:0 .3rem; font-weight:700 }}
.meters {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem }}
.meter {{ font-size:.8rem }}
.meterlab {{ color:var(--mut); margin-bottom:.15rem }}
.meterbar {{ background:var(--chip); border-radius:99px; height:8px }}
.meterfill {{ height:8px; border-radius:99px }}
.meterval {{ color:var(--mut); font-variant-numeric:tabular-nums;
  margin-top:.15rem }}
footer {{ color:var(--mut); font-size:.82rem; margin-top:2.4rem;
  border-top:1px solid var(--grid); padding-top:1rem }}
svg text {{ font-family:"Segoe UI",-apple-system,"Noto Sans Hebrew",Arial,
  sans-serif }}
</style>
<div class="wrap">
<header>
  <div class="kicker">התזה · שלב 2 · לוח הממצאים</div>
  <h1>המודל יודע "מתי" על אנשים — אבל לא על טקסטים. פירקנו את הפער עד לפיצ'רים.</h1>
  <p class="sub">קורפוס: 1,187 כתובות מלכותיות אשוריות מתוארכות (40 מלכים).
  ‏probe לינארי קורא שנה מתוך ייצוגי-מודל מצוין על שמות (ρ≈.88) — ונכשל על
  המסמכים עצמם. שלב 2 שאל למה, ב-27 ניסויים. אלה התמונות.</p>
</header>

{section("ניסויים E1 + E8 · סידור זוגות עם פרמוטציה ברמת-מלך",
 "שום מודל לא מנצח את פני-השטח באכדית; באנגלית — רק מודלים מאומנים נושאים אות אמיתי",
 parts[0],
 "כל שורה = מודל; הנקודה = דיוק בסידור זוגות פרגמנטים (“מי נכתב קודם?”), הקו המקווקו = רצפה של n-grams של תווים בלי שום מודל-שפה. עיגול חלול = רשת עם משקלים אקראיים שמעולם לא אומנה. משמאל (אכדית): תאומים אקראיים בצמרת והכול צמוד לרצפה — הסדר בא מצורת הטקסט, לא מידע. מימין (אנגלית): רק OLMo ו-Qwen המאומנים מובהקים (p=.0066, פרמוטציה שמערבבת מלכים שלמים) בעוד הרצפה עצמה לא (p=.11).",
 "זו הבעיה כולה בתמונה אחת: כשמסירים את היתרון של ידע-שפה, אין שום ייצוג זמן מסמכי באכדית — והאות האנגלי הקטן הוא מה שהמשכנו לנתח.")}

{section("ניסוי F6 · logit lens על כיוון ה-probe",
 "כיוון-השנה של הישויות ממש “אומר” עתיקוּת — וכיוון-המסמכים אומר כלום",
 c_lens,
 "לכל מודל שתי עמודות של טוקנים אמיתיים מהקרנת הכיוון על אוצר-המילים. למעלה: הקצה המוקדם של כיוון-השנה שנלמד על שמות — Ancient, BCE, BC ‏(OLMo), ‏ancient/BC ‏(Llama), וב-Qwen אפילו בסינית: 公元前 (“לפנה”ס”), 古代 (“עתיק”), 战国 (“תקופת המדינות הלוחמות”) — מודגשים בענבר. למטה: אותו מבחן בדיוק על הכיוון שנלמד לסדר מסמכים — ג'יבריש: ‏exus, packing, ‏.getChannel.",
 "בלי לאמן שום דבר נוסף, הכיוון עצמו מעיד: ציר-הישויות הוא ציר זמן סמנטי; ציר-המסמכים הוא לא ציר-זמן בכלל.")}

{section("ניסוי F21 · ספקטרוסקופיה על כל אוצר-המילים",
 "לא רק הקצוות: לאורך כל הספקטרום, מילות-עתיקוּת מצטופפות בדלי הראשון של ציר-הישויות בלבד",
 c_spec,
 "השדרוג של הבדיקה הקודמת: מדרגים את כל ~150 אלף הטוקנים לאורך הכיוון, מחלקים ל-10 דליים, ובודקים את אחוז מילות-העתיקוּת בכל דלי מול 50 כיוונים אקראיים. הקו הכחול (ציר-הישויות) מזנק בדלי המוקדם בשלושת המודלים — מובהק (z עד 6.8). הקו האפור (ציר-המסמכים) לא יוצא מהרעש באף דלי.",
 "הטענה “הציר סמנטי” הפכה מאנקדוטת טופ-10 לקביעה על כל הספקטרום, עם ביקורת אקראית ותיקון ריבוי-השוואות.")}

{section("ניסויים F8 + F25 · פירוק ל-SAE ופרשנות הפיצ'רים",
 "כשמפרקים את אות-השנה לפיצ'רים — מקבלים גלאי תרבות-שמות, לא מושג של זמן",
 c_cards,
 "כל כרטיס = פיצ'ר אמיתי מה-SAE, עם שלושת הקטעים שהכי מדליקים אותו (הטוקן היורה מסומן בענבר — הפורמט של דשבורדי Anthropic) ושיעורי-הירי שלו על שלוש אוכלוסיות. הפיצ'רים שהכי מתואמים עם שנה מזהים: סיומות שמות גרמניים (Kienzle, Rusch), אצולת “X of England”, נוסחות גנאלוגיה עתיקות (“son Cambyses I” — היחיד שנדלק גם על הגלוסות), ושמות סיניים. המתאם-עם-שנה נובע מכך שתרבות-שמות עוקבת אחרי תקופה.",
 "זו התשובה ל“מה זה הזמן שהמודל יודע על ישויות”: קוד מבוזר של גלאים אונומסטיים — מי-אתה — שממנו אפשר לשחזר מתי-חיית.")}

{section("ניסוי F23 · התערבות סיבתית עם ביקורות מותאמות",
 "דחיפת פיצ'ר-שמות מזיזה את חיזוי-השנה — בכיוון הנכון, ורק אצל פיצ'רים אמיתיים",
 c_caus,
 "לוקחים פרומפט של ישות, מצמידים בכוח פיצ'ר אחד בעוצמה α, ומודדים את חיזוי-השנה של אותו probe קפוא. כל קו צבעוני = פיצ'ר-שמות; הרצועה האפורה = חמישה פיצ'רי-ביקורת עם אותו שיעור-ירי אבל בלי קשר לשנה. הקווים חוצים את הרצועה מונוטונית ובכיוון ה-ρ של כל פיצ'ר: “שמות סיניים” (ρ<0) גורר את החיזוי מוקדם-יותר עד ‎−0.6 סטיות-תקן; “שמות גרמניים” (ρ>0) מאוחר-יותר.",
 "מתאם הפך לסיבה: הפיצ'רים האונומסטיים באמת מזינים את חיזוי-השנה. (הפתעה נלווית: הצמדת “שמות גרמניים” בזמן ג'נרציה חופשית גוררת את המודל לכתוב גרמנית.)")}

{section("ניסוי F23 (הגשר) + F26 (ההצתה)",
 "ולמה זה לא עובר למסמכים: גם בכפייה, האות לא מגיע לנקודת-הקריאה",
 c_bridge,
 "לכל פיצ'ר שני פסים: בהיר = מצב טבעי, מלא = אחרי שהכרחנו את הפיצ'ר לדלוק על כל אמצע-המסמך. השאלה: האם הוא מגיע לטוקן האחרון, שממנו ה-probe קורא? המספרים זהים בכל שורה (0.0→0.0, ‏34.0→34.0) — כלום לא זז, גם אצל המטופלים וגם אצל הביקורות. ניסוי-ההמשך F26 ניסה גם להצית את הפיצ'רים בדיוק בטוקני שם-המלך בתוך הגלוסות — ושוב שטוח לחלוטין מול הביקורות.",
 "החותם הסיבתי של התזה: הקריסה ישות→מסמך אינה חוסר-ידע אלא ניתוק — יש מנגנון זמן-ישויות, אין שום ערוץ שמוליך אותו אל ייצוג המסמך, אפילו כשמזריקים את האות בכוח.")}

<footer>כל המספרים, הטוקנים והקטעים נקראים מקובצי התוצאות שב-repo
(‏pairs/, traces/, sae/, sae2/, steering/) על-ידי
<span dir="ltr">figs/make_story_html.py</span>; אפס הקלדה ידנית.
המדריך המלא: TEACHING_GUIDE_HE.md.</footer>
</div>'''
    p = os.path.join(OUT, "PHASE2_STORY_HE.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[done] -> {p} ({len(page)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
