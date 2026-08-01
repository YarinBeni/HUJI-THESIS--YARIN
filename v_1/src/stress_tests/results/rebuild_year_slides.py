#!/usr/bin/env python3
"""Rewrite deck slides 25 and 27 (the two fragment-level YEAR slides) from the
committed mc_group results.

Both slides described their protocol as "stratified-by-ruler CV", which contradicts
slide 2 ("200 Monte-Carlo draws over 8 balanced rulers, GroupKFold by ruler") and is
the leaky variant: with only 17 distinct year values across 8 rulers, a ruler that
appears in both train and test lets the probe re-identify scribal style and read the
date off it. That is why the old slides showed TF-IDF at rho .707-.775, beating every
model. Under the deck's own stated protocol it does not.

Two reporting changes fall out of the fix:

  * R-squared is dropped. Under GroupKFold a test fold is (nearly) a single ruler, so
    its year variance is ~0 and R-squared collapses to the same degenerate -0.22 for
    every arm including the floor. It carries no signal here; Spearman does.
  * Both poolings are shown. `last` is the paper-faithful site, but on fragments
    `mean` is uniformly stronger, and hiding it would misrepresent the ceiling.

Idempotent: re-running rewrites the same two <section> blocks.

    python rebuild_year_slides.py
"""
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))       # .../v_1/src
PROBES = os.path.join(_SRC, "world_models", "akkadian", "results", "probes")
assert os.path.isdir(PROBES), PROBES
HTML = os.path.join(_HERE, "thesis_story_9.html")

ARMS = [("tfidf", "TF-IDF*", True), ("llama2_70b", "Llama-2-70B", False),
        ("llama2_13b", "Llama-2-13B", False), ("llama2_7b", "Llama-2-7B", False),
        ("qwen3_32b", "Qwen3-32B", False), ("qwen3_8b", "Qwen3-8B", False),
        ("qwen3_1b7", "Qwen3-1.7B", False), ("gpt_oss_120b", "gpt-oss-120B", False),
        ("thalesian_cunei400m", "cuneiform-400M", False),
        ("thalesian_akk300m", "AKK-300M", False), ("umt5_base", "uMT5-base", False),
        ("llama2_70b_random", "Llama-2-70B random*", True),
        ("llama2_13b_random", "Llama-2-13B random*", True),
        ("llama2_7b_random", "Llama-2-7B random*", True),
        ("random", "random Qwen3-8B*", True)]


def block(arm, variant, pool):
    f = os.path.join(PROBES, arm, f"{variant}.r8.year.{pool}.ridge.json")
    if not os.path.exists(f):
        return {}
    return json.load(open(f)).get("mc_group") or {}


def cell(v):
    if not isinstance(v, (int, float)) or v != v:
        return '<td class="num">&ndash;</td>'
    s = f"{v:.3f}".replace("0.", ".").replace("-.", "&minus;.")
    return f'<td class="num">{s}</td>'


def table(variant):
    rows = []
    for arm, lab, is_ctrl in ARMS:
        if arm == "tfidf":
            g = block(arm, variant, "text")
            vals = [g.get("spearman_mean"), g.get("pls_spearman_mean"), None, None]
        else:
            gl, gm = block(arm, variant, "last"), block(arm, variant, "mean")
            vals = [gl.get("spearman_mean"), gl.get("pls_spearman_mean"),
                    gm.get("spearman_mean"), gm.get("pls_spearman_mean")]
        cls = ' class="rand"' if is_ctrl else ""
        rows.append(f'    <tr{cls}><td><span class="mdl">{lab}</span></td>'
                    + "".join(cell(v) for v in vals) + "</tr>")
    return ('  <table class="rtbl compact"><thead>'
            '<tr><th>model</th><th class="num">ridge &rho;<br><small>last</small></th>'
            '<th class="num">PLS &rho;<br><small>last</small></th>'
            '<th class="num">ridge &rho;<br><small>mean</small></th>'
            '<th class="num">PLS &rho;<br><small>mean</small></th></tr>'
            "</thead><tbody>\n" + "\n".join(rows) + "\n  </tbody></table>")


CFG = (
    '  <div class="cfg"><div class="cfg-k">Setup</div><div class="cfg-v">entity = '
    "{setup}; whole-fragment embedding at the hold-out-best layer &rarr; ridge probe "
    "and a PLS sweep for <strong>year</strong>.</div>"
    '<div class="cfg-k">Metric</div><div class="cfg-v"><strong>balanced Monte-Carlo</strong> '
    "on the 8 dense rulers (r8): cap 21 per ruler, 200 draws, "
    "<strong>GroupKFold&nbsp;<em>by ruler</em></strong> &mdash; a ruler is wholly in train "
    "or wholly in test, so the probe cannot re-identify a scribe and read the date off "
    "the identity. Reported: <strong>Spearman &rho;</strong> (mean over draws) for ridge "
    "and best-<em>k</em> PLS, at both poolings. "
    "<strong>R&sup2; is not reported</strong>: a grouped test fold is essentially one "
    "ruler, so its year variance is ~0 and R&sup2; degenerates to &minus;0.22 for every "
    "arm, floor included. Real learning must beat the TF-IDF floor <em>and</em> the "
    "arm&rsquo;s own random twin.</div></div>"
)

SLIDES = {
    25: dict(
        variant="eng_tier0",
        eyebrow="Experiment 1 &middot; English tier-0 gloss &middot; grouped Monte-Carlo",
        h2="Year (time) &mdash; can the date be read from the faithful English gloss?",
        setup=("a whole fragment&rsquo;s <strong>faithful literal English gloss</strong> "
               "(tier-0, no hallucinating cleaner &mdash; the aggressive <em>eng-maximal</em> "
               "clean is dropped)"),
        cap="grouped-MC (r8) &mdash; Spearman &rho;, ridge and PLS, both poolings",
        note=("Under the non-leaky protocol the gloss result <strong>inverts the earlier "
              "slide</strong>. The char-n-gram floor collapses from &rho;&nbsp;.775 to "
              "<strong>.066</strong> (PLS) &mdash; almost all of its apparent skill was "
              "ruler re-identification. Meanwhile the trained arms hold up on "
              "<code>mean</code> pooling: <strong>AKK-300M .422</strong>, Qwen3-8B .383, "
              "Qwen3-32B .381, Llama-2-13B .359, against random twins at .23&ndash;.28. "
              "So the date <em>is</em> linearly recoverable from a faithful English gloss "
              "of an unseen ruler&rsquo;s tablets, and the signal is learned rather than "
              "architectural. Note <code>last</code> pooling is much weaker than "
              "<code>mean</code> here &mdash; a fragment&rsquo;s final token is a poor "
              "summary of a document. <em>* = control.</em>")),
    27: dict(
        variant="akk_maximal",
        eyebrow="Experiment 2 &middot; Akkadian (maximal) &middot; grouped Monte-Carlo",
        h2="Year (time) &mdash; can the date be read from the raw Akkadian?",
        setup=("the whole fragment&rsquo;s <strong>maximal Akkadian text</strong> "
               "(no translation, ruler names stripped)"),
        cap="grouped-MC (r8) &mdash; Spearman &rho;, ridge and PLS, both poolings",
        note=("On raw Akkadian the floor goes to <strong>&minus;.016</strong> &mdash; a "
              "char-n-gram model has <em>no</em> transferable date signal once the ruler "
              "it memorised is held out. The ordering at the top reproduces the deck&rsquo;s "
              "headline: <strong>cuneiform-400M .349</strong>, Llama-2-70B .311, "
              "AKK-300M .292. But the honest caveat is the control column: "
              "<strong>Llama-2-70B-random scores .322</strong>, third overall, and random "
              "Qwen .280. With only 8 held-out rulers the trained/random margin is inside "
              "the noise for every arm except cuneiform-400M, so this cell should be read "
              "as <em>weak, not clean</em> evidence of a learned chronology &mdash; unlike "
              "the English gloss in Exp&nbsp;1. <em>* = control.</em>")),
}


def render(idx, s):
    return (f'<section class="slide slide-text" data-index="{idx}">\n'
            f'  <div class="eyebrow">{s["eyebrow"]}</div>\n'
            f'  <h2 class="sh">{s["h2"]}</h2>\n'
            + CFG.format(setup=s["setup"]) + "\n"
            f'  <p class="tbl-cap">{s["cap"]}</p>\n'
            + table(s["variant"]) + "\n"
            f'  <p class="fig-note">{s["note"]}</p>\n'
            "</section>")


def main():
    html = open(HTML).read()
    for idx, s in SLIDES.items():
        pat = re.compile(r'<section class="slide slide-text" data-index="%d">.*?</section>'
                         % idx, re.S)
        if not pat.search(html):
            raise SystemExit(f"slide {idx} not found")
        html = pat.sub(lambda m: render(idx, s), html, count=1)
    open(HTML, "w").write(html)
    left = [m for m in re.findall(r"[Ss]tratified-by-ruler", html)]
    print(f"rewrote slides {sorted(SLIDES)}; "
          f"remaining 'stratified-by-ruler' mentions: {len(left)}")


if __name__ == "__main__":
    main()
