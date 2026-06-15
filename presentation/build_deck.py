#!/usr/bin/env python3
"""Build a self-contained HTML slide deck for the advisor meeting.

Each figure is base64-embedded so the resulting HTML opens on any machine
with no dependency on the repo. Run:  python presentation/build_deck.py
"""
import base64
import mimetypes
import os
import json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_uri(rel_path):
    """Return a base64 data URI for an image in the repo, or None if missing."""
    path = os.path.join(REPO, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Slide content. Each slide:
#   kind:     "title" | "text" | "figure" | "method"
#   eyebrow:  small label above the title
#   title:    slide heading
#   takeaway: the ONE sentence you want the advisors to remember
#   fact:     a small supporting line (numbers) for your own reference
#   fig:      relative path to figure (for kind="figure")
#   body:     list of (heading, text) for text/method slides
# ---------------------------------------------------------------------------
SLIDES = [
    {
        "kind": "title",
        "title": "Decoding Linguistic Evolution",
        "subtitle": "Causal AI for the Diachronic Dating of Ancient Akkadian",
        "meta": "M.Sc. Thesis &middot; Yarin Beer &middot; HUJI Computer Science<br>"
                "Advisors: Prof. Nathan Wasserman &middot; Dr. Barak Sober &middot; Prof. Gabriel Stanovsky",
        "takeaway": "What follows is the evaluation method we built and what it revealed — "
                    "not a list of next steps.",
    },
    {
        "kind": "text",
        "eyebrow": "The question",
        "title": "Can a language model actually date Akkadian — or only pretend to?",
        "body": [
            ("The goal",
             "3,000 years of Akkadian literature is largely undated. Dating is, in part, a "
             "distributional problem — features that change in frequency over time — so "
             "language models should be able to help."),
            ("The catch",
             "A model can score well by exploiting <b>corpus artifacts</b> — royal names, "
             "formulaic openings, archive provenance — without representing linguistic change "
             "at all."),
            ("Our real question",
             "Do the model's representations carry a <b>causally grounded</b> diachronic signal, "
             "or only an <b>apparent</b> one? Telling these two apart is the whole methodological "
             "problem — and the reason we built the protocol in this talk."),
        ],
    },
    {
        "kind": "figure",
        "eyebrow": "How we got here · 1",
        "title": "Period classification looks solved",
        "fig": "v_1/src/linear_probing/results/letters__probe_cls__period/figures/layer_accuracy_curve.png",
        "takeaway": "A simple linear probe on Qwen separates Old Babylonian / Neo-Assyrian / Late "
                    "Babylonian letters at ~98% accuracy — but “looking solved” is exactly "
                    "the trap, because this signal could be royal names and formulae, not language.",
        "fact": "Letters corpus, 4,957 texts, 3 periods. tier0 ≈ 98%, maximal cleaning ≈ 87–93%, "
                "still above the n-gram baselines — so something real is there, but what?",
    },
    {
        "kind": "figure",
        "eyebrow": "How we got here · 2",
        "title": "Strip the artifacts and the easy clusters blur",
        "fig": "v_1/src/linear_probing/results/letters__probe_cls__period/figures/tsne_best_layer.png",
        "takeaway": "When we aggressively clean the text (maximal: remove names, logograms, "
                    "determinatives, formulae), the crisp left-panel period clusters dissolve on the "
                    "right — evidence that much of the easy “dating” was riding on surface cues.",
        "fact": "t-SNE of the same Qwen layer. Left = tier0 (clean OB/NA/LB blobs). "
                "Right = maximal cleaning (periods overlap). This is why we needed a harder, "
                "artifact-controlled protocol.",
    },
    {
        "kind": "method",
        "eyebrow": "The method we developed",
        "title": "The maximal-balanced PLS protocol",
        "lead": "Our evaluation is five deliberate choices. Each one closes a specific door a model "
                "could use to fake a date — this is the core methodological contribution.",
        "items": [
            ("PLS", "Partial Least-Squares",
             "Collapses each model's 4,096-dim activations onto the few latent directions aligned "
             "with the date axis. <b>Solves:</b> high dimensionality — finds the diachronic "
             "direction instead of overfitting noise."),
            ("ρ", "Spearman correlation",
             "Scores the <b>ordering</b> of predicted vs. true dates, not the exact year. "
             "<b>Solves:</b> dating is ordinal, and one scale-free number lets us compare a 400M "
             "encoder against a 120B decoder fairly."),
            ("μ", "Mean pooling",
             "Averages over all tokens rather than reading the last one. <b>Solves:</b> Akkadian is "
             "many short syllabic tokens — the document-level signal lives in the average."),
            ("✂", "Maximal cleaning",
             "Removes royal names, logograms, determinatives and formulae. <b>Solves:</b> separates "
             "<i>apparent</i> dating ability from <i>causal</i> linguistic signal."),
            ("⚖", "Balanced (200 MC draws)",
             "Monte-Carlo resampling + GroupKFold by ruler. <b>Solves:</b> 76% of fragments are "
             "three Sargonid kings — stops a model from scoring high by just guessing the "
             "majority, and blocks ruler leakage across folds."),
        ],
    },
    {
        "kind": "figure",
        "eyebrow": "Why balanced · 1",
        "title": "The imbalance trap",
        "fig": "v_1/src/geodesic/fig1_followups/error_overlap/predictions_maximal_balanced/accuracy_at_N.png",
        "takeaway": "Because three Sargonid kings make up 76% of the dated corpus, a dummy that just "
                    "guesses the median date already looks accurate — so raw accuracy is "
                    "meaningless and we had to score differently.",
        "fact": "Accuracy@±N years vs. a predict-the-mean dummy (grey). Real models barely clear "
                "the dummy at most tolerances — the metric, not the model, is the problem.",
    },
    {
        "kind": "figure",
        "eyebrow": "Why balanced · 2",
        "title": "Where a real model actually earns its keep",
        "fig": "v_1/src/geodesic/fig1_followups/error_overlap/predictions_maximal_balanced/balanced_maximal_chronology.png",
        "takeaway": "Balanced scoring exposes the truth: the dummy owns the crowded centre of the "
                    "timeline, and a genuine model only pulls ahead at the chronological extremes "
                    "— which is exactly where dating is hard and valuable.",
        "fact": "Right panel: median error by true date. Thalesian wins the early/late extremes; "
                "TF-IDF only beats the dummy in the dense Sargonid centre.",
    },
    {
        "kind": "figure",
        "eyebrow": "The headline result",
        "title": "A 400M model beats every giant",
        "fig": "v_1/src/chronorank/autopsy/results/figures/factor_ladder_bars.png",
        "takeaway": "Under our honest protocol, a 400M translation-finetuned model (Thalesian) tops "
                    "every LLM up to 120B — and a Qwen with fully random weights ties the trained "
                    "giants, so scale and pretraining are not what's doing the dating.",
        "fact": "Balanced, maximal, best-layer year-PLS Spearman. Thalesian 0.411 > Qwen3-8B 0.363 > "
                "32B 0.340 > gpt-oss-120B 0.333; random floor 0.301; vanilla uMT5 (no finetune) 0.297.",
    },
    {
        "kind": "figure",
        "eyebrow": "The finetune analysis · 1",
        "title": "Scale and next-token finetuning add nothing",
        "fig": "v_1/src/finetune/results/figures/maximal_pls_bestlayer.png",
        "takeaway": "Growing the model to 120B, or finetuning it on Akkadian with next-token "
                    "prediction, lands every arm right on top of its untouched base — more "
                    "parameters and more exposure buy no dating signal.",
        "fact": "Each “+NTP (ft)” bar sits within noise of its base bar. The lever is clearly "
                "not size and not raw next-token exposure.",
    },
    {
        "kind": "figure",
        "eyebrow": "The finetune analysis · 2",
        "title": "Next-token finetuning is a no-op, layer by layer",
        "fig": "v_1/src/finetune/results/figures/ftcurves_qwen3_8b_maximal.png",
        "takeaway": "Across every layer, the next-token-finetuned Qwen curves sit exactly on the base "
                    "curve — confirming it's the training <b>objective</b>, not the amount of "
                    "Akkadian seen, that's missing.",
        "fact": "Qwen3-8B, all NTP checkpoints (ft00–ft32) overlay the base. This is what sends us "
                "toward a translation/seq2seq objective instead.",
    },
    {
        "kind": "figure",
        "eyebrow": "The tokenizer analysis",
        "title": "It is not the tokenizer either",
        "fig": "v_1/src/chronorank/autopsy/results/figures/fertility_by_corpus.png",
        "takeaway": "Thalesian wins despite having the <b>least</b> efficient Akkadian tokenizer of the "
                    "whole group — it spends the most tokens per word — so cheaper or smarter "
                    "tokenization is not the explanation.",
        "fact": "Fertility = tokens per Akkadian word (lower = better). Thalesian is highest on every "
                "corpus; the win survives a worse tokenizer.",
    },
    {
        "kind": "figure",
        "eyebrow": "What carries the win",
        "title": "The translation finetune builds the signal",
        "fig": "v_1/src/chronorank/autopsy/results/figures/factor_ladder_layerwise.png",
        "takeaway": "The un-finetuned uMT5 base sits at the random floor and decays with depth; only "
                    "the cuneiform <b>translation</b> finetune grows a dating representation that "
                    "deepens through the network — the objective that aligns surface text with "
                    "meaning is what matters.",
        "fact": "Depth profile. uMT5 base (red) peaks at the embedding layer then falls below floor; "
                "Thalesian (green) rises to 0.41 at layer 10. Δ finetune = +0.114.",
    },
    {
        "kind": "text",
        "eyebrow": "What we learned",
        "title": "A negative result, productively framed",
        "body": [
            ("The one-sentence summary",
             "Today's large LLMs do not yet date Akkadian to philological standards — and our "
             "protocol shows <b>why</b>, with controls, rather than just reporting a low score."),
            ("They date on surface distribution, not learned structure",
             "Random-weight Qwen ties the trained ones, and scale is flat from 1.7B to 120B — "
             "whatever lets them order texts is recoverable from architecture, not from learned "
             "Akkadian."),
            ("The win is the objective",
             "A 400M model trained to <b>translate</b> Akkadian beats them all. Not tokenizer "
             "(it's the worst), not architecture (the base is at the floor), not next-token "
             "finetuning (a no-op) — the seq2seq/translation objective is what instills the signal."),
            ("The method is the contribution",
             "The maximal-balanced PLS protocol — with its controls (random floor, NTP-null, "
             "tokenizer audit, balanced resampling) — is a reusable benchmark for separating "
             "apparent from causal dating ability."),
        ],
    },
    {
        "kind": "text",
        "eyebrow": "Wrapping into a thesis",
        "title": "What is done, and where this sits",
        "body": [
            ("Done — ready to write up",
             "(1) Unified 5-database corpus + cleaning pipeline. (2) The maximal-balanced PLS "
             "benchmark protocol. (3) The model comparison (TF-IDF / MLM / Qwen 1.7–32B / "
             "gpt-oss-120B / Thalesian / random). (4) The Thalesian autopsy: finetune, not "
             "tokenizer or architecture."),
            ("In progress",
             "Manifold / geodesic structure of the embeddings; error-overlap analysis across models."),
            ("The thesis argument it all supports",
             "“We show large LLMs cannot genuinely represent Akkadian diachronic change, build a "
             "controlled protocol that proves why, and identify the translation objective as the "
             "factor that does work.”"),
            ("For discussion today",
             "Which of these becomes the central chapter, and what is the minimum still missing to "
             "converge on a complete M.Sc. thesis (and a possible publication)."),
        ],
    },
]


def render_slide(idx, s):
    n = idx + 1
    kind = s["kind"]
    eyebrow = s.get("eyebrow", "")
    eyebrow_html = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""

    if kind == "title":
        inner = f"""
          <div class="title-block">
            <h1 class="big-title">{s['title']}</h1>
            <div class="subtitle">{s['subtitle']}</div>
            <div class="meta">{s['meta']}</div>
            <div class="takeaway title-takeaway">{s['takeaway']}</div>
          </div>"""
    elif kind == "text":
        rows = "".join(
            f'<div class="point"><div class="point-h">{h}</div>'
            f'<div class="point-t">{t}</div></div>'
            for h, t in s["body"]
        )
        inner = f"""
          {eyebrow_html}
          <h2 class="slide-title">{s['title']}</h2>
          <div class="points">{rows}</div>"""
    elif kind == "method":
        cards = "".join(
            f'<div class="mcard"><div class="micon">{ic}</div>'
            f'<div class="mname">{name}</div>'
            f'<div class="mtext">{txt}</div></div>'
            for ic, name, txt in s["items"]
        )
        inner = f"""
          {eyebrow_html}
          <h2 class="slide-title">{s['title']}</h2>
          <div class="lead">{s['lead']}</div>
          <div class="mgrid">{cards}</div>"""
    elif kind == "figure":
        uri = data_uri(s["fig"])
        fact = f'<div class="fact">{s["fact"]}</div>' if s.get("fact") else ""
        inner = f"""
          {eyebrow_html}
          <h2 class="slide-title">{s['title']}</h2>
          <div class="figwrap"><img src="{uri}" alt="{s['title']}"></div>
          <div class="takeaway"><span class="tk-label">Take away</span>{s['takeaway']}</div>
          {fact}"""
    else:
        inner = ""

    return f'<section class="slide slide-{kind}" data-index="{idx}">{inner}</section>'


def build():
    slides_html = "\n".join(render_slide(i, s) for i, s in enumerate(SLIDES))
    total = len(SLIDES)
    titles = [s.get("title", f"Slide {i+1}") for i, s in enumerate(SLIDES)]

    html = TEMPLATE.replace("__SLIDES__", slides_html)
    html = html.replace("__TOTAL__", str(total))
    html = html.replace("__TITLES__", json.dumps(titles))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_story.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(out) / 1e6
    print(f"Wrote {out}  ({size_mb:.1f} MB, {total} slides)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decoding Linguistic Evolution &mdash; Thesis Story</title>
<style>
  :root{
    --bg:#0f1419; --panel:#fbfaf7; --ink:#1a1f24; --muted:#5b6670;
    --accent:#1f6f5c; --accent2:#a3341f; --line:#e3ddd0; --gold:#b8860b;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  html,body{height:100%;}
  body{background:var(--bg);font-family:var(--sans);color:var(--ink);
       overflow:hidden;}
  .stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
         padding:3vh 3vw;}
  .slide{display:none;width:min(1180px,94vw);height:min(720px,90vh);
         background:var(--panel);border-radius:14px;
         box-shadow:0 24px 70px rgba(0,0,0,.55);
         padding:46px 60px 64px;position:relative;flex-direction:column;}
  .slide.active{display:flex;animation:fade .35s ease;}
  @keyframes fade{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}

  .eyebrow{font-size:13px;letter-spacing:.18em;text-transform:uppercase;
           color:var(--accent);font-weight:700;margin-bottom:10px;}
  .slide-title{font-family:var(--serif);font-size:34px;line-height:1.12;
               color:var(--ink);margin-bottom:14px;}
  .big-title{font-family:var(--serif);font-size:56px;line-height:1.05;letter-spacing:-.5px;}

  /* title slide */
  .slide-title.slide{justify-content:center;}
  .title-block{margin:auto 0;}
  .subtitle{font-family:var(--serif);font-size:25px;color:var(--accent);
            margin-top:14px;font-style:italic;}
  .meta{margin-top:26px;color:var(--muted);font-size:16px;line-height:1.7;}
  .title-takeaway{margin-top:34px;max-width:760px;}

  /* figure */
  .figwrap{flex:1;display:flex;align-items:center;justify-content:center;
           min-height:0;margin:4px 0 16px;}
  .figwrap img{max-width:100%;max-height:100%;object-fit:contain;
               border-radius:8px;border:1px solid var(--line);
               background:#fff;padding:6px;}

  .takeaway{background:linear-gradient(0deg,#f3efe6,#faf8f2);
            border-left:5px solid var(--accent);border-radius:8px;
            padding:15px 20px;font-size:19px;line-height:1.45;color:var(--ink);}
  .tk-label{display:inline-block;font-size:11px;letter-spacing:.16em;
            text-transform:uppercase;color:var(--accent);font-weight:800;
            margin-right:12px;vertical-align:1px;}
  .fact{margin-top:10px;font-size:14px;color:var(--muted);line-height:1.5;
        padding-left:25px;}

  /* text slides */
  .points{display:flex;flex-direction:column;gap:18px;margin-top:8px;
          overflow:auto;padding-right:6px;}
  .point-h{font-family:var(--serif);font-size:20px;color:var(--accent2);
           margin-bottom:4px;}
  .point-t{font-size:18px;line-height:1.5;color:var(--ink);}
  .point{border-left:3px solid var(--line);padding-left:18px;}

  /* method slide */
  .lead{font-size:18px;line-height:1.5;color:var(--ink);margin-bottom:18px;
        max-width:900px;}
  .mgrid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;flex:1;
         min-height:0;}
  .mcard{background:#fff;border:1px solid var(--line);border-radius:10px;
         padding:16px 14px;display:flex;flex-direction:column;}
  .micon{font-size:26px;color:var(--accent);font-weight:700;margin-bottom:6px;}
  .mname{font-family:var(--serif);font-size:17px;font-weight:700;
         margin-bottom:8px;color:var(--ink);}
  .mtext{font-size:13px;line-height:1.45;color:var(--muted);}
  .mtext b{color:var(--accent2);}

  /* chrome */
  .chrome{position:fixed;left:0;right:0;bottom:18px;display:flex;
          align-items:center;justify-content:center;gap:18px;z-index:10;}
  .btn{background:rgba(255,255,255,.12);color:#fff;border:1px solid rgba(255,255,255,.25);
       width:44px;height:44px;border-radius:50%;font-size:20px;cursor:pointer;
       display:flex;align-items:center;justify-content:center;transition:.15s;}
  .btn:hover{background:rgba(255,255,255,.25);}
  .counter{color:#cfd6dd;font-size:14px;font-variant-numeric:tabular-nums;
           min-width:120px;text-align:center;}
  .progress{position:fixed;top:0;left:0;height:3px;background:var(--gold);
            transition:width .3s;z-index:20;}
  .slidename{position:fixed;top:14px;right:20px;color:rgba(255,255,255,.5);
             font-size:12px;letter-spacing:.05em;z-index:10;}
  .hint{position:fixed;bottom:18px;right:20px;color:rgba(255,255,255,.35);
        font-size:12px;z-index:10;}
  @media print{
    body{overflow:visible;background:#fff;}
    .stage{position:static;padding:0;}
    .slide{display:flex!important;page-break-after:always;box-shadow:none;
           width:100%;height:100vh;border-radius:0;}
    .chrome,.progress,.slidename,.hint{display:none;}
  }
</style>
</head>
<body>
  <div class="progress" id="progress"></div>
  <div class="slidename" id="slidename"></div>
  <div class="stage">
    __SLIDES__
  </div>
  <div class="chrome">
    <button class="btn" id="prev" title="Previous (←)">&#8249;</button>
    <div class="counter" id="counter"></div>
    <button class="btn" id="next" title="Next (→)">&#8250;</button>
  </div>
  <div class="hint">&larr; &rarr; navigate &middot; F fullscreen</div>
<script>
  const TOTAL = __TOTAL__;
  const TITLES = __TITLES__;
  let i = 0;
  const slides = document.querySelectorAll('.slide');
  function show(n){
    i = Math.max(0, Math.min(TOTAL-1, n));
    slides.forEach((s,k)=>s.classList.toggle('active', k===i));
    document.getElementById('counter').textContent = (i+1)+' / '+TOTAL;
    document.getElementById('progress').style.width = ((i+1)/TOTAL*100)+'%';
    document.getElementById('slidename').textContent = TITLES[i];
    location.hash = i+1;
  }
  document.getElementById('next').onclick = ()=>show(i+1);
  document.getElementById('prev').onclick = ()=>show(i-1);
  document.addEventListener('keydown', e=>{
    if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown') show(i+1);
    else if(e.key==='ArrowLeft'||e.key==='PageUp') show(i-1);
    else if(e.key==='Home') show(0);
    else if(e.key==='End') show(TOTAL-1);
    else if(e.key==='f'||e.key==='F'){
      if(!document.fullscreenElement) document.documentElement.requestFullscreen();
      else document.exitFullscreen();
    }
  });
  const start = parseInt((location.hash||'#1').slice(1),10);
  show(isNaN(start)?0:start-1);
</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
