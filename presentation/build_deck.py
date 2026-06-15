#!/usr/bin/env python3
"""
Build a self-contained HTML slide deck for the advisor meeting.
Run:  python presentation/build_deck.py
"""
import base64, mimetypes, os, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def img(rel_path):
    path = os.path.join(REPO, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


# ─── SLIDES ───────────────────────────────────────────────────────────────────
SLIDES = [

    # 1 ── TITLE ────────────────────────────────────────────────────────────────
    {
        "kind": "title",
        "title": "Honest, Interpretable Dating of Low-Resource Akkadian",
        "subtitle": "A benchmark protocol, a surprising result, and what it says about LLM temporal representations",
        "meta": (
            "M.Sc. Thesis &middot; Yarin Beer &middot; Computer Science, HUJI<br>"
            "Advisors: Prof. Nathan Wasserman &nbsp;·&nbsp; Dr. Barak Sober &nbsp;·&nbsp; Prof. Gabriel Stanovsky"
        ),
    },

    # 2 ── THE QUESTION ─────────────────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "1 — The question",
        "title": "Two questions in one",
        "body": [
            (
                "The humanities question",
                "3,000 years of Akkadian literature is largely undated. "
                "Can we build a model that assigns chronological order to texts in an "
                "<strong>interpretable</strong> way — one a philologist can actually trust and inspect?"
            ),
            (
                "The CS question",
                "LLMs are trained on 15 trillion tokens of modern text. "
                "Akkadian has roughly 10 million digitised words — a <em>million-fold</em> less. "
                "Does the broad world-knowledge in a large pretrained model "
                "<strong>transfer</strong> to this tiny, ancient, out-of-distribution language?"
            ),
            (
                'Why "honest" is the hard part',
                "A model can score well on a dating benchmark by exploiting "
                "<strong>corpus artifacts</strong> — royal names, letter-opening formulae, "
                "archive provenance — without learning anything about linguistic change. "
                "Almost every methodological decision in this thesis is a deliberate move "
                "<em>away</em> from inflating results and <em>toward</em> measuring something real."
            ),
        ],
    },

    # 3 ── WHY HONEST EVALUATION IS HARD ────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "2 — The confounders we had to eliminate",
        "title": "Four ways a model can fake a date",
        "body": [
            (
                "Genre / format as period proxy",
                "Almost all Old Babylonian texts in our corpus are <em>letters</em>; "
                "Neo-Assyrian texts are royal inscriptions. "
                "A model that recognises letter-opening formulae dates by genre, not language. "
                "<strong>Fix:</strong> switch to a single homogeneous genre (royal inscriptions) "
                "and aggressively remove structural markers."
            ),
            (
                "Ruler name as year lookup table",
                "Royal inscriptions contain the king's name — "
                "and each king maps almost perfectly to a narrow date range. "
                "Classifying rulers is not dating language; it's reading a name. "
                "<strong>Fix:</strong> maximal cleaning removes names and formulae; "
                "Spearman on chronological order replaces ruler classification."
            ),
            (
                "Class imbalance inflating accuracy",
                "Three Sargonid kings account for 76% of dated inscriptions. "
                "A model that always predicts the majority king's year window "
                "looks accurate even if it has learned nothing. "
                "<strong>Fix:</strong> Monte Carlo balanced resampling over 8 well-attested rulers."
            ),
            (
                "Overclaiming on a 180-year window",
                "Our test corpus spans only 188 years. "
                "Fine-grained year prediction in that window overstates precision. "
                "<strong>Fix:</strong> Spearman rank correlation — we ask for correct "
                "<em>ordering</em>, not exact years."
            ),
        ],
    },

    # 4 ── THE PROTOCOL ─────────────────────────────────────────────────────────
    {
        "kind": "method",
        "eyebrow": "3 — The evaluation protocol",
        "title": "Five deliberate choices — each closing one confound",
        "lead": (
            "This protocol is the methodological contribution. "
            "It is reusable for any low-resource ancient language with dated texts. "
            "Each ingredient addresses one specific failure mode listed on the previous slide."
        ),
        "items": [
            (
                "✂", "Maximal cleaning",
                "Remove royal names, logograms, determinatives, formulae, digits. "
                "<strong>Closes:</strong> genre-format shortcut and ruler-name lookup."
            ),
            (
                "μ", "Mean pooling",
                "Average all token hidden states into one document vector. "
                "<strong>Closes:</strong> single-token assumptions; "
                "the diachronic signal is document-level, not concentrated in one token "
                "(last-token pooling consistently underperformed)."
            ),
            (
                "⚖", "Balanced resampling",
                "200 Monte Carlo draws of 8 balanced rulers, GroupKFold by ruler. "
                "<strong>Closes:</strong> majority-class imbalance and "
                "ruler identity leaking across folds."
            ),
            (
                "PLS", "Partial Least-Squares",
                "Project each embedding onto the latent directions most correlated with date. "
                "<strong>Closes:</strong> the embedding also encodes genre, provenance, location — "
                "PLS isolates only the date-relevant subspace (k≈3–5 optimal)."
            ),
            (
                "ρ", "Spearman correlation",
                "Score chronological <em>ordering</em>, not absolute year prediction. "
                "<strong>Closes:</strong> overclaiming on a narrow window; "
                "one scale-free number allows fair comparison across any model type."
            ),
        ],
    },

    # 5 ── ARTIFACT EVIDENCE ────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "4 — Evidence: the artifact confounders are real",
        "title": "Without our fixes: TF-IDF beats every neural model by reading the king's name",
        "fig": "v_1/src/linear_probing/results/orcc__probe_cls/figures/best_of_ruler.png",
        "takeaway": (
            "On raw ORCC text, TF-IDF (green) reaches 32.5% macro-F1 on ruler classification — "
            "because the king's name appears in the inscription title. "
            "Trained Qwen (blue) falls <em>below</em> the random-weight baseline (purple): "
            "the model is confused, not dating. After maximal cleaning, "
            "TF-IDF collapses to near-random — proving the win was entirely the artifact."
        ),
        "note": (
            "This is the result that forced us off ruler classification entirely. "
            "If the top method wins by reading a name, the benchmark is testing reading, not dating."
        ),
    },

    # 6 ── IMBALANCE EVIDENCE ───────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "5 — Evidence: the imbalance confound is real",
        "title": "A dummy that predicts the corpus mean clears most accuracy thresholds",
        "fig": "v_1/src/geodesic/fig1_followups/error_overlap/predictions_maximal_balanced/accuracy_at_N.png",
        "takeaway": (
            "Because three Sargonid kings dominate the corpus, "
            "a model that always guesses the median year (grey dashed) "
            "already achieves 77% accuracy within ±50 years — "
            "better than any real model at that threshold. "
            "Raw accuracy was measuring class distribution, not dating ability."
        ),
        "note": (
            "This is why we switched to Spearman rank correlation under balanced resampling. "
            "The dummy cannot get a better Spearman score by guessing the mean — "
            "Spearman only rewards correct <em>ordering</em>."
        ),
    },

    # 7 ── SPEARMAN FIX ─────────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "6 — After our fixes: Spearman reveals real structure",
        "title": "Real models outperform the dummy only at the chronological extremes",
        "fig": "v_1/src/geodesic/fig1_followups/error_overlap/predictions_maximal_balanced/balanced_maximal_chronology.png",
        "takeaway": (
            "Under balanced Spearman scoring, Thalesian and TF-IDF beat the dummy "
            "at the early and late ends of the timeline — "
            "exactly where dating is hardest and most useful to philologists. "
            "In the dense Sargonid centre, both collapse to the dummy's level: "
            "this is honest, and it's what an honest protocol should look like."
        ),
        "note": (
            "Right panel: median absolute error by true year (BCE). "
            "Left panel: accuracy-at-N curves confirm all models beat the dummy at ±25 years "
            "but the advantage is modest and meaningful — not inflated."
        ),
    },

    # 8 ── HEADLINE RESULT ──────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "7 — The result",
        "title": "A 400M multilingual translation model beats every LLM up to 120B",
        "fig": "v_1/src/chronorank/autopsy/results/figures/factor_ladder_bars.png",
        "takeaway": (
            "Under our honest protocol, Thalesian-400M (Spearman 0.41) outperforms "
            "Qwen3-8B (0.36), Qwen3-32B (0.34), GPT-OSS-120B (0.33), "
            "and our Akkadian MLM (0.31). "
            "Crucially: a random-weight Qwen-8B scores 0.30 — "
            "statistically tied with the trained giants. "
            "Scale and LLM pretraining are not the lever here."
        ),
        "note": (
            "Vanilla uMT5-base (no fine-tune, red bar far right) = 0.297 — at the random floor. "
            "The full Thalesian advantage (Δ = +0.114) comes from the fine-tune alone, "
            "not from the architecture or pretraining. This is what we investigate in the autopsy."
        ),
    },

    # 9 ── NTP + SCALE NULL ─────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "8 — Ruling out scale and next-token finetuning",
        "title": "Bigger models and Akkadian NTP finetuning both do nothing",
        "fig": "v_1/src/finetune/results/figures/maximal_pls_bestlayer.png",
        "takeaway": (
            "Every '+NTP (ft)' bar sits within noise of its untouched base model — "
            "across all scales from 1.7B to 120B. "
            "We tried finetuning Qwen on our entire Akkadian corpus with next-token prediction: "
            "zero improvement at every scale, at every layer. "
            "The training <em>objective</em> is what's missing, not exposure to Akkadian."
        ),
        "note": (
            "We also tested zero-shot, few-shot, and chain-of-thought prompting — "
            "none improved results. And Qwen can correctly name Akkadian rulers and their date ranges "
            "in plain English, yet that declarative knowledge does not surface "
            "as a linearly decodable temporal representation."
        ),
    },

    # 10 ── TOKENIZER RULED OUT ─────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "9 — Ruling out the tokenizer",
        "title": "Thalesian wins despite the least efficient Akkadian tokenizer",
        "fig": "v_1/src/chronorank/autopsy/results/figures/fertility_by_corpus.png",
        "takeaway": (
            "Thalesian (blue) produces the most subword tokens per Akkadian word "
            "on every corpus in our benchmark — it fragments Akkadian more than any other model. "
            "If tokenizer efficiency drove performance, Thalesian should lose. "
            "It wins across the board, ruling the tokenizer out as the explanation."
        ),
        "note": (
            "GPT-OSS has the most efficient tokenizer (4.43 tokens/word) "
            "and ranks fourth. Thalesian sits at 6.22 tokens/word and ranks first. "
            "Tokenizer quality and dating performance are inversely ordered."
        ),
    },

    # 11 ── TRANSLATION FINETUNE ────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "10 — What does carry the win",
        "title": "The translation fine-tune alone builds a diachronic representation",
        "fig": "v_1/src/chronorank/autopsy/results/figures/factor_ladder_layerwise.png",
        "takeaway": (
            "Vanilla uMT5-base (red) sits at the random floor at the embedding layer "
            "and <em>decays below it</em> as depth increases — its deeper layers actively destroy "
            "any date signal. "
            "The cuneiform <strong>translation</strong> fine-tune (green, Thalesian) "
            "builds a representation that rises steadily to 0.41 at layer 10. "
            "Training a model to map Akkadian text to its English meaning "
            "is what instills temporal structure — next-token prediction alone cannot."
        ),
        "note": (
            "This is the key ablation: same architecture, same tokenizer, same size (0.4B) — "
            "the only difference is the fine-tuning objective. "
            "Thalesian 0.411 vs. vanilla uMT5 0.297: Δ = +0.114, entirely from the fine-tune."
        ),
    },

    # 12 ── INTERPRETABILITY / ENTANGLEMENT ─────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "11 — What we found (and didn't find) about LLM temporal representations",
        "title": "Time is encoded — but entangled with genre and provenance",
        "fig": "v_1/src/linear_probing/results/letters__probe_cls__period/figures/tsne_best_layer.png",
        "takeaway": (
            "Prior work claims LLMs carry an internal timeline recoverable by linear probing. "
            "With these tools, we did not find it for Akkadian in large pretrained LLMs — "
            "random weights nearly match trained ones on the same task. "
            "The t-SNE shows that period structure <em>does</em> exist in the embeddings (left, tier-0), "
            "but it collapses under cleaning (right, maximal), suggesting it is entangled with "
            "genre and provenance rather than encoding pure linguistic change."
        ),
        "note": (
            "The open question: is the diachronic signal absent, or is it nonlinear? "
            "A kernel or MLP probe on the same Qwen embeddings would distinguish these two cases. "
            "Two outcomes, both publishable: if weak → 'LLMs don't encode it'; "
            "if strong → 'the signal exists but is nonlinearly entangled, which is why the "
            "translation model wins'."
        ),
    },

    # 13 ── THESIS DISCUSSION ───────────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "12 — Wrapping into a thesis",
        "title": "What's done, what's open, and what to discuss today",
        "body": [
            (
                "Ready to write (solid chapters)",
                "The evaluation protocol and its justification — "
                "maximal cleaning · mean pooling · balanced MC · PLS · Spearman. "
                "The full model comparison (TF-IDF / MLM / Qwen 1.7–32B / gpt-oss-120B / "
                "Thalesian / random / vanilla uMT5). "
                "The autopsy: tokenizer ruled out, NTP ruled out, translation FT is the lever."
            ),
            (
                "Missing for a publication (minimal effort)",
                "(1) Bootstrap CIs on Spearman across MC draws — you have the data, it's reporting. "
                "(2) One nonlinear probe (MLP or kernel PLS) on Qwen — closes the biggest reviewer hole. "
                "(3) Package the cleaned benchmark + pipeline on GitHub/HuggingFace — "
                "nearly every ALP accepted paper ships a resource."
            ),
            (
                "In progress",
                "Geodesic / manifold analysis of Thalesian's PLS subspace. "
                "Error-overlap analysis across models. "
                "The –ma particle: a previously unreported diachronic marker "
                "(rate drops sixfold OB→LB). [Paper in prep, Wasserman & Ni.]"
            ),
            (
                "Discussion for today",
                "Is the methodological protocol + translation-FT finding the thesis core? "
                "What is the minimal experiment still needed? "
                "Is this shaped as an ALP paper, an ACL findings paper, or a chapter first?"
            ),
        ],
    },
]


# ─── HTML RENDERING ───────────────────────────────────────────────────────────

def slide_html(idx, s):
    kind = s["kind"]
    ew = s.get("eyebrow", "")
    ew_html = f'<div class="eyebrow">{ew}</div>' if ew else ""

    if kind == "title":
        return f"""
<section class="slide slide-title" data-index="{idx}">
  <div class="title-inner">
    <div class="title-kicker">M.Sc. Thesis &mdash; Advisor Meeting &mdash; June 2026</div>
    <h1 class="title-h1">{s["title"]}</h1>
    <div class="title-sub">{s["subtitle"]}</div>
    <div class="title-meta">{s["meta"]}</div>
  </div>
</section>"""

    if kind == "text":
        rows = "".join(
            f'<div class="tp"><div class="tp-h">{h}</div><div class="tp-b">{b}</div></div>'
            for h, b in s["body"]
        )
        return f"""
<section class="slide slide-text" data-index="{idx}">
  {ew_html}
  <h2 class="sh">{s["title"]}</h2>
  <div class="text-points">{rows}</div>
</section>"""

    if kind == "method":
        cards = "".join(
            f'<div class="mc"><div class="mc-icon">{ic}</div>'
            f'<div class="mc-name">{nm}</div>'
            f'<div class="mc-body">{body}</div></div>'
            for ic, nm, body in s["items"]
        )
        return f"""
<section class="slide slide-method" data-index="{idx}">
  {ew_html}
  <h2 class="sh">{s["title"]}</h2>
  <p class="method-lead">{s["lead"]}</p>
  <div class="method-grid">{cards}</div>
</section>"""

    if kind == "figure":
        uri = img(s["fig"])
        note = f'<div class="fig-note">{s["note"]}</div>' if s.get("note") else ""
        return f"""
<section class="slide slide-figure" data-index="{idx}">
  {ew_html}
  <h2 class="sh">{s["title"]}</h2>
  <div class="fig-wrap"><img src="{uri}" alt="{s["title"]}"></div>
  <div class="takeaway"><span class="tk-label">Key takeaway</span>{s["takeaway"]}</div>
  {note}
</section>"""

    return f'<section class="slide" data-index="{idx}"><p>Unknown kind: {kind}</p></section>'


def build():
    slides = "\n".join(slide_html(i, s) for i, s in enumerate(SLIDES))
    total = len(SLIDES)
    titles = json.dumps([s.get("title", f"Slide {i+1}") for i, s in enumerate(SLIDES)])
    html = TEMPLATE.replace("__SLIDES__", slides).replace("__TOTAL__", str(total)).replace("__TITLES__", titles)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_story.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, {total} slides)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Akkadian Dating — Thesis Story</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#ddd8cf;
  --white:#ffffff;
  --ink:#1c2028;
  --ink-mid:#3d4552;
  --ink-light:#6b7484;
  --green:#1a5c3a;
  --green-bg:#eaf4ef;
  --green-mid:#2d7a52;
  --red:#8b1a10;
  --border:#dde1e9;
  --border-light:#eef0f5;
  --shadow:0 4px 28px rgba(0,0,0,.14),0 1px 4px rgba(0,0,0,.07);
  --serif:"Iowan Old Style","Palatino Linotype",Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
}
html,body{height:100%;overflow:hidden;background:var(--bg);font-family:var(--sans);}

/* progress bar */
#progress{position:fixed;top:0;left:0;height:3px;background:var(--green);z-index:100;transition:width .3s;}

/* top info bar */
#topbar{position:fixed;top:8px;left:24px;right:24px;display:flex;justify-content:space-between;
        align-items:center;z-index:90;pointer-events:none;}
#slide-label{font-size:11px;letter-spacing:.07em;color:rgba(60,60,60,.6);max-width:55%;}
#counter-top{font-size:11px;color:rgba(60,60,60,.6);font-variant-numeric:tabular-nums;}

/* stage */
.stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
       padding:32px 28px 58px;}

/* slide base */
.slide{display:none;flex-direction:column;
       width:min(1200px,96vw);height:min(700px,88vh);
       background:var(--white);border-radius:12px;
       box-shadow:var(--shadow);
       padding:46px 58px 46px;position:relative;overflow:hidden;}
.slide::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;
               background:linear-gradient(90deg,var(--green) 0%,#2ea86b 100%);}
.slide.active{display:flex;animation:appear .28s ease;}
@keyframes appear{from{opacity:0;transform:translateY(5px);}to{opacity:1;transform:none;}}

/* slide number badge */
.slide::after{content:attr(data-num);position:absolute;bottom:14px;right:20px;
              font-size:10.5px;color:var(--border);font-variant-numeric:tabular-nums;}

/* eyebrow */
.eyebrow{font-size:11px;font-weight:700;letter-spacing:.22em;text-transform:uppercase;
         color:var(--green);margin-bottom:10px;}

/* slide heading */
.sh{font-family:var(--serif);font-size:28px;line-height:1.15;color:var(--ink);
    margin-bottom:16px;max-width:94%;}

/* ── TITLE SLIDE ── */
.slide-title{justify-content:center;}
.title-kicker{font-size:11.5px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;
              color:var(--green);margin-bottom:22px;}
.title-h1{font-family:var(--serif);font-size:48px;line-height:1.08;
          color:var(--ink);letter-spacing:-.3px;margin-bottom:18px;}
.title-sub{font-family:var(--serif);font-size:21px;color:var(--ink-mid);
           font-style:italic;margin-bottom:30px;line-height:1.35;}
.title-meta{font-size:14.5px;line-height:1.9;color:var(--ink-light);}

/* ── TEXT SLIDES ── */
.text-points{display:flex;flex-direction:column;gap:14px;flex:1;min-height:0;overflow:auto;
             padding-right:4px;}
.tp{border-left:3px solid var(--border-light);padding-left:18px;}
.tp:hover{border-left-color:var(--green-mid);}
.tp-h{font-size:16px;font-weight:700;color:var(--green);margin-bottom:4px;}
.tp-b{font-size:16.5px;line-height:1.55;color:var(--ink-mid);}
.tp-b strong,.tp-b em{color:var(--ink);}

/* ── FIGURE SLIDES ── */
.fig-wrap{flex:1;display:flex;align-items:center;justify-content:center;
          min-height:0;margin:2px 0 12px;}
.fig-wrap img{max-width:100%;max-height:100%;object-fit:contain;
              border-radius:5px;border:1px solid var(--border-light);}
.takeaway{background:var(--green-bg);border-left:4px solid var(--green);
          border-radius:7px;padding:12px 18px;font-size:16.5px;line-height:1.5;color:var(--ink);}
.tk-label{display:inline-block;font-size:10px;letter-spacing:.18em;text-transform:uppercase;
          font-weight:800;color:var(--green);margin-right:10px;vertical-align:2px;}
.fig-note{margin-top:8px;font-size:13px;color:var(--ink-light);line-height:1.5;
          padding-left:22px;}

/* ── METHOD SLIDE ── */
.method-lead{font-size:15.5px;line-height:1.55;color:var(--ink-mid);margin-bottom:16px;}
.method-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:11px;flex:1;min-height:0;}
.mc{background:#f8f9fb;border:1px solid var(--border);border-radius:9px;
    padding:15px 13px;display:flex;flex-direction:column;gap:7px;}
.mc-icon{font-size:20px;font-weight:800;color:var(--green);}
.mc-name{font-family:var(--serif);font-size:14.5px;font-weight:700;color:var(--ink);}
.mc-body{font-size:12px;line-height:1.45;color:var(--ink-light);}
.mc-body strong{color:var(--red);}

/* ── BOTTOM NAV ── */
#chrome{position:fixed;bottom:12px;left:0;right:0;display:flex;align-items:center;
        justify-content:center;gap:14px;z-index:90;}
.btn{background:rgba(255,255,255,.75);border:1px solid rgba(0,0,0,.18);
     width:36px;height:36px;border-radius:50%;font-size:17px;cursor:pointer;
     display:flex;align-items:center;justify-content:center;color:#333;
     box-shadow:0 1px 4px rgba(0,0,0,.1);transition:.13s;}
.btn:hover{background:var(--white);box-shadow:0 2px 8px rgba(0,0,0,.18);}
#hint{position:fixed;bottom:12px;right:16px;font-size:11px;color:rgba(0,0,0,.32);z-index:90;}

@media print{
  body{overflow:visible;background:#fff;}
  .stage{position:static;padding:0;}
  .slide{display:flex!important;page-break-after:always;box-shadow:none;
         width:100%;height:100vh;border-radius:0;}
  #chrome,#progress,#topbar,#hint,.slide::after{display:none;}
}
</style>
</head>
<body>
<div id="progress"></div>
<div id="topbar">
  <div id="slide-label"></div>
  <div id="counter-top"></div>
</div>
<div class="stage">__SLIDES__</div>
<div id="chrome">
  <button class="btn" id="prev">&#8249;</button>
  <button class="btn" id="next">&#8250;</button>
</div>
<div id="hint">← → navigate &nbsp;·&nbsp; F fullscreen</div>
<script>
const TOTAL = __TOTAL__;
const TITLES = __TITLES__;
let cur = 0;
const slides = Array.from(document.querySelectorAll('.slide'));
slides.forEach((s,i) => s.setAttribute('data-num', (i+1)+'/'+TOTAL));

function go(n){
  cur = Math.max(0, Math.min(TOTAL-1, n));
  slides.forEach((s,i) => s.classList.toggle('active', i===cur));
  document.getElementById('counter-top').textContent = (cur+1)+' / '+TOTAL;
  document.getElementById('progress').style.width = ((cur+1)/TOTAL*100)+'%';
  document.getElementById('slide-label').textContent = TITLES[cur]||'';
  history.replaceState(null,'','#'+(cur+1));
}
document.getElementById('next').onclick = ()=>go(cur+1);
document.getElementById('prev').onclick = ()=>go(cur-1);
document.addEventListener('keydown', e=>{
  if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){e.preventDefault();go(cur+1);}
  else if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();go(cur-1);}
  else if(e.key==='Home') go(0);
  else if(e.key==='End') go(TOTAL-1);
  else if(e.key.toLowerCase()==='f'){
    if(!document.fullscreenElement) document.documentElement.requestFullscreen().catch(()=>{});
    else document.exitFullscreen();
  }
});
const h = parseInt((location.hash||'#1').slice(1),10);
go(isNaN(h)?0:h-1);
</script>
</body>
</html>"""

if __name__ == "__main__":
    build()
