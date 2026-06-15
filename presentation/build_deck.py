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
        "title": "Honest Computational Dating of Low-Resource Akkadian",
        "subtitle": "Confounder-controlled chronological probing across embedding sources",
        "meta": (
            "M.Sc. Thesis &middot; Yarin Beer &middot; Computer Science, HUJI<br>"
            "Advisors: Prof. Nathan Wasserman &nbsp;·&nbsp; Dr. Barak Sober &nbsp;·&nbsp; Prof. Gabriel Stanovsky"
        ),
    },

    # 2 ── THE QUESTION ─────────────────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "1 — The problem and our approach",
        "title": "Dating Akkadian is expert, slow, and uncertain — can it be done computationally in a way scholars can trust?",
        "body": [
            (
                "The scholarly problem",
                "Assyriologists date cuneiform texts by hand: ruler names, script style, "
                "archival context. It is subjective, expertise-bound, and for many fragments "
                "the date remains uncertain. We ask whether computation can help — "
                "and more importantly, whether it can be done in a way a philologist "
                "can actually trust and inspect."
            ),
            (
                "Why not just run a year regressor?",
                "With a tiny labelled test set, a powerful regressor posts strong test numbers "
                "while learning surface shortcuts that generalise poorly. "
                "We instead cast dating as a <strong>chronological-ordering</strong> task: "
                "probe frozen text embeddings with a deliberately weak linear model "
                "and measure <strong>Spearman rank correlation</strong> against true date. "
                "This resists the overfitting and shortcuts regression invites."
            ),
            (
                "The honest-evaluation spine",
                "Every design choice below exists so the system measures <em>dating</em> "
                "and nothing merely correlated with it — genre, ruler name, class imbalance. "
                "A model that exploits those shortcuts scores well but teaches us nothing "
                "and generalises to nothing. "
                "The protocol we built to close those doors is the methodological contribution."
            ),
        ],
    },

    # 3 ── BEST LAYER PER MODEL ─────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "3 — Where in each model does the year signal live?",
        "title": "Thalesian's signal deepens with layer; LLMs peak mid-network then decay",
        "fig": "v_1/src/geodesic/maximal_figs/figures/fig4_maximal_A.png",
        "takeaway": (
            "Each model's year-PLS Spearman across all layers (balanced · maximal · mean). "
            "Thalesian-400M (dark brown) rises steadily to its best at layer 10 (★). "
            "Qwen models (green dashes) peak around layers 15–16 then fall. "
            "Most models are at or below TF-IDF (blue dotted) "
            "except Thalesian — which climbs well above it. "
            "We use each model's starred best layer for the main comparison."
        ),
        "note": (
            "Random-8B (purple dotted) sits flat near 0.30 across all layers — "
            "this is the random floor: what the architecture gives you for free "
            "before any training. Trained LLMs barely clear it."
        ),
    },

    # 4 ── K SWEEP ──────────────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "4 — How many PLS dimensions does the date signal need?",
        "title": "k = 3–5 components captures the date signal; Ridge (dashed) confirms PLS is not cherry-picking",
        "fig": "v_1/src/geodesic/maximal_figs/figures/ksweep_tradeoff_maximal.png",
        "takeaway": (
            "Spearman ρ vs. number of PLS components k for all models (solid lines). "
            "Most models plateau between k = 3 and k = 5 — the chronological information "
            "in the embedding fits in very few latent directions. "
            "Dashed lines show Ridge regression (all dimensions) at each model's best layer: "
            "PLS matches or beats Ridge at k ≈ 3, proving the compression is principled, "
            "not overfitting."
        ),
        "note": (
            "We use k = 3 in the fixed-k validation and allow the MC draws to pick "
            "the best k per draw in the headline comparison — "
            "both give consistent rankings (see PLS vs. Ridge slide)."
        ),
    },

    # 5 ── PLS VS RIDGE ─────────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "5 — Validation: PLS methodology is sound",
        "title": "PLS at fixed k = 3 matches Ridge on all four representative models",
        "fig": "v_1/src/geodesic/fig1_followups/pls_ksweep/fixed_k3_pls_vs_ridge.png",
        "takeaway": (
            "Fair head-to-head: PLS at k = 3 (blue) vs. Ridge with all embedding dimensions (red), "
            "each at its own best layer. "
            "PLS matches Ridge for MLM (0.382 vs 0.408) and Thalesian (0.376 vs 0.384), "
            "and <em>beats</em> Ridge for Qwen3-32B (0.375 vs 0.326). "
            "Using PLS is not cherry-picking — it recovers the same signal with a fraction of the dimensions."
        ),
        "note": (
            "This removes the concern that per-draw best-k selection inflated results. "
            "The ranking (Thalesian ≈ MLM > TF-IDF > Qwen) is preserved under both methods."
        ),
    },

    # 6 ── NTP + SCALE NULL (was slide 9) ───────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "6 — Does finetuning or scale help?",
        "title": "NTP finetuning and model scale both do nothing",
        "fig": "v_1/src/finetune/results/figures/maximal_pls_bestlayer.png",
        "takeaway": (
            "Every '+NTP (ft)' bar sits within noise of its untouched base model "
            "across all scales from 1.7B to 120B. "
            "We finetuned all Qwen sizes on our full Akkadian corpus with next-token prediction — "
            "zero improvement everywhere. "
            "And the random-weight Qwen (0.301) ties the trained giants (0.33–0.36): "
            "LLM pretraining provides almost no advantage over random weights on this task."
        ),
        "note": (
            "We also tested zero-shot, few-shot, and chain-of-thought prompting: no improvement. "
            "Qwen can name Akkadian rulers and their dates correctly in plain English, "
            "yet that declarative knowledge does not surface as a decodable temporal representation."
        ),
    },

    # 7 ── HEADLINE + AUTOPSY SETUP (was slide 8) ────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "7 — The result and the autopsy question",
        "title": "A 400M translation model beats every LLM — what explains the win?",
        "fig": "v_1/src/chronorank/autopsy/results/figures/factor_ladder_bars.png",
        "takeaway": (
            "Thalesian-400M (0.411) leads; Qwen3-8B (0.363) and all larger models cluster "
            "just above the dashed random floor at 0.30. "
            "The <strong>random floor</strong> (bar labelled \"random (floor)\", dashed line) "
            "is a Qwen3-8B with fully randomized weights — scoring 0.301. "
            "Trained LLMs are only 3–6 points above a random network. "
            "The red bar at far right is vanilla uMT5-base (no finetune) at 0.297 — "
            "at the random floor. So the entire Thalesian advantage "
            "comes from the finetune alone (Δ = +0.114)."
        ),
        "note": (
            "Next three slides ask: is it the tokenizer? the architecture? the training objective? "
            "Each question is answered by one controlled comparison."
        ),
    },

    # 8 ── HEADLINE RESULT (AFTER AUTOPSY SETUP) ────────────────────────────────
    # (factor_ladder_bars is already shown as slide 7/autopsy — this is now the
    #  scale-scatter view if we want, or we re-use after the tokenizer/FT analysis)

    # 9 ── NTP + SCALE NULL ─────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "8 — Does scale or Akkadian finetuning help?",
        "title": "Scale and NTP finetuning both do nothing — the training objective is what matters",
        "fig": "v_1/src/finetune/results/figures/maximal_pls_bestlayer.png",
        "takeaway": (
            "Every '+NTP (ft)' bar lands within noise of its untouched base model "
            "across all scales from 1.7B to 120B. "
            "We finetuned Qwen on all available Akkadian text with next-token prediction — "
            "zero improvement at any scale. "
            "Queried in plain English, Qwen correctly names Akkadian rulers and dates; "
            "yet that declarative knowledge does not appear as a linearly decodable "
            "temporal representation in the hidden states."
        ),
        "note": (
            "Zero-shot, few-shot, and chain-of-thought prompting also produced no gains. "
            "The evidence points to the training <em>objective</em> as the missing ingredient, "
            "not the amount of Akkadian data or the number of parameters."
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

    # 12 ── WHY THE SMALL MODEL WINS — INTERPRETABILITY AS SUPPORT ─────────────
    {
        "kind": "figure",
        "eyebrow": "11 — Why does the small translation model win? (Interpretability as support)",
        "title": "LLM embeddings entangle time with genre — the internal timeline does not surface linearly",
        "fig": "v_1/src/linear_probing/results/letters__probe_cls__period/figures/tsne_best_layer.png",
        "takeaway": (
            "Prior work reports an internal LLM timeline recoverable by linear probing. "
            "Under our protocol it does not surface for Akkadian: random-weight Qwen "
            "nearly ties the trained giants. "
            "The t-SNE shows <em>why</em>: period structure exists in LLM embeddings (left, raw text) "
            "but collapses when confounders are cleaned away (right, maximal). "
            "Chronology is entangled with genre and provenance on a likely nonlinear manifold — "
            "exactly the form a linear probe cannot reach, and the translation model "
            "sidesteps by packaging chronological signal in a linearly decodable form."
        ),
        "note": (
            "Open experiment: a small MLP or kernel probe on the same Qwen embeddings "
            "would distinguish 'signal absent' from 'signal nonlinearly entangled' — "
            "both outcomes are publishable and close the main reviewer objection. "
            "Visualising Thalesian's PLS component vectors would further confirm "
            "that its chronological signal is genuine and structured."
        ),
    },

    # 13 ── THESIS DISCUSSION ───────────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "12 — Contributions and discussion",
        "title": "What we built, what we found, and what's still open",
        "body": [
            (
                "Contributions (ready to write)",
                "<strong>(1)</strong> The first confounder-controlled method for dating low-resource Akkadian, "
                "cast as chronological-ordering probing — resists overfitting and surface shortcuts. "
                "<strong>(2)</strong> A cleaned royal-inscription benchmark for Akkadian chronology (to be released). "
                "<strong>(3)</strong> A benchmark across TF-IDF / MLM / Qwen 1.7–120B / multilingual seq2seq, "
                "identifying the 400M translation encoder as the recommended dating system. "
                "<strong>(4)</strong> An explanation: the LLM internal timeline from prior work "
                "does not surface linearly for Akkadian — accounting for why scale and prompting fail."
            ),
            (
                "Minimal gap to publication",
                "<strong>(a)</strong> Bootstrap CIs across MC draws — data exists, reporting step only. "
                "<strong>(b)</strong> One nonlinear probe (MLP or kernel) on Qwen — closes the "
                "biggest reviewer hole (absent vs. nonlinearly entangled). "
                "<strong>(c)</strong> Release the benchmark and cleaning pipeline on GitHub/HuggingFace — "
                "ALP and similar venues expect a shipped resource."
            ),
            (
                "In progress",
                "Geodesic / manifold structure of Thalesian's PLS subspace. "
                "Error-overlap analysis across models. "
                "The –ma particle: sixfold OB→LB decline, previously unreported "
                "(paper in prep, Wasserman & Ni)."
            ),
            (
                "Discussion for today",
                "Does the applied framing (dating method + winning model) "
                "feel right for the CS thesis, or should the interpretability angle lead? "
                "Which venue — ALP, ACL Findings, or chapter-first? "
                "What is the minimum still needed to converge?"
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
