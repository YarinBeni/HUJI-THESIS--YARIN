#!/usr/bin/env python3
"""
Build a self-contained HTML slide deck for the advisor meeting.
Run:  python presentation/build_deck.py

TWO figures are left as placeholders for you to supply (you said you'd append
them yourself). To fill them, just drop a PNG into presentation/figures/ with
the matching slot name and re-run this script:

    presentation/figures/pls_vs_ridge_thalesian.png   <- "PLS vs Ridge, Thalesian best"
    presentation/figures/qwen_geometry.png            <- "geometry structure of Qwen"

If the file is present it gets embedded automatically; if absent, a labelled
placeholder box is shown instead. No HTML editing required.
"""
import base64, mimetypes, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SLOTS = os.path.join(HERE, "figures")


def img(rel_path):
    """Embed a repo figure as a base64 data URI."""
    path = os.path.join(REPO, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def slot_uri(slot):
    """Return data URI for presentation/figures/<slot>.png if it exists, else None."""
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(SLOTS, slot + ext)
        if os.path.exists(p):
            mime = mimetypes.guess_type(p)[0] or "image/png"
            with open(p, "rb") as f:
                return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    return None


# ─── SLIDES ───────────────────────────────────────────────────────────────────
SLIDES = [

    # 1 ── TITLE ────────────────────────────────────────────────────────────────
    {
        "kind": "title",
        "title": "Honest Computational Dating of Low-Resource Akkadian",
        "subtitle": "A confounder-controlled benchmark, and the model that best supports automatic dating",
        "meta": (
            "M.Sc. Thesis &middot; Yarin Beer &middot; Computer Science, HUJI<br>"
            "Advisors: Prof. Nathan Wasserman &nbsp;·&nbsp; Dr. Barak Sober &nbsp;·&nbsp; Prof. Gabriel Stanovsky"
        ),
    },

    # 2 ── PROBLEM 1 — THE HOOK ─────────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "1 — The problem",
        "title": "Dating cuneiform is done by hand — can a computer do it in a way scholars trust?",
        "body": [
            (
                "How dating works today",
                "Assyriologists date cuneiform texts by hand, leaning on ruler names, "
                "script style, and archival context — a subjective, expertise-bound, "
                "time-consuming process. For many fragments the date stays uncertain."
            ),
            (
                "What we ask",
                "Can this be done computationally in a way scholars can <strong>trust</strong>? "
                "A method that measures <em>dating</em> rather than a surface confounder, "
                "and whose predictions reflect chronological signal actually present in the text."
            ),
            (
                "What we deliver",
                "We build such a method, benchmark a wide range of embedding sources under it, "
                "and identify <strong>which model best supports automatic dating of Akkadian</strong>."
            ),
        ],
    },

    # 3 ── PROBLEM 2 — THE REFRAME ──────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "2 — The reframe",
        "title": "Honest dating is not year regression — it is chronological ordering",
        "body": [
            (
                "Why not regression",
                "With a small labelled test set, a year regressor posts strong test numbers "
                "while learning shortcuts that generalise poorly. "
                "The numbers look good; the model has learned nothing about language change."
            ),
            (
                "Our task definition",
                "We cast dating as a <strong>chronological-ordering</strong> task: "
                "probe frozen text embeddings with a deliberately <em>weak</em> linear model "
                "and measure <strong>Spearman rank correlation</strong> against true chronological order. "
                "A weak probe tests whether the signal is present and decodable — "
                "it cannot overfit the tiny test set."
            ),
            (
                "The spine of the method",
                "Every design choice that follows exists for one reason: "
                "so the system measures <em>dating</em> and nothing merely correlated with it — "
                "genre, ruler name, or class imbalance."
            ),
        ],
    },

    # 4 ── NUTSHELL: WHAT WE DID (PROTOCOL) ─────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "3 — In a nutshell: what we did",
        "title": "An honest-evaluation protocol — each step closes a dishonest shortcut",
        "body": [
            (
                "Remove the genre &amp; name shortcuts",
                "Restrict to one homogeneous genre (royal inscriptions); "
                "aggressively normalize — strip signs, numbers, normalize order. "
                "This <em>flipped TF-IDF from winner to loser</em>, direct evidence the confounder was real."
            ),
            (
                "Control imbalance &amp; the ruler-lookup leak",
                "A few kings dominate, so Monte-Carlo resample (after Wasserman) "
                "down to 8 rulers with ≥21 fragments each (from 38). "
                "Score chronological <em>ordering</em>, not ruler classification — "
                "which would degenerate into a year lookup table."
            ),
            (
                "Measure modestly and isolate the year direction",
                "Spearman over a ~180-year span — fine-grained year prediction would overclaim. "
                "Use <strong>PLS</strong> to extract only the chronology-relevant directions "
                "before correlating, since embeddings entangle time with genre and location."
            ),
        ],
    },

    # 5 ── NUTSHELL: WHAT WE COMPARED ───────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "4 — In a nutshell: what we compared",
        "title": "One protocol, many embedding sources",
        "body": [
            (
                "Classical &amp; from-scratch baselines",
                "TF-IDF over lemmata n-grams · a masked language model trained from scratch "
                "on our Akkadian corpus · a random-weight network (architecture-only control)."
            ),
            (
                "Large pretrained LLMs (the scale question)",
                "Qwen3 at 1.7B / 8B / 32B · GPT-OSS ~120B. "
                "Probed layer-by-layer; also tested zero-shot, few-shot, and chain-of-thought prompting."
            ),
            (
                "Seq2seq translation encoders (the transfer question)",
                "A multilingual low-resource translation encoder (~400M) · "
                "its Akkadian-only variant (~300M) · the un-finetuned base. "
                "Uniquely, these were trained to <em>translate</em> ancient languages, not just predict tokens."
            ),
        ],
    },

    # 6 ── NUTSHELL: WHAT WE FOUND ──────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "5 — In a nutshell: what we found",
        "title": "A 400M translation encoder is the best dating model — and we explain why",
        "body": [
            (
                "The recommended system",
                "<strong>A 400M multilingual translation encoder beats every other source, "
                "including the 120B LLM.</strong> "
                "And the multilingual signal — not domain data alone — drives it: "
                "the Akkadian-only 300M does far worse."
            ),
            (
                "What does NOT help",
                "Scale (1.7B→120B: flat) · prompting (zero/few-shot, CoT: no gain) · "
                "Akkadian NTP finetuning of the LLM (no gain). "
                "Mean pooling beats last-token pooling across the board."
            ),
            (
                "Why the small model wins (interpretability as support)",
                "The LLM internal timeline reported in prior work does <em>not</em> surface "
                "for Akkadian under linear probing — explaining why scale and prompting fail. "
                "The translation encoder packages chronological signal in a directly decodable form."
            ),
        ],
    },

    # 7 ── THE EVALUATION PIPELINE (method cards — restored) ─────────────────────
    {
        "kind": "method",
        "eyebrow": "6 — The evaluation pipeline",
        "title": "Maximal · mean-pool · balanced · PLS · Spearman",
        "lead": (
            "The five-stage pipeline applied identically to every embedding source. "
            "This is the reusable methodological contribution — "
            "valid for any low-resource ancient language with some dated texts."
        ),
        "items": [
            (
                "✂", "Maximal cleaning",
                "Strip royal names, logograms, determinatives, formulae, digits. "
                "<strong>Closes:</strong> genre-format and ruler-name shortcuts."
            ),
            (
                "μ", "Mean pooling",
                "Average all token hidden states into one document vector. "
                "<strong>Closes:</strong> the last-token assumption — "
                "the diachronic signal is document-level."
            ),
            (
                "⚖", "Balanced MC",
                "200 Monte-Carlo draws over 8 balanced rulers, GroupKFold by ruler. "
                "<strong>Closes:</strong> majority-class imbalance and ruler leakage."
            ),
            (
                "PLS", "PLS projection",
                "Extract the k≈3–5 latent directions most correlated with date. "
                "<strong>Closes:</strong> entanglement of time with genre and location."
            ),
            (
                "ρ", "Spearman",
                "Score chronological ordering, not absolute year. "
                "<strong>Closes:</strong> overclaiming on a narrow span; "
                "enables fair cross-model comparison."
            ),
        ],
    },

    # 8 ── BEST LAYER PER MODEL ─────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "7 — Where in each model does the year signal live?",
        "title": "Thalesian's signal deepens with layer; LLMs peak mid-network then decay",
        "fig": "v_1/src/geodesic/maximal_figs/figures/fig4_maximal_A.png",
        "takeaway": (
            "Year-PLS Spearman across all layers (balanced · maximal · mean). "
            "Thalesian-400M (dark brown) rises steadily to its best at layer 10 (★), "
            "climbing well above TF-IDF (blue dotted). "
            "Qwen models (green dashes) peak around layers 15–16 then fall back toward the baseline. "
            "The translation model concentrates its signal in late layers; LLMs early-to-middle."
        ),
        "note": (
            "Random-8B (purple dotted) stays flat near 0.30 at every layer — the random floor, "
            "what the architecture gives for free before any training. Trained LLMs barely clear it."
        ),
    },

    # 9 ── K SWEEP ──────────────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "8 — How many dimensions does the date signal need?",
        "title": "k = 3–5 PLS components capture it; Ridge (dashed) confirms PLS is not cherry-picking",
        "fig": "v_1/src/geodesic/maximal_figs/figures/ksweep_tradeoff_maximal.png",
        "takeaway": (
            "Spearman ρ vs. number of PLS components k for every model (solid lines). "
            "Most plateau between k = 3 and k = 5 — chronological information lives in "
            "a very low-dimensional subspace of the embedding. "
            "Dashed lines (Ridge, all dimensions) sit at the same level: "
            "PLS recovers the full signal with a handful of directions, not by overfitting."
        ),
        "note": (
            "Low intrinsic dimensionality is itself a finding — "
            "the chronological axis is simple once the confounders are removed."
        ),
    },

    # 10 ── PLACEHOLDER: PLS vs RIDGE — THALESIAN BEST ──────────────────────────
    {
        "kind": "placeholder",
        "slot": "pls_vs_ridge_thalesian",
        "eyebrow": "9 — The headline result",
        "title": "PLS vs Ridge across all models — the 400M translation encoder wins",
        "drop_hint": "Drop your “PLS vs Ridge / Thalesian-best” screenshot here",
        "takeaway": (
            "Under the honest protocol, the 400M multilingual translation encoder achieves the "
            "highest balanced year-Spearman of every source — above all Qwen scales, the 120B model, "
            "the from-scratch MLM, and TF-IDF. PLS and Ridge agree on the ranking, "
            "so the win is not an artifact of dimensionality reduction. "
            "This is our recommended system for automatic Akkadian dating."
        ),
        "note": (
            "Candidate repo figure if you prefer it to your screenshot: "
            "v_1/src/geodesic/maximal_figs/figures/fig1_maximal_ACD.png  (Panel A = PLS vs Ridge). "
            "To fill: save as presentation/figures/pls_vs_ridge_thalesian.png and re-run build_deck.py."
        ),
    },

    # 11 ── NTP + SCALE NULL ─────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "10 — Does scale or Akkadian finetuning help?",
        "title": "Scale and next-token finetuning both do nothing",
        "fig": "v_1/src/finetune/results/figures/maximal_pls_bestlayer.png",
        "takeaway": (
            "Every '+NTP (ft)' bar lands within noise of its untouched base model, "
            "at every scale from 1.7B to 120B — finetuning Qwen on all our Akkadian text "
            "with next-token prediction yields zero improvement. "
            "Queried in plain English, Qwen names Akkadian rulers and their dates correctly, "
            "yet that declarative knowledge never surfaces as a decodable temporal representation. "
            "The training <em>objective</em> is the missing ingredient, not data or parameters."
        ),
        "note": (
            "Zero-shot, few-shot, and chain-of-thought prompting also produced no gains. "
            "This is what motivates the autopsy: if not scale, data, or prompting — then what?"
        ),
    },

    # 12 ── FACTOR LADDER BARS — random floor explained ─────────────────────────
    {
        "kind": "figure",
        "eyebrow": "11 — The autopsy: what carries the win?",
        "title": "Trained LLMs sit just above a random network; the win is the finetune alone",
        "fig": "v_1/src/chronorank/autopsy/results/figures/factor_ladder_bars.png",
        "takeaway": (
            "Thalesian-400M (0.411) leads; Qwen3-8B (0.363) and all larger models cluster "
            "just above the dashed line at 0.30. "
            "That line is the <strong>random floor</strong> — a Qwen3-8B with fully randomized weights "
            "scoring 0.301 — i.e. what the architecture + probe recover with no training at all. "
            "Trained LLMs beat it by only 3–6 points. "
            "Vanilla uMT5-base (no finetune, far-right red) is at the floor too (0.297): "
            "the entire Thalesian advantage (Δ +0.114) comes from the finetune."
        ),
        "note": (
            "Random floor ≈ 0.30 because residual length/structure leaks a little signal "
            "under maximal cleaning. The bar to beat is the floor, not zero — "
            "and only Thalesian clearly does."
        ),
    },

    # 13 ── TOKENIZER RULED OUT ──────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "12 — Autopsy: is it the tokenizer?",
        "title": "Thalesian wins despite the least efficient Akkadian tokenizer",
        "fig": "v_1/src/chronorank/autopsy/results/figures/fertility_by_corpus.png",
        "takeaway": (
            "Thalesian (blue) splits Akkadian words into more subword tokens than every other model "
            "on every corpus — it is the <em>least</em> efficient tokenizer. "
            "If tokenizer quality drove performance it should lose; it wins. "
            "The tokenizer is ruled out."
        ),
        "note": (
            "GPT-OSS is most efficient (4.43 tok/word) and ranks fourth; "
            "Thalesian (6.22) ranks first. Efficiency and dating skill are inversely ordered."
        ),
    },

    # 14 ── TRANSLATION FINETUNE BUILDS THE SIGNAL ──────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "13 — Autopsy: it is the translation objective",
        "title": "The translation finetune alone builds a deep diachronic representation",
        "fig": "v_1/src/chronorank/autopsy/results/figures/factor_ladder_layerwise.png",
        "takeaway": (
            "Same architecture, same tokenizer, same 0.4B size — only the objective differs. "
            "Vanilla uMT5-base (red) starts at the random floor and <em>decays below it</em> with depth. "
            "The cuneiform translation finetune (green, Thalesian) builds a representation "
            "that rises to 0.41 at layer 10. "
            "Training a model to map Akkadian text to its meaning is what instils temporal structure — "
            "next-token prediction cannot."
        ),
        "note": (
            "Thalesian 0.411 vs. vanilla uMT5 0.297 → Δ = +0.114, entirely from the finetune. "
            "This is the mechanism behind the headline result."
        ),
    },

    # 15 ── PLACEHOLDER: QWEN GEOMETRY ──────────────────────────────────────────
    {
        "kind": "placeholder",
        "slot": "qwen_geometry",
        "eyebrow": "14 — Why the LLM cannot: the geometry of its embedding",
        "title": "Chronology is entangled in Qwen's embedding space, not laid out on a clean axis",
        "drop_hint": "Drop your “geometry structure of Qwen” figure here",
        "takeaway": (
            "Prior work reports a linearly-recoverable internal timeline in LLMs. "
            "For Akkadian it does not surface: the chronological signal is tangled together "
            "with genre and location, plausibly on a non-linear manifold a linear probe cannot reach. "
            "This is why scaling and prompting the LLM do not help — "
            "and why a translation encoder, which packages chronology linearly, wins instead."
        ),
        "note": (
            "Candidate repo figures: v_1/src/geodesic/results/phase_d/phase_d_qwen_maximal_mean_L01_year.png "
            "(3-D geodesic, coloured by year) or the t-SNE in "
            "linear_probing/.../letters__probe_cls__period/figures/tsne_best_layer.png. "
            "To fill: save as presentation/figures/qwen_geometry.png and re-run build_deck.py."
        ),
    },

    # 16 ── THESIS DISCUSSION ───────────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "15 — Contributions and discussion",
        "title": "What we contribute, and what is missing to make it a thesis",
        "body": [
            (
                "Contributions",
                "<strong>(1)</strong> The first confounder-controlled method for dating low-resource Akkadian, "
                "cast as chronological-ordering probing. "
                "<strong>(2)</strong> A cleaned royal-inscription benchmark for Akkadian chronology (to release). "
                "<strong>(3)</strong> A benchmark across TF-IDF / MLM / Qwen 1.7–120B / multilingual seq2seq, "
                "identifying the 400M translation encoder as the recommended dating system. "
                "<strong>(4)</strong> An explanation: the prior-work LLM timeline does not surface linearly "
                "for Akkadian — accounting for why scale and prompting fail."
            ),
            (
                "In progress",
                "Geodesic / manifold structure of Thalesian's PLS subspace. "
                "Error-overlap analysis across models. "
                "The –ma particle: a sixfold OB→LB decline, previously unreported "
                "(paper in prep, Wasserman &amp; Ni)."
            ),
            (
                "What is missing to wrap it into a thesis (if at all)",
                "Is the applied story — honest dating method + recommended model + the "
                "“why the small model wins” explanation — already a coherent CS thesis? "
                "If not, what is the minimum still needed: bootstrap CIs over the MC draws, "
                "one non-linear probe (MLP/kernel) to settle absent-vs-entangled, "
                "a small qualitative ordering of held-out fragments, and releasing the benchmark?"
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

    if kind == "placeholder":
        uri = slot_uri(s["slot"])
        note = f'<div class="fig-note">{s["note"]}</div>' if s.get("note") else ""
        if uri:
            fig = f'<div class="fig-wrap"><img src="{uri}" alt="{s["title"]}"></div>'
        else:
            fig = (
                '<div class="fig-wrap"><div class="placeholder-box">'
                '<div class="ph-icon">&#128206;</div>'
                f'<div class="ph-hint">{s["drop_hint"]}</div>'
                f'<div class="ph-slot">presentation/figures/{s["slot"]}.png</div>'
                '<div class="ph-sub">drop the file &amp; re-run build_deck.py — it embeds automatically</div>'
                '</div></div>'
            )
        return f"""
<section class="slide slide-figure" data-index="{idx}">
  {ew_html}
  <h2 class="sh">{s["title"]}</h2>
  {fig}
  <div class="takeaway"><span class="tk-label">Key takeaway</span>{s["takeaway"]}</div>
  {note}
</section>"""

    return f'<section class="slide" data-index="{idx}"><p>Unknown kind: {kind}</p></section>'


def build():
    slides = "\n".join(slide_html(i, s) for i, s in enumerate(SLIDES))
    total = len(SLIDES)
    titles = json.dumps([s.get("title", f"Slide {i+1}") for i, s in enumerate(SLIDES)])
    html = TEMPLATE.replace("__SLIDES__", slides).replace("__TOTAL__", str(total)).replace("__TITLES__", titles)
    out = os.path.join(HERE, "thesis_story.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    filled = [s["slot"] for s in SLIDES if s["kind"] == "placeholder" and slot_uri(s["slot"])]
    empty = [s["slot"] for s in SLIDES if s["kind"] == "placeholder" and not slot_uri(s["slot"])]
    print(f"Wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, {total} slides)")
    if filled:
        print("  filled placeholders:", ", ".join(filled))
    if empty:
        print("  EMPTY placeholders (drop a PNG into presentation/figures/):", ", ".join(empty))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Akkadian Dating — Thesis Story</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#ddd8cf; --white:#ffffff; --ink:#1c2028; --ink-mid:#3d4552; --ink-light:#6b7484;
  --green:#1a5c3a; --green-bg:#eaf4ef; --green-mid:#2d7a52; --red:#8b1a10;
  --border:#dde1e9; --border-light:#eef0f5;
  --shadow:0 4px 28px rgba(0,0,0,.14),0 1px 4px rgba(0,0,0,.07);
  --serif:"Iowan Old Style","Palatino Linotype",Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
}
html,body{height:100%;overflow:hidden;background:var(--bg);font-family:var(--sans);}
#progress{position:fixed;top:0;left:0;height:3px;background:var(--green);z-index:100;transition:width .3s;}
#topbar{position:fixed;top:8px;left:24px;right:24px;display:flex;justify-content:space-between;
        align-items:center;z-index:90;pointer-events:none;}
#slide-label{font-size:11px;letter-spacing:.07em;color:rgba(60,60,60,.6);max-width:55%;}
#counter-top{font-size:11px;color:rgba(60,60,60,.6);font-variant-numeric:tabular-nums;}
.stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;padding:32px 28px 58px;}
.slide{display:none;flex-direction:column;width:min(1200px,96vw);height:min(700px,88vh);
       background:var(--white);border-radius:12px;box-shadow:var(--shadow);
       padding:46px 58px 46px;position:relative;overflow:hidden;}
.slide::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;
               background:linear-gradient(90deg,var(--green) 0%,#2ea86b 100%);}
.slide.active{display:flex;animation:appear .28s ease;}
@keyframes appear{from{opacity:0;transform:translateY(5px);}to{opacity:1;transform:none;}}
.slide::after{content:attr(data-num);position:absolute;bottom:14px;right:20px;
              font-size:10.5px;color:var(--border);font-variant-numeric:tabular-nums;}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:.22em;text-transform:uppercase;
         color:var(--green);margin-bottom:10px;}
.sh{font-family:var(--serif);font-size:28px;line-height:1.15;color:var(--ink);margin-bottom:16px;max-width:94%;}
.slide-title{justify-content:center;}
.title-kicker{font-size:11.5px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;
              color:var(--green);margin-bottom:22px;}
.title-h1{font-family:var(--serif);font-size:46px;line-height:1.08;color:var(--ink);
          letter-spacing:-.3px;margin-bottom:18px;}
.title-sub{font-family:var(--serif);font-size:21px;color:var(--ink-mid);font-style:italic;
           margin-bottom:30px;line-height:1.35;}
.title-meta{font-size:14.5px;line-height:1.9;color:var(--ink-light);}
.text-points{display:flex;flex-direction:column;gap:15px;flex:1;min-height:0;overflow:auto;padding-right:4px;}
.tp{border-left:3px solid var(--border-light);padding-left:18px;}
.tp:hover{border-left-color:var(--green-mid);}
.tp-h{font-size:16.5px;font-weight:700;color:var(--green);margin-bottom:4px;}
.tp-b{font-size:16.5px;line-height:1.55;color:var(--ink-mid);}
.tp-b strong,.tp-b em{color:var(--ink);}
.fig-wrap{flex:1;display:flex;align-items:center;justify-content:center;min-height:0;margin:2px 0 12px;}
.fig-wrap img{max-width:100%;max-height:100%;object-fit:contain;border-radius:5px;border:1px solid var(--border-light);}
.placeholder-box{width:100%;height:100%;border:2.5px dashed #c3a94e;border-radius:10px;
                 background:repeating-linear-gradient(45deg,#fdfaf0,#fdfaf0 14px,#faf4e2 14px,#faf4e2 28px);
                 display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;}
.ph-icon{font-size:40px;opacity:.55;}
.ph-hint{font-size:19px;font-weight:700;color:#7c5e00;font-family:var(--serif);}
.ph-slot{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px;color:#5a4600;
         background:#fff;border:1px solid #e3d7a8;border-radius:5px;padding:4px 10px;}
.ph-sub{font-size:13px;color:#8a7740;}
.takeaway{background:var(--green-bg);border-left:4px solid var(--green);border-radius:7px;
          padding:12px 18px;font-size:16.5px;line-height:1.5;color:var(--ink);}
.tk-label{display:inline-block;font-size:10px;letter-spacing:.18em;text-transform:uppercase;
          font-weight:800;color:var(--green);margin-right:10px;vertical-align:2px;}
.fig-note{margin-top:8px;font-size:13px;color:var(--ink-light);line-height:1.5;padding-left:22px;}
.method-lead{font-size:15.5px;line-height:1.55;color:var(--ink-mid);margin-bottom:16px;}
.method-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:11px;flex:1;min-height:0;}
.mc{background:#f8f9fb;border:1px solid var(--border);border-radius:9px;padding:15px 13px;
    display:flex;flex-direction:column;gap:7px;}
.mc-icon{font-size:20px;font-weight:800;color:var(--green);}
.mc-name{font-family:var(--serif);font-size:14.5px;font-weight:700;color:var(--ink);}
.mc-body{font-size:12px;line-height:1.45;color:var(--ink-light);}
.mc-body strong{color:var(--red);}
#chrome{position:fixed;bottom:12px;left:0;right:0;display:flex;align-items:center;justify-content:center;gap:14px;z-index:90;}
.btn{background:rgba(255,255,255,.75);border:1px solid rgba(0,0,0,.18);width:36px;height:36px;border-radius:50%;
     font-size:17px;cursor:pointer;display:flex;align-items:center;justify-content:center;color:#333;
     box-shadow:0 1px 4px rgba(0,0,0,.1);transition:.13s;}
.btn:hover{background:var(--white);box-shadow:0 2px 8px rgba(0,0,0,.18);}
#hint{position:fixed;bottom:12px;right:16px;font-size:11px;color:rgba(0,0,0,.32);z-index:90;}
@media print{
  body{overflow:visible;background:#fff;}
  .stage{position:static;padding:0;}
  .slide{display:flex!important;page-break-after:always;box-shadow:none;width:100%;height:100vh;border-radius:0;}
  #chrome,#progress,#topbar,#hint,.slide::after{display:none;}
}
</style>
</head>
<body>
<div id="progress"></div>
<div id="topbar"><div id="slide-label"></div><div id="counter-top"></div></div>
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
