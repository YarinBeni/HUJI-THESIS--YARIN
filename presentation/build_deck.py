#!/usr/bin/env python3
"""
Build a self-contained HTML slide deck for the advisor meeting.
Run:  python presentation/build_deck.py

Two user-supplied figures are auto-embedded from presentation/figures/:
    presentation/figures/year_pls_vs_ridge_maximal.png
    presentation/figures/qwen_tsne_diachronic_manifold.png

If a file is absent a labelled placeholder box is shown instead.
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

    # 2 ── POINTS 1–4: HOOK · REFRAME · SURPRISE · CLAIM ───────────────────────
    {
        "kind": "text",
        "eyebrow": "1–4 — Hook · Reframe · Surprise · Claim",
        "title": "The thesis: a 400M translation model beats the 120B LLM — and we explain why",
        "body": [
            (
                "1 — The scholarly problem",
                "Assyriologists date cuneiform texts by hand, leaning on ruler names, script "
                "style, and archival context — a subjective, expertise-bound, time-consuming "
                "process. For most fragments the date stays uncertain."
            ),
            (
                "2 — We identify that honest dating is a chronological-ordering probing task",
                "Year regression on a small test set inflates numbers by learning format "
                "shortcuts, not language change. We reframe: probe frozen embeddings with a "
                "deliberately weak linear model and measure <strong>Spearman rank "
                "correlation</strong> — a probe that tests whether the signal is present and "
                "decodable, not one that can overfit a tiny test set."
            ),
            (
                "3 — The surprising finding",
                "<strong>A 400M multilingual translation encoder beats the 120B LLM.</strong> "
                "Scale is flat: 1.7B → 8B → 32B → 120B all cluster near each other. "
                "Zero-shot, few-shot, and chain-of-thought prompting add nothing. "
                "The internal linear timeline reported in prior LLM work does not surface for "
                "Akkadian under linear probing."
            ),
            (
                "4 — The claims (skeleton)",
                "<strong>Mean pooling &gt; last-token</strong> across all models. "
                "<strong>400M &gt; 120B.</strong> "
                "Scale is not the lever. "
                "Next-token finetuning on Akkadian is flat. "
                "<strong>The translation objective is what builds the diachronic signal</strong> "
                "— not architecture, not tokenizer, not data volume alone."
            ),
        ],
    },

    # 3 ── THE EVALUATION PIPELINE (method cards) ────────────────────────────────
    {
        "kind": "method",
        "eyebrow": "The evaluation pipeline",
        "title": "Maximal · mean-pool · balanced · PLS · Spearman",
        "lead": (
            "The five-stage pipeline applied identically to every embedding source. "
            "Each step closes one dishonest shortcut. "
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

    # 4 ── POINTS 5–8: MEASUREMENT · NEGATIVES · GENRE · STRUCTURE ──────────────
    {
        "kind": "text",
        "eyebrow": "5–8 — Measurement · Negatives · Genre · Structure",
        "title": "How we earned the right to those claims",
        "body": [
            (
                "5 — Honest measurement machinery",
                "Confounder elimination (maximal cleaning, genre restriction) + 200 balanced "
                "Monte Carlo draws (GroupKFold by ruler) + Spearman rank correlation = numbers "
                "that measure <em>dating</em>, not format recognition. "
                "This is the same obsession with not overclaiming that controls for a "
                "convenient proxy by building a controlled setup — each step earns the next claim."
            ),
            (
                "6 — Negative results as findings — with a hypothesized why",
                "Next-token finetuning on Akkadian adds nothing, <em>possibly because</em> NTP "
                "rewards local next-sign prediction and does not force alignment of surface "
                "orthography with meaning — which the translation objective does. "
                "Scale adds nothing, <em>possibly because</em> the LLM's internal chronology is "
                "entangled with genre and provenance, not laid out on a clean linear axis a probe can reach."
            ),
            (
                "7 — Genre as the analytical lens",
                "Royal inscriptions are a homogeneous genre — fixed register, single-voice "
                "authority, direct ruler–date link. Restricting to one genre closes the genre "
                "confounder. The fact that TF-IDF <em>flips from winner to loser</em> under "
                "maximal cleaning is direct evidence the confounder was real — and that our "
                "cleaning works."
            ),
            (
                "8 — Standard paper structure",
                "This is a methods paper with a benchmark, a results table, and an explanation. "
                "Each section earns the next: the protocol earns the results, the results earn "
                "the autopsy, the autopsy earns the explanation. "
                "Contributions: <strong>(1)</strong> the honest method, "
                "<strong>(2)</strong> the benchmark, "
                "<strong>(3)</strong> the model ranking, "
                "<strong>(4)</strong> the why."
            ),
        ],
    },

    # 5 ── PLS vs RIDGE — HEADLINE RESULT ───────────────────────────────────────
    {
        "kind": "placeholder",
        "slot": "year_pls_vs_ridge_maximal",
        "eyebrow": "The headline result",
        "title": "PLS vs Ridge across all models — the 400M translation encoder wins",
        "drop_hint": "year_pls_vs_ridge_maximal.png",
        "takeaway": (
            "Under the honest protocol, the 400M multilingual translation encoder achieves the "
            "highest balanced year-Spearman of every source — above all Qwen scales, the 120B model, "
            "the from-scratch MLM, and TF-IDF. PLS and Ridge agree on the ranking, "
            "so the win is not an artifact of dimensionality reduction. "
            "This is the recommended system for automatic Akkadian dating."
        ),
        "note": None,
    },

    # 6 ── BEST LAYER PER MODEL ─────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "Where in each model does the year signal live?",
        "title": "Thalesian's signal deepens with layer; LLMs peak mid-network then decay",
        "fig": "v_1/src/geodesic/maximal_figs/figures/fig4_maximal_A.png",
        "takeaway": (
            "Year-PLS Spearman across all layers (balanced · maximal · mean). "
            "Thalesian-400M rises steadily to its best at layer 10 (★), "
            "climbing well above TF-IDF. "
            "Qwen models peak around layers 15–16 then fall back toward the baseline. "
            "The translation model concentrates its diachronic signal in late layers; "
            "LLMs peak early-to-middle and lose it."
        ),
        "note": (
            "Random-8B stays flat near 0.30 at every layer — the random floor, "
            "what the architecture gives for free before any training. Trained LLMs barely clear it."
        ),
    },

    # 7 ── K SWEEP ──────────────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "How many dimensions does the date signal need?",
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

    # 8 ── NTP + SCALE NULL ─────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "Does scale or Akkadian finetuning help?",
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

    # 9 ── TOKENIZER RULED OUT ──────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "Autopsy: is it the tokenizer?",
        "title": "Thalesian wins despite the least efficient Akkadian tokenizer",
        "fig": "v_1/src/chronorank/autopsy/results/figures/fertility_by_corpus.png",
        "takeaway": (
            "Thalesian splits Akkadian words into more subword tokens than every other model "
            "on every corpus — it is the <em>least</em> efficient tokenizer. "
            "GPT-OSS is most efficient (4.43 tok/word) and ranks fourth; "
            "Thalesian (6.22 tok/word) ranks first. "
            "If tokenizer quality drove performance it should lose; it wins. "
            "The tokenizer is ruled out as the cause."
        ),
        "note": None,
    },

    # 10 ── TRANSLATION FINETUNE BUILDS THE SIGNAL ──────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "Autopsy: it is the translation objective",
        "title": "The translation finetune alone builds a deep diachronic representation",
        "fig": "v_1/src/chronorank/autopsy/results/figures/factor_ladder_layerwise.png",
        "takeaway": (
            "Same architecture, same tokenizer, same 0.4B size — only the objective differs. "
            "Vanilla uMT5-base (red) starts at the random floor and <em>decays below it</em> with depth. "
            "The cuneiform translation finetune (Thalesian) builds a representation "
            "that rises to 0.41 at layer 10. "
            "Training a model to map Akkadian text to its meaning is what instils temporal structure — "
            "next-token prediction cannot."
        ),
        "note": (
            "Thalesian 0.411 vs. vanilla uMT5 0.297 → Δ = +0.114, entirely from the finetune. "
            "This is the mechanism behind the headline result."
        ),
    },

    # 11 ── QWEN GEOMETRY ───────────────────────────────────────────────────────
    {
        "kind": "placeholder",
        "slot": "qwen_tsne_diachronic_manifold",
        "eyebrow": "Why the LLM cannot: the geometry of its embedding",
        "title": "Chronology is entangled in Qwen's embedding space, not laid out on a clean axis",
        "drop_hint": "qwen_tsne_diachronic_manifold.png",
        "takeaway": (
            "Prior work reports a linearly-recoverable internal timeline in LLMs. "
            "For Akkadian it does not surface: the chronological signal is tangled together "
            "with genre and location, plausibly on a non-linear manifold a linear probe cannot reach. "
            "This is why scaling and prompting the LLM do not help — "
            "and why a translation encoder, which packages chronology in a directly decodable form, wins instead."
        ),
        "note": None,
    },

    # 12 ── CONTRIBUTIONS & DISCUSSION ──────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "Contributions and discussion",
        "title": "What we contribute, and what is missing to wrap it into a thesis",
        "body": [
            (
                "Contributions",
                "<strong>(1)</strong> The first confounder-controlled method for dating "
                "low-resource Akkadian, cast as chronological-ordering probing. "
                "<strong>(2)</strong> A cleaned royal-inscription benchmark for Akkadian "
                "chronology (to release). "
                "<strong>(3)</strong> A benchmark across TF-IDF / MLM / Qwen 1.7–120B / "
                "multilingual seq2seq, identifying the 400M translation encoder as the "
                "recommended dating system. "
                "<strong>(4)</strong> An explanation: the prior-work LLM timeline does not "
                "surface linearly for Akkadian — accounting for why scale and prompting fail, "
                "and implicating the translation objective as what instils temporal structure."
            ),
            (
                "What is missing to wrap it into a thesis — if at all?",
                "Is the applied story — honest dating method + recommended model + "
                "&ldquo;why the small model wins&rdquo; explanation — already a coherent CS thesis? "
                "If not, what is the minimum still needed: bootstrap confidence intervals "
                "over the MC draws, one non-linear probe (MLP/kernel) to settle "
                "absent-vs-entangled, a small qualitative ordering of held-out fragments, "
                "or releasing the benchmark?"
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
