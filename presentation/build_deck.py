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


# ─── SLIDE DEFINITIONS ────────────────────────────────────────────────────────
# kind: "title" | "text" | "figure" | "method"
# takeaway: one bold sentence – the thing to remember from this slide
# note: smaller context line (shown lighter, for reference)
SLIDES = [

    # ── 1. TITLE ───────────────────────────────────────────────────────────────
    {
        "kind": "title",
        "title": "Dating Ancient Akkadian with Language Models",
        "subtitle": "How we built an honest evaluation — and what it revealed",
        "meta": (
            "M.Sc. Thesis &middot; Yarin Beer &middot; Computer Science, HUJI<br>"
            "Advisors: Prof. Nathan Wasserman · Dr. Barak Sober · Prof. Gabriel Stanovsky"
        ),
    },

    # ── 2. THE QUESTION ────────────────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "1 — The question",
        "title": "Can a model date Akkadian — or is it just pattern-matching artifacts?",
        "body": [
            (
                "The goal",
                "3,000 years of Akkadian literature is largely undated. Dating is partly a "
                "distributional problem — features that change in frequency over time. "
                "Language models should be able to help. But <em>should</em> is the question."
            ),
            (
                "The fundamental trap",
                "A model can date texts by exploiting <strong>corpus artifacts</strong> — "
                "royal names, letter-opening formulae, archive provenance, sign conventions — "
                "without learning anything about linguistic change. We needed a way to "
                "tell the difference."
            ),
            (
                "Our approach",
                "We use LLMs as <strong>frozen feature extractors</strong> and probe their "
                "hidden-state representations with a minimal linear model. A linear probe "
                "cannot learn the task — it can only reveal what is already encoded. "
                "Every experiment below is a step toward a probe we could actually trust."
            ),
        ],
    },

    # ── 3. THE MODELS ──────────────────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "2 — The setup",
        "title": "Six model families — three questions answered",
        "body": [
            (
                "The scale question (do bigger LLMs know more Akkadian?)",
                "Qwen3&thinsp;1.7B / 8B / 32B &nbsp;·&nbsp; GPT-OSS 120B &nbsp;·&nbsp; "
                "Random-weight Qwen (architecture-only control)"
            ),
            (
                "The Akkadian-specialist question (does domain training help?)",
                "MLM trained from scratch on our 10M-word corpus &nbsp;·&nbsp; "
                "TF-IDF over lemmata n-grams (the vocabulary-counting baseline)"
            ),
            (
                "The transfer-learning question (does multilingual fine-tuning matter?)",
                "Thalesian cuneiformBase-400M &mdash; a uMT5-base fine-tuned to <em>translate</em> "
                "Akkadian, Sumerian, Hittite, Linear&thinsp;B, Elamite into English/German &nbsp;·&nbsp; "
                "Thalesian akk300M (Akkadian-only fine-tune) &nbsp;·&nbsp; "
                "Vanilla uMT5-base (no fine-tune &mdash; our strongest control)"
            ),
        ],
    },

    # ── 4. FIRST RESULT — letters 99% — red flag ───────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "3 — First alarm: the result is too good",
        "title": "99% period accuracy — but n-gram counting is already at 96.7%",
        "fig": "v_1/src/linear_probing/results/letters__probe_cls__period/figures/layer_accuracy_curve.png",
        "takeaway": (
            "On the 4,957-letter corpus, Qwen scores 99% at tier-0 — "
            "but the 2–5-gram baseline (dotted red line) already reaches 96.7%. "
            "The LLM is not using learned representations; "
            "it is counting vocabulary, and the vocabulary gives away the period."
        ),
        "note": (
            "Maximal cleaning (green) drops the curve to 87–93%, which is where honest "
            "evaluation starts. Tier-0 = raw text; maximal = 11 cleaning filters "
            "(names, logograms, determinatives, formulae, digits removed)."
        ),
    },

    # ── 5. RANDOM WEIGHTS ──────────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "4 — Proof the easy result is fake",
        "title": "Random-weight Qwen scores 97% on raw text — pretraining is nearly irrelevant",
        "fig": "v_1/src/linear_probing/results/letters__probe_cls__period/figures/random_baseline_comparison.png",
        "takeaway": (
            "On tier-0 letters (left), pretrained Qwen and a Qwen with fully randomized weights "
            "are almost identical — the letter <em>format</em> labels the period, "
            "not anything the model learned. "
            "Only after maximal cleaning (right) does a 7-point pretraining gap emerge."
        ),
        "note": (
            "This finding forced us to move to a harder dataset and stricter cleaning. "
            "A result that holds even with random weights is a result we cannot publish."
        ),
    },

    # ── 6. HARDER TASK — ORCC — cleaning flips TF-IDF ──────────────────────────
    {
        "kind": "figure",
        "eyebrow": "5 — Harder dataset: royal inscriptions with exact year dates",
        "title": "With raw text, Qwen falls below random on ruler classification",
        "fig": "v_1/src/linear_probing/results/orcc__probe_cls/figures/best_of_ruler.png",
        "takeaway": (
            "On ORCC royal inscriptions (893 texts, 38 rulers), "
            "TF-IDF dominates at 32.5% macro-F1 because it can read the king's name in the title — "
            "while trained Qwen (blue) sits below the random-weight baseline (purple). "
            "Royal names are a strong artifact; the model is confused, not dating."
        ),
        "note": (
            "After maximal cleaning, TF-IDF collapses to near-random — "
            "proving the TF-IDF advantage was entirely the ruler name in the text. "
            "This is what we want: an evaluation where the easy shortcut is gone."
        ),
    },

    # ── 7. METRIC PROBLEM — accuracy@N → Spearman ──────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "6 — The metric is also broken",
        "title": "A dummy that guesses the mean date passes most accuracy thresholds",
        "fig": "v_1/src/geodesic/fig1_followups/error_overlap/predictions_maximal_balanced/accuracy_at_N.png",
        "takeaway": (
            "76% of ORCC inscriptions belong to three Sargonid kings within a 188-year window. "
            "A model that always predicts the corpus mean (grey dashed) clears ±50 years at 77% — "
            "better than any real model. Accuracy is measuring class distribution, not dating ability."
        ),
        "note": (
            "This forced two changes: (1) switch from accuracy to Spearman rank correlation — "
            "test chronological ordering, not exact-year prediction; "
            "(2) Monte-Carlo balanced resampling to prevent the majority class from dominating."
        ),
    },

    # ── 8. CHRONOLOGY — models win at extremes ──────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "7 — Balanced Spearman: where real models earn their keep",
        "title": "Real models beat the dummy only at the chronological extremes",
        "fig": "v_1/src/geodesic/fig1_followups/error_overlap/predictions_maximal_balanced/balanced_maximal_chronology.png",
        "takeaway": (
            "Under balanced Spearman scoring, Thalesian outperforms the dummy "
            "at the early and late extremes of the timeline — "
            "exactly where dating is hardest and most valuable to philologists. "
            "TF-IDF only wins in the dense Sargonid centre where names are present."
        ),
        "note": (
            "Left panel: accuracy-at-N curves confirm all models beat the dummy by ±25 years. "
            "Right panel (MAE by true date): the real story — Thalesian's advantage is at the extremes."
        ),
    },

    # ── 9. THE PROTOCOL — METHOD SLIDE ─────────────────────────────────────────
    {
        "kind": "method",
        "eyebrow": "8 — The evaluation protocol we built",
        "title": "Maximal · mean-pool · balanced · PLS · Spearman",
        "lead": (
            "Every ingredient was forced by a specific confound we discovered. "
            "This is the methodological contribution — a reusable honest benchmark "
            "for probing diachronic signal in low-resource ancient-language models."
        ),
        "items": [
            (
                "✂", "Maximal cleaning",
                "Remove royal names, logograms, determinatives, formulae, digits. "
                "<strong>Solves:</strong> surface artifacts (ruler names, letter format) "
                "that make any model look like it's dating when it's just pattern-matching."
            ),
            (
                "μ", "Mean pooling",
                "Average all token representations into one document vector. "
                "<strong>Solves:</strong> Akkadian has many short syllabic tokens — "
                "the diachronic signal is document-level, not concentrated in one token."
            ),
            (
                "⚖", "Balanced (200 MC draws)",
                "Monte-Carlo stratified resampling + GroupKFold by ruler. "
                "<strong>Solves:</strong> 76% Sargonid imbalance and ruler-leakage "
                "across folds — prevents majority-class guessing from looking like dating."
            ),
            (
                "PLS", "Partial Least-Squares",
                "Find the k latent directions in the embedding space most aligned with the date axis. "
                "<strong>Solves:</strong> high dimensionality — different models have different "
                "embedding sizes; PLS finds the right subspace in each."
            ),
            (
                "ρ", "Spearman correlation",
                "Score chronological ordering, not exact year prediction. "
                "<strong>Solves:</strong> 188-year window, ordinal task, "
                "and allows fair comparison across very different model types."
            ),
        ],
    },

    # ── 10. THE HEADLINE RESULT ─────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "9 — The headline result",
        "title": "A 400M translation model beats every LLM up to 120B",
        "fig": "v_1/src/chronorank/autopsy/results/figures/factor_ladder_bars.png",
        "takeaway": (
            "Under our honest protocol, Thalesian-400M (Spearman 0.41) outperforms "
            "Qwen3-8B (0.36), Qwen3-32B (0.34), GPT-OSS-120B (0.33), "
            "and our homegrown Akkadian MLM (0.31). "
            "Most strikingly: random-weight Qwen (0.30) "
            "ties the trained giants — scale and pretraining are not the answer."
        ),
        "note": (
            "Vanilla uMT5-base (the un-fine-tuned base of Thalesian, red bar far right) = 0.297, "
            "at the random floor. The entire Thalesian advantage (Δ+0.114) comes from the fine-tune — "
            "this is what we investigate next."
        ),
    },

    # ── 11. NTP FINETUNE NULL ───────────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "10 — Rule out 1: more Akkadian exposure",
        "title": "Fine-tuning Qwen on Akkadian with next-token prediction does nothing",
        "fig": "v_1/src/finetune/results/figures/maximal_pls_bestlayer.png",
        "takeaway": (
            "Each '+NTP (ft)' bar lands exactly on its base model — "
            "fine-tuning Qwen on all our Akkadian text with standard next-token prediction "
            "produces zero improvement at any scale from 1.7B to 120B. "
            "The objective matters, not the exposure."
        ),
        "note": (
            "We tried checkpoints at multiple stages (ft00 through ft32). "
            "Layer-by-layer curves (not shown) confirm the overlay is exact: "
            "NTP shifts no layer's Spearman by more than noise."
        ),
    },

    # ── 12. LAYER-BY-LAYER NTP CONFIRMATION ────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "10b — The NTP null, layer by layer",
        "title": "Every fine-tune checkpoint overlays the base model at every layer",
        "fig": "v_1/src/finetune/results/figures/ftcurves_qwen3_8b_maximal.png",
        "takeaway": (
            "Across all 37 layers of Qwen3-8B, "
            "every NTP checkpoint (ft00–ft32, coloured lines) "
            "traces the exact same curve as the untouched base model (black). "
            "Next-token finetuning leaves the diachronic representation entirely unchanged."
        ),
        "note": (
            "This is the complement of the Thalesian result: "
            "Thalesian's fine-tune DID change the representation (rises to 0.41 at layer 10). "
            "The difference between them is the fine-tuning objective — NTP vs translation."
        ),
    },

    # ── 13. TOKENIZER NOT THE CAUSE ────────────────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "11 — Rule out 2: the tokenizer",
        "title": "Thalesian wins despite the worst Akkadian tokenizer in the group",
        "fig": "v_1/src/chronorank/autopsy/results/figures/fertility_by_corpus.png",
        "takeaway": (
            "Thalesian (blue) fragments Akkadian words into more subword tokens "
            "than every other model on every corpus — it is the least efficient tokenizer. "
            "If tokenizer quality drove performance, Thalesian should lose. "
            "It wins, so the tokenizer is ruled out."
        ),
        "note": (
            "Lower fertility (fewer tokens per word) is assumed to be better "
            "because it preserves morphological units. "
            "GPT-OSS (4.43 tok/word) is the most efficient; Thalesian (6.22) the least. "
            "Performance ranking is the reverse of efficiency ranking."
        ),
    },

    # ── 14. THE ANSWER — TRANSLATION FINETUNE ──────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "12 — What does carry the win",
        "title": "The translation fine-tune alone builds a deep diachronic representation",
        "fig": "v_1/src/chronorank/autopsy/results/figures/factor_ladder_layerwise.png",
        "takeaway": (
            "Vanilla uMT5-base (red) starts at the random floor and decays below it "
            "as depth increases — its deeper layers actively hurt. "
            "The cuneiform <em>translation</em> fine-tune (green, Thalesian) "
            "builds a representation that rises to 0.41 at layer 10. "
            "Forcing the model to map text to meaning instills a diachronic signal "
            "that next-token prediction cannot."
        ),
        "note": (
            "Qwen3-1.7B (blue, size-matched to 0.4B uMT5) peaks around 0.35 then declines. "
            "uMT5 base at layer 0 = 0.297; Thalesian at best layer = 0.411. Δ finetune = +0.114."
        ),
    },

    # ── 15. OPEN QUESTION — MANIFOLD STRUCTURE ─────────────────────────────────
    {
        "kind": "figure",
        "eyebrow": "13 — What we still don't fully understand",
        "title": "The embeddings have structure — but it's entangled",
        "fig": "v_1/src/linear_probing/results/letters__probe_cls__period/figures/tsne_best_layer.png",
        "takeaway": (
            "t-SNE of the best layer shows the embeddings do organize by period — "
            "but the signal is entangled with genre, provenance, and text length. "
            "A linear probe may be the wrong tool: "
            "the diachronic axis could live on a non-linear manifold "
            "that PLS and Spearman never fully reach."
        ),
        "note": (
            "Left: tier-0 (raw text) — period clusters are sharp. "
            "Right: maximal cleaning — clusters blur. "
            "The next step is to inspect the PLS component vectors of Thalesian directly "
            "to understand what geometric structure the translation fine-tune creates."
        ),
    },

    # ── 16. SUMMARY & THESIS ───────────────────────────────────────────────────
    {
        "kind": "text",
        "eyebrow": "14 — Wrapping into a thesis",
        "title": "What we know, what we built, and what's open",
        "body": [
            (
                "The methodological contribution (ready to write up)",
                "A reusable, honest evaluation protocol for probing diachronic signal: "
                "maximal cleaning · mean pooling · balanced Monte-Carlo resampling · "
                "PLS probing · Spearman correlation. "
                "Each ingredient is justified by a specific confound we discovered and eliminated."
            ),
            (
                "The empirical finding (ready to write up)",
                "Large LLMs do not encode a useful Akkadian diachronic signal — "
                "scale and NTP fine-tuning are null. "
                "A 400M translation-fine-tuned model wins, "
                "and the win comes from the translation objective, "
                "not the tokenizer or architecture."
            ),
            (
                "In progress",
                "Geodesic / manifold structure of Thalesian's PLS embedding. "
                "Error-overlap analysis: where models agree and disagree. "
                "The -ma particle: a previously unreported diachronic marker "
                "(rate drops sixfold from OB to LB). [Paper in prep with Wasserman & Ni.]"
            ),
            (
                "Discussion for today",
                "Which of the above is the central chapter of the CS thesis? "
                "What is the minimum still needed to converge? "
                "Is the methodological protocol alone, "
                "paired with the translation-objective finding, enough for a publication?"
            ),
        ],
    },
]


# ─── HTML RENDERING ───────────────────────────────────────────────────────────

def slide_html(idx, s):
    kind = s["kind"]
    ew = s.get("eyebrow", "")
    ew_html = f'<div class="eyebrow">{ew}</div>' if ew else ""
    n = idx + 1

    if kind == "title":
        return f"""
<section class="slide slide-title" data-index="{idx}">
  <div class="title-inner">
    <div class="title-kicker">M.Sc. Thesis Advisor Meeting &mdash; June 2026</div>
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
/* ── RESET & VARIABLES ── */
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#ddd8cf;
  --slide-bg:#ffffff;
  --ink:#1c2028;
  --ink-mid:#3d4552;
  --ink-light:#6b7484;
  --green:#1a5c3a;
  --green-light:#e8f4ee;
  --green-mid:#2d7a52;
  --red:#8b1a10;
  --gold:#7c5e00;
  --border:#dde1e9;
  --border-light:#eef0f4;
  --shadow:0 4px 24px rgba(0,0,0,.13), 0 1px 4px rgba(0,0,0,.08);
  --serif:"Iowan Old Style","Palatino Linotype",Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
}
html,body{height:100%;overflow:hidden;background:var(--bg);font-family:var(--sans);}

/* ── PROGRESS ── */
#progress{position:fixed;top:0;left:0;height:3px;background:var(--green);z-index:100;transition:width .3s;}

/* ── TOP BAR ── */
#topbar{position:fixed;top:6px;left:0;right:0;display:flex;align-items:center;
        justify-content:space-between;padding:0 22px;z-index:90;pointer-events:none;}
#slide-title-label{font-size:11.5px;letter-spacing:.06em;color:rgba(80,80,80,.75);
                   font-family:var(--sans);max-width:60%;}

/* ── STAGE ── */
.stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
       padding:32px 28px 60px;}

/* ── SLIDE BASE ── */
.slide{display:none;flex-direction:column;
       width:min(1200px,96vw);height:min(700px,88vh);
       background:var(--slide-bg);border-radius:12px;
       box-shadow:var(--shadow);padding:48px 60px 52px;
       position:relative;overflow:hidden;}
.slide::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;
               background:linear-gradient(90deg,var(--green),#2ea86b);}
.slide.active{display:flex;animation:appear .3s ease;}
@keyframes appear{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;}}

/* ── EYEBROW ── */
.eyebrow{font-size:11.5px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;
         color:var(--green);margin-bottom:11px;}

/* ── SLIDE HEADING ── */
.sh{font-family:var(--serif);font-size:30px;line-height:1.15;color:var(--ink);
    margin-bottom:18px;max-width:92%;}

/* ── TITLE SLIDE ── */
.slide-title{justify-content:center;background:var(--slide-bg);}
.title-inner{max-width:800px;}
.title-kicker{font-size:12px;font-weight:600;letter-spacing:.18em;text-transform:uppercase;
              color:var(--green);margin-bottom:20px;}
.title-h1{font-family:var(--serif);font-size:52px;line-height:1.07;color:var(--ink);
          letter-spacing:-.3px;margin-bottom:18px;}
.title-sub{font-family:var(--serif);font-size:23px;color:var(--ink-mid);
           font-style:italic;margin-bottom:28px;}
.title-meta{font-size:15px;line-height:1.8;color:var(--ink-light);}

/* ── TEXT SLIDES ── */
.text-points{display:flex;flex-direction:column;gap:16px;flex:1;min-height:0;overflow:auto;}
.tp{border-left:3px solid var(--border);padding-left:18px;}
.tp-h{font-family:var(--serif);font-size:18px;font-weight:600;color:var(--green);
      margin-bottom:5px;}
.tp-b{font-size:17px;line-height:1.55;color:var(--ink-mid);}
.tp-b strong,.tp-b em{color:var(--ink);}

/* ── FIGURE SLIDES ── */
.slide-figure{padding-bottom:44px;}
.fig-wrap{flex:1;display:flex;align-items:center;justify-content:center;
          min-height:0;margin:2px 0 14px;}
.fig-wrap img{max-width:100%;max-height:100%;object-fit:contain;
              border-radius:6px;border:1px solid var(--border-light);}
.takeaway{background:var(--green-light);border-left:4px solid var(--green);
          border-radius:7px;padding:13px 18px;font-size:17px;line-height:1.5;color:var(--ink);}
.tk-label{display:inline-block;font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;
          font-weight:800;color:var(--green);margin-right:10px;vertical-align:1px;}
.fig-note{margin-top:9px;font-size:13.5px;color:var(--ink-light);line-height:1.5;
          padding-left:22px;}

/* ── METHOD SLIDE ── */
.method-lead{font-size:16px;line-height:1.55;color:var(--ink-mid);margin-bottom:18px;}
.method-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;flex:1;min-height:0;}
.mc{background:#f9fafb;border:1px solid var(--border);border-radius:9px;
    padding:16px 14px;display:flex;flex-direction:column;gap:6px;}
.mc-icon{font-size:22px;font-weight:800;color:var(--green);}
.mc-name{font-family:var(--serif);font-size:15.5px;font-weight:700;color:var(--ink);}
.mc-body{font-size:12.5px;line-height:1.45;color:var(--ink-light);}
.mc-body strong{color:var(--red);}

/* ── CHROME ── */
#chrome{position:fixed;bottom:14px;left:0;right:0;display:flex;align-items:center;
        justify-content:center;gap:16px;z-index:90;}
.btn{background:rgba(255,255,255,.7);border:1px solid rgba(0,0,0,.18);
     width:38px;height:38px;border-radius:50%;font-size:18px;cursor:pointer;
     display:flex;align-items:center;justify-content:center;color:#333;
     box-shadow:0 1px 4px rgba(0,0,0,.12);transition:.15s;}
.btn:hover{background:#fff;box-shadow:0 2px 8px rgba(0,0,0,.18);}
#counter{font-size:13px;color:#555;font-variant-numeric:tabular-nums;
         min-width:110px;text-align:center;background:rgba(255,255,255,.65);
         border-radius:20px;padding:4px 14px;}
#hint{position:fixed;bottom:14px;right:18px;font-size:11.5px;color:rgba(0,0,0,.35);z-index:90;}

/* ── SLIDE NUMBER ── */
.slide::after{content:attr(data-index);position:absolute;bottom:16px;right:22px;
              font-size:11px;color:var(--border);font-variant-numeric:tabular-nums;}

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
  <div id="slide-title-label"></div>
  <div id="counter"></div>
</div>
<div class="stage">
__SLIDES__
</div>
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

function go(n){
  cur = Math.max(0, Math.min(TOTAL - 1, n));
  slides.forEach((s,i) => s.classList.toggle('active', i === cur));
  document.getElementById('counter').textContent = (cur + 1) + ' / ' + TOTAL;
  document.getElementById('progress').style.width = ((cur + 1) / TOTAL * 100) + '%';
  document.getElementById('slide-title-label').textContent = TITLES[cur] || '';
  history.replaceState(null,'','#'+(cur+1));
}

document.getElementById('next').onclick = () => go(cur + 1);
document.getElementById('prev').onclick = () => go(cur - 1);
document.addEventListener('keydown', e => {
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
go(isNaN(h) ? 0 : h - 1);
</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
