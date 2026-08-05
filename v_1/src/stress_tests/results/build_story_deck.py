"""Rebuild `thesis_story_9.html` in the ladder order defined by STORY_SPINE.md.

The deck's slide bodies (including the base64 figures) are reused verbatim from the
current HTML; this script only reorders them, drops the superseded ones, and inserts
the newly authored slides. Keeping it as a script rather than hand-editing 1.2 MB of
HTML means the deck can be regenerated whenever a new experiment lands — in
particular slide 8, which is a placeholder until the WB (cell-B entity) jobs finish.

    python build_story_deck.py                 # rewrite thesis_story_9.html in place
    python build_story_deck.py --out new.html  # write elsewhere instead
    python build_story_deck.py --check         # verify counters only

SPINE entries are either `("old", index)` — reuse that slide from the current deck —
or `("new", key)` — emit NEW_SLIDES[key]. TOTAL/TITLES/data-index are rewritten from
the result, which is the invariant the audit doc warns about.
"""
import argparse
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "thesis_story_9.html")

# The reusable slide bodies come from the *pre-reorder* 33-slide deck, not from the
# current file — otherwise a second run would try to reuse indices the first run
# already removed. Git blob SHAs are immutable and survive rebases, so pinning the
# blob keeps this script re-runnable forever (e.g. to refresh a slide when new
# results land) without committing a second 4 MB copy of the deck.
SOURCE_BLOB = "538265f1af40652a0cee88d9862766d62576b2d0"   # thesis_story_9.html @ 4177af08

# ---------------------------------------------------------------- the spine ----
# (kind, ref, title-for-the-nav-strip)
SPINE = [
    ("new", "title", "When Are Space and Time Linearly Represented in Language Models?"),
    ("new", "motivation", "Why we care: a world model you could point at an excavation"),
    ("new", "paper", "Gurnee & Tegmark: a strong claim, tested in one cell"),
    ("new", "matrix", "The climbing map: entity salience x language resource"),
    ("new", "protocol", "How every experiment in this deck is set up and read"),
    ("new", "cellA_repro", "A: the paper reproduces on our models, with the controls it never ran"),
    ("new", "cellA_layers", "A: where in the network space and time live"),
    ("new", "cellA_pls", "A: how many PLS directions the world model needs"),
    ("new", "b_entity", "Obscure entities in English: the date survives, the place does not"),
    ("new", "b_frag_year", "Whole fragments in English: a bag of character n-grams dates them best"),
    ("new", "b_frag_geo", "Whole fragments in English: place is no better than an untrained network"),
    ("new", "explorer_eng", "Inside the best English-side embedding: the year gradient and its confounds"),
    ("new", "mlm_model", "Our own Akkadian model: a small masked language model trained on the corpus"),
    ("new", "c_kingtoken", "Raw Akkadian: the king's name is readable, the chronology is not"),
    ("new", "ruler_not_chrono", "The name token identifies the king, and an untrained network does it too"),
    ("new", "c_frag_year", "Raw Akkadian, whole fragments: every model falls to its untrained twin"),
    ("new", "c_frag_geo", "Raw Akkadian, whole fragments: the find-spot survives where the date does not"),
    ("new", "c_layers", "Raw Akkadian, layer by layer: the translation encoder is the only arm that builds depth"),
    ("new", "c_plsk", "Raw Akkadian: the signal saturates far earlier than it does in English"),
    ("new", "explorer_akk", "Inside the winner's Akkadian embedding: the same gradient, from the raw language"),
    ("new", "t9_knowledge", "The models do know these kings and their dates when simply asked"),
    ("new", "ask_directly", "Asking the model directly, and prompting it hard, changes nothing"),
    ("new", "ntp_finetune", "Training the LLMs further on our own Akkadian moves nothing"),
    ("new", "shuffle", "Scrambling the word order costs almost nothing, so the probe reads a bag of words"),
    ("old", 20, "Curved and kernel probes do no better than the straight line"),
    ("old", 21, "Turning the supervision dial: the year is not lying in the cloud's shape"),
    ("new", "conditions", "What a linear temporal world model needs in order to exist"),
    ("new", "winner", "What does work: the 400M translation encoder beats every LLM"),
    ("new", "translation_line", "Why it works: translation finetuning, and multilingual translation most of all"),
    ("new", "tokenizer", "It is not the tokenizer: the winner has the worst one"),
    ("new", "contributions", "The boundary condition: where the linear world model ends"),
]

def _std_takeaway(sec):
    """Convert a trailing fig-note into the standard takeaway box."""
    sec = sec.replace('<p class="fig-note">',
                      '<div class="takeaway tight"><span class="tk-label">Key takeaway</span>')
    i = sec.rfind('</p>\n</section>')
    if i == -1:
        i = sec.rfind('</p></section>')
        if i != -1:
            sec = sec[:i] + '</div></section>' + sec[i+len('</p></section>'):]
    else:
        sec = sec[:i] + '</div>\n</section>' + sec[i+len('</p>\n</section>'):]
    return sec


def _fix_p9(sec):
    sec = sec.replace(
        '<div class="eyebrow">Stress test &middot; geometry-aware probes (working note &sect;2)</div>',
        '<div class="eyebrow">Rescue 5 &middot; a stronger probe family</div>')
    sec = sec.replace(
        '<h2 class="sh">P9 &mdash; geodesic kernel PLS: does the manifold&rsquo;s curvature help?</h2>',
        '<h2 class="sh">Curved and kernel probes do no better than the straight line</h2>')
    sec = sec.replace('>maximal</th>', '>cleaned Akkadian</th>')
    sec = sec.replace('Akkadian maximal + English tier0 (other cleanings in the CSV)',
                      'cleaned Akkadian and the English translation')
    sec = sec.replace(' &mdash; isolates curvature vs kernel', ': isolates curvature against kernel')
    sec = sec.replace(' &mdash; isolates PLS vs kernel', ': isolates PLS against kernel')
    sec = sec.replace('{.001, .01, .1} &mdash; it picked', '{.001, .01, .1}; it picked')
    return _std_takeaway(sec)


def _fix_p8(sec):
    sec = sec.replace(
        '<div class="eyebrow">Stress test &middot; geometry-aware probes (working note &sect;4)</div>',
        '<div class="eyebrow">Rescue 6 &middot; how much supervision it takes</div>')
    sec = sec.replace(
        '<h2 class="sh">P8 &mdash; the supervision dial: how much supervision does chronology need?</h2>',
        '<h2 class="sh">Turning the supervision dial: the year is not lying in the cloud&rsquo;s shape</h2>')
    sec = sec.replace('same 200 balanced draws, train/test split by ruler &mdash; the probe',
                      'same 200 balanced draws, train/test split by ruler, so the probe')
    sec = sec.replace('Laplacian eigenmaps). In-between', 'Laplacian eigenmaps). An in-between')
    sec = sec.replace('on PCA-100 features; test fragments are projected linearly &mdash; no leakage',
                      'on PCA-100 features; test fragments are projected linearly, with no leakage')
    sec = sec.replace('NOT the KPLS family: no components &ldquo;a&rdquo; and no kernel choice here &mdash; the only',
                      'NOT the KPLS family: no components and no kernel choice here; the only')
    sec = sec.replace('Akkadian shown name-stripped (maximal); tier0/maxking in the CSV.',
                      'Akkadian shown with names stripped; the uncleaned variants are in the CSV.')
    sec = sec.replace('Akkadian maximal &mdash; &lambda; (0 = supervised &rarr; 1 = geometry)',
                      'cleaned Akkadian &middot; &lambda; (0 = supervised &rarr; 1 = geometry)')
    sec = sec.replace('English tier0 &mdash; &lambda;', 'English translation &middot; &lambda;')
    return _std_takeaway(sec)


def _sweep(sec):
    """Residual jargon and dash cleanup for reused slides (tables keep numeric
    em-dash placeholders, which read as blanks and stay)."""
    sec = sec.replace('&mdash;&mdash;', '&ndash;&ndash;')
    sec = sec.replace('0.199&mdash;', '0.199&ndash;')
    sec = sec.replace(' &mdash; ', '; ')
    sec = sec.replace('tier0, the only valid translation (eng_maximal broken; hallucinated names); Akkadian = name-stripped maximal',
                      'the English translation and the name-stripped Akkadian')
    sec = sec.replace('English = tier0 (eng_maximal broken; hallucinated names); tier0/maxking + the full',
                      'English = the translation; the uncleaned variants and the full')
    sec = sec.replace('name-stripped maximal; tier0/maxking in the CSV.', 'name-stripped Akkadian; other variants in the CSV.')
    sec = sec.replace('tier0/maxking', 'the uncleaned variants')
    sec = sec.replace('&mdash;', '&ndash;')   # any residue is a table blank
    return sec


SLIDE_TRANSFORMS = {20: lambda s_: _sweep(_fix_p9(s_)), 21: lambda s_: _sweep(_fix_p8(s_))}

# which matrix cell each slide sits in (spine position -> cell); "" = show the map
# with nothing active yet, as orientation.
CELLMAP_AT = {4: "", 5: "A", 6: "A", 7: "A", 8: "B", 9: "B", 10: "B", 11: "B",
              13: "C", 14: "C", 15: "C", 16: "C", 17: "C", 18: "C", 19: "C"}

# --------------------------------------------------------- newly authored ------
NEW_SLIDES = {

"title": """<section class="slide slide-title">
  <div class="title-inner">
    <div class="title-kicker">M.Sc. Thesis &middot; Advisor Meeting &middot; 2026</div>
    <h1 class="title-h1">When Are Space and Time Linearly Represented in Language Models?</h1>
    <div class="title-sub">Entity salience, language resource, and pooling</div>
    <div class="title-meta">M.Sc. Thesis &middot; Yarin Beer &middot; Computer Science, HUJI<br>Advisors: Prof. Gabriel Stanovsky &nbsp;&middot;&nbsp; Dr. Barak Sober</div>
  </div>
</section>""",

"motivation": """<section class="slide slide-text">
  <div class="eyebrow">Motivation</div>
  <h2 class="sh">Why we care: a world model you could point at an excavation</h2>
  <div class="text-points">
    <div class="tp"><div class="tp-h">The claim we are following up</div><div class="tp-b">Recent interpretability work reports that language models build <strong>linear internal maps of time and space</strong>. A ridge probe on frozen activations recovers when a person died, or where a city is. If that geometry is a general property of large models, it is a free measuring instrument.</div></div>
    <div class="tp"><div class="tp-h">Why an archaeologist should care</div><div class="tp-b">Dating and provenancing cuneiform fragments is done by hand, from ruler names, script style and archival context: slow, expert-bound, and for most fragments simply unresolved. A probe that reads date and find-spot straight out of a frozen model would turn a scholarly bottleneck into an inference. <strong>That is the payoff we are testing for.</strong></div></div>
    <div class="tp"><div class="tp-h">But the claim was only ever tested in its comfort zone</div><div class="tp-b">Famous entities (world capitals, celebrated figures, headline news), written in <strong>English</strong>, read out of <strong>explicitly marked entity tokens</strong>. Our regime differs on three axes at once: the entities are <strong>obscure</strong>, the language is <strong>low-resource</strong>, and the unit is a <strong>damaged fragment</strong> rather than a name. This deck climbs those axes one at a time and reports where the map survives and where it breaks.</div></div>
  </div>
</section>""",

"paper": """<section class="slide slide-text">
  <div class="eyebrow">The work we extend &middot; Gurnee &amp; Tegmark, ICLR 2024</div>
  <h2 class="sh">&ldquo;Language Models Represent Space and Time&rdquo;: a strong claim, tested in one cell</h2>
  <div class="text-points">
    <div class="tp"><div class="tp-h">Their method, which we adopt unchanged</div><div class="tp-b">Run an entity name through a frozen model; save the residual-stream activation at the <strong>last entity token</strong> for every layer; fit <strong>ridge regression</strong> (&lambda; by leave-one-out CV) to latitude/longitude or to a year; report <strong>R&sup2; on held-out entities</strong>. Six datasets, roughly 20&ndash;40k entities each: World, USA, NYC (space) and Figures, Art, Headlines (time).</div></div>
    <div class="tp"><div class="tp-h">Their finding</div><div class="tp-b">Recovery is strong (World R&sup2; = .911 at Llama-2-70B), rises with scale, and improves through the first half of the layers before plateauing. Nonlinear MLP probes add almost nothing, which is their argument that the structure really is linear.</div></div>
    <div class="tp"><div class="tp-h">What the setting quietly holds fixed</div><div class="tp-b"><strong>Three things at once.</strong> Every dataset is <em>salient</em> entities; every dataset is <em>English</em>; and the read-out is always an <strong>explicit entity token</strong>, a span the experimenter marks in advance. So the study cannot separate the model's geometry from the well-represented slice of the world it was trained on, and it never asks whether the same information is recoverable <strong>implicitly</strong>, from a whole text in which no entity is singled out. Our corpus varies all three.</div></div>
    <div class="tp"><div class="tp-h">And one control it does not run</div><div class="tp-b">No <strong>random-initialised twin</strong>. Without it, a probe's success cannot be separated from what a linear map extracts from any high-dimensional representation. We add that control everywhere, plus a <strong>TF-IDF character n-gram floor</strong>.</div></div>
  </div>
</section>""",

"matrix": """<section class="slide slide-text">
  <div class="eyebrow">The design</div>
  <h2 class="sh">The climbing map: entity salience &times; language resource</h2>
  <div class="exp-config">Going straight from the paper's setting to Akkadian changes <strong>two things at once</strong>, so a collapse could not be attributed to either. We lay the space out as a 2&times;2 and climb it one factor at a time.</div>
  <table class="rtbl matrix"><thead><tr><th></th><th>High-resource language (English)</th><th>Low-resource language (Akkadian)</th></tr></thead><tbody>
    <tr><td><span class="mdl">Salient entities</span></td><td><strong>A:</strong> the paper's cell. World capitals, famous figures, headlines. <em>Slides 6&ndash;8.</em></td><td><strong>D: empty.</strong> No famous entities exist in Akkadian outside these same royal names, so no honest filler exists.</td></tr>
    <tr><td><span class="mdl">Obscure entities</span></td><td><strong>B:</strong> Assyrian rulers and find-spots, written in English. <em>Slides 9&ndash;11.</em></td><td><strong>C:</strong> the same entities in raw Akkadian. <em>Slides 13&ndash;18.</em></td></tr>
  </tbody></table>
  <div class="text-points" style="margin-top:10px">
    <div class="tp"><div class="tp-h">A &rarr; B isolates entity obscurity</div><div class="tp-b">Same language, different entities. It also asks a second question: <strong>is the language itself the barrier to linear recoverability?</strong> If a faithful English translation restores the signal that raw Akkadian loses, then the knowledge is stored in activation space by <strong>meaning rather than surface form</strong>, and translation becomes a way to reach an LLM's internal representations for any low-resource language.</div></div>
    <div class="tp"><div class="tp-h">B &rarr; C isolates language resource</div><div class="tp-b">Same entities, different language. This is what tells us whether the model has learned the <strong>actual patterns of Akkadian</strong> from the real text, or whether it only ever had access to the content once someone rendered it in a language it knows.</div></div>
    <div class="tp"><div class="tp-h">Inside every cell: what counts as the entity, and how we pool</div><div class="tp-b">We first run the paper's <strong>entity-level</strong> protocol, then extend to <strong>whole fragments</strong>, and pool both at the <strong>last token</strong> (theirs) and as a <strong>mean over the text</strong> (new).</div></div>
  </div>
</section>""",

"protocol": """<section class="slide slide-text">
  <div class="eyebrow">Protocol</div>
  <h2 class="sh">How every experiment in this deck is set up, and how to read its numbers</h2>
  <div class="cfg">
    <div class="cfg-k">Models</div><div class="cfg-v">Llama-2 <strong>7B / 13B / 70B</strong> (the paper's own series) &middot; Qwen3 <strong>1.7B / 8B / 32B</strong> &middot; <strong>gpt-oss-120B</strong> &middot; three small seq2seq translation encoders (<strong>cuneiform-400M</strong>, <strong>AKK-300M</strong>, <strong>uMT5-base</strong>).</div>
    <div class="cfg-k">Controls</div><div class="cfg-v">a <strong>random-initialised twin</strong> of every decoder (same architecture and tokenizer, untrained weights) and a <strong>TF-IDF character n-gram floor</strong>. Both appear as rows in every results table, never as a slide of their own.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v"><strong>entity last token</strong> (the paper's read-out, an explicitly marked span) &middot; <strong>text last token</strong> (their Headlines read-out, applied to our whole fragments) &middot; <strong>mean over the text</strong> (new: no span is marked, so the signal must be recoverable implicitly, and it is the only option for the encoders, which have no causal last token).</div>
    <div class="cfg-k">Metrics</div><div class="cfg-v"><strong>R&sup2;</strong> on held-out entities, as in the paper, and <strong>Spearman &rho;</strong>, which the paper also reports. We lean on &rho; for year because dating is a ranking problem and our year distribution is compressed and unevenly spaced.</div>
    <div class="cfg-k">Balancing</div><div class="cfg-v"><strong>balanced Monte-Carlo</strong>, used wherever the corpus is ours: each of the 8 dense rulers is capped at the same number of fragments and the probe is refit over <strong>200 redrawn splits</strong>, so no score can come from one over-represented king. The paper's English datasets are already balanced and need none of this.</div>
    <div class="cfg-k">Verdict</div><div class="cfg-v">a score witnesses <em>learning</em> only if it beats <strong>both</strong> the TF-IDF floor <strong>and</strong> that arm's own random twin. Beating neither means the probe found geometry that any random projection would have offered.</div>
  </div>
  <p class="fig-note">Two protocol families run through the deck, and each slide states which one it uses in the box at the top. The <strong>replication line</strong> (cells A, B, C) follows the paper: last-token pooling, ridge, R&sup2;. The <strong>thesis line</strong> (the closing slides) uses what the applied dating work settled on: mean pooling, PLS, Spearman. Numbers are comparable within a line, not across the two.</p>
</section>""",

"cellA_repro": """<section class="slide slide-text">
  <div class="eyebrow">A &middot; salient entities, English &middot; the paper's own setting</div>
  <h2 class="sh">The paper reproduces on our models, and the controls it never ran hold up</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">the paper's six English datasets (World, USA, NYC &rarr; <strong>latitude/longitude</strong>; Figures, Art, Headlines &rarr; <strong>year</strong>). Each entity's last-token embedding, every layer, probed on held-out entities. We reproduce it with <strong>our extended set of models</strong> (both decoder families, gpt-oss-120B, three translation encoders) and with the <strong>controls the paper never ran</strong>.</div>
    <div class="cfg-k">Metrics</div><div class="cfg-v">best-layer held-out <strong>R&sup2;</strong> (the paper's headline) and <strong>Spearman &rho;</strong> (which the paper also reports), for all six datasets. Every cell shows <strong>Ridge</strong> then <strong>PLS</strong>, PLS taken at its best k &le; 64 on the same layer.</div>
  </div>
  <p class="tbl-cap">best-layer held-out test score &middot; each cell = <strong>Ridge</strong>|<span style="color:#6b7484">PLS</span> &middot; PLS at its best k &le; 64 on the same layer</p>
  {{TABLE:cellA}}
  <p class="fig-note"><strong>On English the published result holds, and it holds against controls the paper never ran.</strong> Our Llama-2-70B lands within .02 of every published number, and the effect extends to a second family: Qwen3 scales 1.7B &rarr; 8B &rarr; 32B exactly as the paper's scaling claim predicts. The gap to the controls is large in <em>both</em> space and time (Llama-2-70B World .905 against its random twin .170; Art .860 against .029), so the geometry is learned rather than architectural, and TF-IDF is a genuine floor that trained models clear and random ones fall below. The three <strong>translation models</strong> (uMT5-base, cuneiform-400M, AKK-300M) sit <em>above</em> the random twins but <em>below</em> TF-IDF on English: they are not generically good probes. <strong>PLS tracks Ridge closely</strong> for every decoder. <em>* = control.</em></p>
</section>""",

"cellA_layers": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">A &middot; depth</div>
  <h2 class="sh">Where in the network space and time live</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">per-layer <strong>ridge</strong> probe (no PLS on this slide), all six English datasets pooled into two groups: <strong>SPACE</strong> = World, USA, NYC (predicting latitude/longitude) and <strong>TIME</strong> = Figures, Art, Headlines (predicting year). Plotted against <strong>normalised depth</strong> (layer / total layers) so models of different depth, 28 to 41 layers, are directly comparable, as in the paper's Figure 2.</div>
    <div class="cfg-k">Reading</div><div class="cfg-v">rows = SPACE then TIME; columns = <strong>last token</strong> (the paper's read-out) then <strong>mean pool</strong> (new), each shown under <strong>both metrics</strong>, so nothing here departs from the paper: we only add the read-outs it also reports. R&sup2; uses a <strong>symlog</strong> axis, linear near zero and logarithmic further out, so the deep negative scores of the failing arms fit on the page without squashing the 0 to 1 band. Dashed = random-init controls; &#9733; = each arm's best layer.</div>
  </div>
  <div class="fig-wrap">{{FIG:fig_cellA_layers.png}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>The Qwen and Llama families sit clearly above both the random controls and the encoder-decoder translation models. <strong>Where each arm peaks is as informative as how high it peaks:</strong> for the weakest arms the best layer is at the very start of the network, which is what "no signal was built" looks like, whereas every arm that beats its controls peaks in the <strong>middle-to-late</strong> layers, reproducing the paper's depth profile for time under both poolings. In space the best layers are spread more evenly across depth and sit less often at the very start. Two arms are the exception worth watching in the space panels: they track near the bottom for most of the network under both poolings and then <strong>rise sharply in the last few layers</strong>.</div>
</section>""",

"cellA_pls": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">A &middot; dimensionality</div>
  <h2 class="sh">How many PLS directions the world model needs</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">at each arm's <strong>best ridge layer</strong> from the previous slide, refit with <strong>PLS</strong> using k = 1 to 64 components. Same six English datasets in the same two groups: <strong>SPACE</strong> = World, USA, NYC (latitude/longitude), <strong>TIME</strong> = Figures, Art, Headlines (year).</div>
    <div class="cfg-k">Reading</div><div class="cfg-v">rows = SPACE then TIME; columns = <strong>last token</strong> then <strong>mean pool</strong>, each under <strong>both metrics</strong>. Dashed = random-init controls; &#9733; = the k that maximises the score; the dash-dot vertical line marks k = 16.</div>
  </div>
  <div class="fig-wrap">{{FIG:fig_cellA_plsk.png}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>Most arms settle at around <strong>k &asymp; 16</strong> components. Year is the more concentrated of the two, space is more spread out, but both converge near 16. That is intuitively what the strong scores require: with only a handful of directions there would be no room for a meaningful subspace, and we would expect the weak results the <strong>random-init controls</strong> show, which saturate by k &asymp; 3 to 5 and gain nothing after that. The learned representation is genuinely multi-dimensional rather than one strong axis, a distinction that a single best-layer R&sup2; table hides completely.</div>
</section>""",

"b_entity": """<section class="slide slide-text">
  <div class="eyebrow">B &middot; the same test as the paper, on our entities</div>
  <h2 class="sh">Obscure entities in English: the date survives, the place does not</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">we repeat the paper's own two experiments and change <strong>only who the entities are</strong>. For time, in place of its famous historical figures (slide 6) we use the <strong>34 Assyrian and Babylonian rulers</strong> attested in our corpus, spanning <strong>1132 to 261 BC</strong> (Ashurbanipal, Sennacherib, Sargon II), and predict each ruler's year. For space, in place of its world places we use the <strong>25 excavation sites</strong> our fragments come from (Nineveh, Babylon, Assur, Kalhu) and predict latitude and longitude.</div>
    <div class="cfg-k">Input</div><div class="cfg-v">every name is probed <strong>on its own</strong>, exactly as the paper does (<em>&ldquo;Ashurbanipal&rdquo;</em>), and also inside five short sentences that never mention a date or a region (<em>&ldquo;This tablet dates to the reign of Ashurbanipal.&rdquo;</em>). That mirrors the paper's own robustness check, where the same entity is probed under several surrounding prompts. The table reports the name-alone rows; adding the sentences moves the top scores by less than 0.01.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v"><strong>name, last token</strong>: the activation at the final token of the name, which is the paper's read-out. <strong>name, average</strong>: the activation averaged over all the name's tokens, which is our addition and the only option for the translation encoders, since they have no causal last token.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">Spearman &rho;, averaged over <strong>200 redrawn splits</strong> that hold out 20&#37; of the entities each time. Every cell is <strong>Ridge</strong>|<span style="color:#6b7484">PLS</span>.</div>
  </div>
  {{TABLE:entity}}
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>Time holds up, and the scaling law with it:</strong> reading the last token of the name, Llama-2-70B leads at &rho; .701, both families order by size, and the top arms clear their own random twin (.457) and the n-gram baseline (.344). But the margin is roughly a quarter of what it was on famous entities, and by Llama-2-7B (.527 against its twin .473) it is inside the noise. We also tried a second read-out, <strong>averaging the activation over all of the name's tokens</strong>, which the paper never uses: it lands every model between .40 and .57 on year and never separates from the random controls on either target, so it is left out of the table. <strong>Space still fails, with one number worth a caveat</strong>: OLMo-2-7B posts the best place &rho; (.494 against its twin's .192), but that value carries a &plusmn;.28 spread over the 200 draws and a negative R&sup2; (&minus;.36) — it orders sites in some draws and cannot place them in any, the same shape as the n-gram floor. The next-best place number belongs to an <em>untrained</em> Llama-2-70B (.459), and no other model beats its control. 34 rulers and 25 sites means 6 to 7 held-out entities per draw, so read the ordering, not the third decimal.</div>
</section>""",

"b_frag_year": """<section class="slide slide-text">
  <div class="eyebrow">B &middot; from names to whole fragments</div>
  <h2 class="sh">Whole fragments in English: a bag of character n-grams dates them best</h2>
  <div class="cfg tight">
    <div class="cfg-k">Task</div><div class="cfg-v">predict <strong>the year a fragment was written</strong>. The entity is no longer a name but a <strong>whole tablet fragment</strong>, read in its English translation, so this is the paper's <em>news headlines</em> experiment carried over to our corpus: a full passage rather than a short name, with no entity marked anywhere in it. A fragment reads like <em>&ldquo;&hellip; the palace of my lordship which is in Nineveh I rebuilt and completed &hellip;&rdquo;</em>.</div>
    <div class="cfg-k">Data</div><div class="cfg-v">the corpus names <strong>40 rulers across 1193 dated fragments</strong>, but it is severely unbalanced: Ashurbanipal alone accounts for 268 and eighteen rulers have three fragments or fewer. We therefore restrict to the <strong>8 best-attested rulers</strong>, which is the panel an Assyriologist would actually trust and which still covers most of the corpus. Every ruler is then capped at the same number of fragments so frequency cannot stand in for chronology.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v"><strong>text, last token</strong>: the activation at the final token of the passage, which is what the paper uses for headlines. <strong>text, average</strong>: averaged over the whole passage. Nothing points the model at a date, so the signal has to be recoverable from the passage as a whole.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">Spearman &rho; and R&sup2;, averaged over <strong>200 balanced draws</strong> that cap each of the 8 best-attested rulers at the same number of fragments, so no single king can carry the score.</div>
  </div>
  {{TABLE:frag:eng_tier0:year}}
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>The n-gram baseline wins outright</strong> (&rho; .775), ahead of every embedding in the table. Pooling matters more than the model does: averaging over the passage adds about <strong>+.20 &rho;</strong> to almost every arm, which is the opposite of what we saw on bare names, because a date leaves its trace across the whole passage rather than at one token. But the arms that gain the most from it gain nothing <em>over their controls</em>: at the top, AKK-300M (.740) and Qwen3-8B (.737) sit barely above an <strong>untrained</strong> Llama-2-70B (.661) and an untrained Qwen3-8B (.636). Moving from famous names to obscure passages, the scaling law is gone and the models no longer separate from noise.</div>
</section>""",

"b_frag_geo": """<section class="slide slide-text">
  <div class="eyebrow">B &middot; from names to whole fragments</div>
  <h2 class="sh">Whole fragments in English: place is no better than an untrained network</h2>
  <div class="cfg tight">
    <div class="cfg-k">Task</div><div class="cfg-v">predict <strong>where the tablet was dug up</strong>, its latitude and longitude, from the same fragments and the same English translations as the previous slide. This is the paper's world-place experiment moved to whole passages: the find-spot is <em>never named in the text</em>, so unlike a city name there is nothing to look up and the coordinates have to come from whatever the passage implies.</div>
    <div class="cfg-k">Data</div><div class="cfg-v">1068 fragments whose find-spot is known, grouped into the <strong>10 excavation sites</strong> with enough material to hold one out, again capped equally so that a large site cannot carry the score. The year slide restricts by ruler; this one restricts by site, because the confound here is which dig a fragment came from rather than which king wrote it.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v">the same two read-outs: <strong>text, last token</strong> and <strong>text, average</strong>.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">Spearman &rho; and R&sup2; over <strong>200 draws that hold out whole find-spots</strong>, capping each of the 10 merged sites equally, so a probe cannot succeed by memorising which site a fragment came from.</div>
  </div>
  {{TABLE:frag:eng_tier0:geo}}
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>Every trained model is matched by its own untrained twin.</strong> Averaging over the passage, the best arm is cuneiform-400M (&rho; .640, R&sup2; .445), but an <em>untrained</em> Llama-2-7B reaches .606 / .463 and an untrained Qwen3-8B .587 / .450, so the reading rule is not satisfied anywhere in this table. As with the year, averaging beats last-token pooling for nearly every arm, and again the gain is shared with the controls rather than earned by training. The n-gram baseline is the interesting exception: it ranks well on &rho; (.535) yet collapses on R&sup2; (.022), meaning it orders sites roughly but cannot place them.</div>
</section>""",

"mlm_model": """<section class="slide slide-text">
  <div class="eyebrow">C &middot; the model we built for this language</div>
  <h2 class="sh">Before testing on Akkadian, we trained our own small model on the corpus</h2>
  <div class="two-col">
    <div>
      <div class="text-points">
        <div class="tp"><div class="tp-h">Why an Akkadian-only arm is needed</div><div class="tp-b">Every model so far was trained on text that is overwhelmingly English. A model whose entire experience <em>is</em> Akkadian separates &ldquo;the language cannot support a timeline&rdquo; from &ldquo;the models never saw enough of it&rdquo;.</div></div>
        <div class="tp"><div class="tp-h">The objective, and where it comes from</div><div class="tp-b"><strong>Masked language modelling</strong> rather than next-token prediction, following <a href="https://arxiv.org/abs/2109.04513"><em>Filling the Gaps in Ancient Akkadian Texts</em></a> (Stanovsky et al., EMNLP 2021), which showed that restoring a broken tablet <em>is</em> the masked-token task and that bidirectional context matters when neighbouring signs are damaged. The same instinct runs through <a href="https://www.nature.com/articles/s41586-022-04448-z">Ithaca</a> (Assael et al., Nature 2022) and <a href="https://www.nature.com/articles/s41586-025-09292-5">Aeneas</a> (Assael et al., Nature 2025) for Greek and Latin.</div></div>
        <div class="tp"><div class="tp-h">Architecture</div><div class="tp-b">A <strong>16-layer pre-norm transformer encoder</strong>, 37M parameters: d = 384, feed-forward 1536, 8 heads, RoPE positions, RMSNorm, MLM head over the sign vocabulary (15&#37; masking, 80/10/10). Trained 10 epochs; validation loss 4.55 &rarr; 3.24. The figure shows the full flow, with two deliberate differences from the Ithaca design it follows: a single <strong>sign-level</strong> input row (no word row), and a <strong>single restoration head</strong>, the region and date heads being replaced by linear probes on the frozen per-layer activations.</div></div>
        <div class="tp"><div class="tp-h">Data</div><div class="tp-b"><strong>2.45M words / 4.9M signs</strong> from ORACC, eBL and Archibab, split by fragment, tokenised at the <strong>sign level</strong> following the <a href="https://aclanthology.org/2025.alp-1.33/">EvaCun 2025 shared task</a>. It appears as <strong>MLM</strong> below: Akkadian all the way down, and the only arm that is <em>not</em> a translation model.</div></div>
      </div>
    </div>
    <div class="fig-wrap" style="min-height:0">{{FIG:fig_mlm_arch.png}}</div>
  </div>
</section>""",

"c_kingtoken": """<section class="slide slide-text">
  <div class="eyebrow">C &middot; raw Akkadian &middot; entity token against whole fragment</div>
  <h2 class="sh">In raw Akkadian the king's name is easy to read and the chronology is not</h2>
  <div class="cfg tight">
    <div class="cfg-k">The step</div><div class="cfg-v">we now change the <strong>language</strong>, holding the entities fixed: the same royal inscriptions, read in <strong>Akkadian</strong> rather than in translation. Two read-outs, mirroring the two we have used throughout: the <strong>ruler's name token</strong>, which is the closest thing this corpus has to the paper's marked entity, and the <strong>average over the whole fragment</strong>, where nothing is marked and the date must be recoverable from the passage as a whole.</div>
    <div class="cfg-k">Cleaning</div><div class="cfg-v">the fragment column uses our <strong>maximal</strong> regime, an eleven-filter pipeline plus truncation to 30 words. It strips digits, logograms (all-capital tokens), determinatives, case endings and plural markers, normalises long vowels, and lowercases everything. The reason is measured, not stylistic: without it a bag of character n-grams reaches 99&#37; accuracy by reading <em>document length and royal-name spellings</em>, because well-preserved eras leave long inscriptions and poorly-preserved ones leave scraps. Truncation removes the length crutch, the filters remove the name crutch. The name column deliberately keeps the names, since that is what it is measuring.</div>
    <div class="cfg-k">Target</div><div class="cfg-v"><strong>the year the fragment was written.</strong> Place is not on this slide: a find-spot is never written in the text, so there is no entity token to point at, and the coordinate experiments run on whole fragments only.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v"><strong>Spearman &rho;</strong>, because with 8 rulers the target takes only 8 distinct values and dating is a ranking problem; R&sup2; is unstable at that granularity and is reported for the whole-fragment tables instead. Averaged over <strong>200 balanced draws</strong> of 8 rulers x 21 fragments, grouped by ruler inside every draw. Each cell is <strong>PLS</strong>|<span style="color:#6b7484">Ridge</span>; a shuffled-label null sits at about 0.01.</div>
  </div>
  <table class="rtbl compact"><thead><tr><th>model</th><th class="num">whole fragment, average &nbsp;(names stripped)</th><th class="num">the ruler's name token &nbsp;(names kept)</th></tr></thead><tbody>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.316<i>|.273</i></td><td class="num">.645<i>|.153</i></td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">.332<i>|.302</i></td><td class="num">.645<i>|.425</i></td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.339<i>|.111</i></td><td class="num">.480<i>|.466</i></td></tr>
    <tr><td><span class="mdl">Qwen3-1.7B</span></td><td class="num">.334<i>|.072</i></td><td class="num">.606<i>|.500</i></td></tr>
    <tr><td><span class="mdl">cuneiform-400M</span></td><td class="num">.391<i>|.339</i></td><td class="num">.513<i>|.139</i></td></tr>
    <tr><td><span class="mdl">AKK-300M</span></td><td class="num">.300<i>|.308</i></td><td class="num">.688<i>|.459</i></td></tr>
    <tr><td><span class="mdl">uMT5-base</span></td><td class="num">.277<i>|.273</i></td><td class="num">.423<i>|.407</i></td></tr>
    <tr><td><span class="mdl">MLM</span> (ours, 37M)</td><td class="num">.286<i>|.288</i></td><td class="num">.704<i>|.649</i></td></tr>
    <tr class="rand"><td><span class="mdl">TF-IDF</span></td><td class="num">.271<i>|.269</i></td><td class="num">&ndash;</td></tr>
    <tr class="rand"><td><span class="mdl">random Qwen3-8B</span></td><td class="num">.293<i>|.226</i></td><td class="num">.643<i>|.495</i></td></tr>
  </tbody></table>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>The two columns are measuring different things, and only one of them is chronology.</strong> On the whole fragment every arm lands between .27 and .39, on top of the n-gram baseline (.271) and of an <em>untrained</em> Qwen3-8B (.293): in raw Akkadian, with the names stripped, nothing separates from noise. The name column looks far better, but the same untrained network scores <strong>.643</strong> there, matching every trained model and beating most of them, so that column is reading which king is named rather than when he ruled. Averaging over the ruler's name tokens, the third read-out, collapses to near zero for everyone and is left out. This is the same pattern as English at entity level, one step further down: obscurity weakened the signal, and switching to the low-resource language removes what was left.</div>
</section>""",

"ruler_not_chrono": """<section class="slide slide-text">
  <div class="eyebrow">C &middot; what the name token is really measuring</div>
  <h2 class="sh">The name token identifies the king, and an untrained network does it just as well</h2>
  <div class="cfg tight">
    <div class="cfg-k">The question</div><div class="cfg-v">the previous slide showed the ruler's name token scoring far above the whole fragment. Before reading that as chronology, we ask what the probe is actually doing there. Same activations, same draws, two read-outs: <strong>can it name the ruler</strong>, and <strong>can it order the years</strong>.</div>
    <div class="cfg-k">Metrics</div><div class="cfg-v"><strong>Ruler identification</strong> is macro-F1 over the 8 rulers, where chance is .20. <strong>Year ordering</strong> is Spearman &rho; against the true years. Both on the raw Akkadian, names left in, since the name is the thing being read.</div>
  </div>
  <table class="rtbl compact"><thead><tr><th>model</th><th class="num">identifies the ruler (macro-F1, chance .20)</th><th class="num">orders the years (&rho;)</th></tr></thead><tbody>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.989</td><td class="num">.974</td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">.982</td><td class="num">.977</td></tr>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.982</td><td class="num">.967</td></tr>
    <tr><td><span class="mdl">AKK-300M</span></td><td class="num">.975</td><td class="num">.962</td></tr>
    <tr><td><span class="mdl">cuneiform-400M</span></td><td class="num">.943</td><td class="num">.957</td></tr>
    <tr><td><span class="mdl">MLM</span> (ours)</td><td class="num">.970</td><td class="num">.977</td></tr>
    <tr class="rand"><td><span class="mdl">random Qwen3-8B</span></td><td class="num">.946</td><td class="num">.926</td></tr>
  </tbody></table>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>An untrained network scores .946 and .926.</strong> A randomly initialised Qwen3-8B identifies the eight kings almost perfectly and orders their years almost perfectly, matching every trained model in the table and beating several of them. Telling eight fixed spellings apart requires no learned history at all, and once you know which king wrote a text you know its year for free, because in this corpus each ruler carries a single date. So the name column is a <strong>lookup table, not a timeline</strong>: it measures whether the tokenizer preserved the name, and nothing else. Averaging over the name's tokens rather than taking the last one collapses to near zero for every arm, which is the same thing seen from the other side. <strong>This is why the rest of the deck reads the whole fragment with the names stripped</strong>, and it is the read-out under which the models stop separating from their controls.</div>
</section>""",

"ask_directly": """<section class="slide slide-text">
  <div class="eyebrow">Rescue 2 &middot; remove the probe, or help it</div>
  <h2 class="sh">Asking the model directly, and prompting it as hard as we can, changes nothing</h2>
  <div class="cfg">
    <div class="cfg-k">Why this slide</div><div class="cfg-v">the previous slide showed the models know these rulers in English, so perhaps the failure is the probe: maybe the date is present but a linear read-out cannot reach it. Two ways to find out, both predicting <strong>the year of a cleaned Akkadian fragment</strong>.</div>
    <div class="cfg-k">Ask it</div><div class="cfg-v">show the fragment to a chat model and take <strong>its written answer</strong> as the prediction, a forced single guess of ruler and year, with no probe anywhere in the loop.</div>
    <div class="cfg-k">Prompt it</div><div class="cfg-v">keep the probe, but read the activations from <em>inside</em> the prompt, so the model has been told what we are looking for before we look. Four styles in both cases: plain, expert framing, five worked examples, and chain-of-thought.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">Spearman &rho; against the true year on the same <strong>200 balanced draws</strong> over the 8 rulers used everywhere else, so these numbers sit on the same scale as the probe tables.</div>
  </div>
  <p class="tbl-cap">the written answer, used as the prediction (&rho;)</p>
  <table class="rtbl compact"><thead><tr><th>model</th><th class="num">bare</th><th class="num">expert</th><th class="num">few-shot</th><th class="num">CoT</th></tr></thead><tbody>
    <tr><td><span class="mdl">Qwen3-1.7B</span></td><td class="num">.010</td><td class="num">.123</td><td class="num">.009</td><td class="num">&minus;.074</td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.173</td><td class="num">.029</td><td class="num">.374</td><td class="num">.125</td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">&minus;.045</td><td class="num">.159</td><td class="num">.304</td><td class="num">.216</td></tr>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.168</td><td class="num">.299</td><td class="num">.239</td><td class="num">.217</td></tr>
  </tbody></table>
  <p class="tbl-cap" style="margin-top:8px">the probe, reading activations from inside the prompt (&rho;)</p>
  <table class="rtbl compact"><thead><tr><th>model</th><th class="num">bare</th><th class="num">expert</th><th class="num">few-shot</th><th class="num">CoT</th></tr></thead><tbody>
    <tr><td><span class="mdl">Qwen3-1.7B</span></td><td class="num">.313</td><td class="num">.313</td><td class="num">.287</td><td class="num">.314</td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.284</td><td class="num">.283</td><td class="num">.285</td><td class="num">.285</td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">.310</td><td class="num">.328</td><td class="num">.332</td><td class="num">.327</td></tr>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.310</td><td class="num">.318</td><td class="num">.333</td><td class="num">.320</td></tr>
  </tbody></table>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>Neither escape route works.</strong> Letting the model answer in its own words is <em>worse</em> than the probe: &rho; reaches .37 at best and is often near zero or negative, against the probe's own baseline of about .33. The models will name Assyrian kings fluently in conversation, as the previous slide showed, and still cannot turn a stripped Akkadian fragment into a date. And prompting moves nothing: across four styles and four scales the probed values sit within about .05 of the plain prompt, with no ordering by model size. The signal is not being hidden by a weak read-out or a badly posed question. It is not there.</div>
</section>""",

"c_frag_year": """<section class="slide slide-text">
  <div class="eyebrow">C &middot; raw Akkadian &middot; whole fragments</div>
  <h2 class="sh">Reading the date off a whole Akkadian fragment: every model falls to its untrained twin</h2>
  <div class="cfg tight">
    <div class="cfg-k">Task</div><div class="cfg-v">predict <strong>the year a fragment was written</strong>, exactly as two slides ago, but from the <strong>Akkadian text itself</strong> rather than from its English translation. Nothing else changes, so the gap between this table and that one is the cost of the language alone.</div>
    <div class="cfg-k">Text</div><div class="cfg-v">cleaned, with royal names and other surface giveaways stripped and every fragment cut to the same length, so a score cannot come from spotting a king's name or from the fact that well-preserved periods leave longer texts.</div>
    <div class="cfg-k">Data</div><div class="cfg-v">the same <strong>8 best-attested rulers</strong> as the English slides, capped equally, with <strong>200 redrawn balanced splits</strong> grouped by ruler so no king appears in both halves of a draw.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v"><strong>text, last token</strong> and <strong>text, average</strong>, the same two read-outs used throughout.</div>
  </div>
  {{TABLE:frag:akk_maximal:year}}
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>This is where the claim breaks.</strong> Averaging over the fragment, the best trained arm is cuneiform-400M at &rho; .699, but an <em>untrained</em> Llama-2-70B reaches .588 and an untrained Qwen3-8B .544, and on last-token pooling the untrained Qwen3-8B (.499) beats <em>every</em> trained model in the table. The n-gram baseline sits at .707, above all of them. Compared with the English translation of the very same fragments, every arm loses roughly .15 to .20 &rho;, so what the models had was access to English, not to the content. <strong>Only cuneiform-400M, a multilingual translation encoder, stays clearly above its controls</strong>, which is the one positive signal in this table and the thread the last part of the deck picks up.</div>
</section>""",

"c_frag_geo": """<section class="slide slide-text">
  <div class="eyebrow">C &middot; raw Akkadian &middot; whole fragments</div>
  <h2 class="sh">Reading the find-spot off a whole Akkadian fragment: place survives where the date does not</h2>
  <div class="cfg tight">
    <div class="cfg-k">Task</div><div class="cfg-v">predict <strong>latitude and longitude of the excavation site</strong> from the same cleaned Akkadian fragments. The site is never named in the text, so there is no name to recognise; whatever the probe finds has to come from how the fragment is written.</div>
    <div class="cfg-k">Data</div><div class="cfg-v">1068 fragments with a known find-spot, grouped into the <strong>10 sites</strong> with enough material, capped equally over <strong>200 draws that hold out whole sites</strong>, so a probe cannot win by memorising which dig a fragment came from.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v">the same <strong>last token</strong> and <strong>average</strong> read-outs. Note that place has no entity-token variant at fragment level: a find-spot is not a word in the text, so there is no span to point at, which is why only the two whole-text poolings appear here and on the English geo slide.</div>
  </div>
  {{TABLE:frag:akk_maximal:geo}}
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>Place behaves differently from date.</strong> Under last-token pooling several trained arms clear the n-gram baseline decisively on R&sup2; (Llama-2-13B and Llama-2-7B at .326 against TF-IDF's .019), which never happens for the year. But the controls climb with them: an untrained Qwen3-8B reaches .351, the best number in that column. Averaging again favours cuneiform-400M (&rho; .612, R&sup2; .430), and again its untrained neighbours are close behind. So the honest reading is that <strong>coordinates are partially recoverable from the writing itself while the date is not</strong>, and that most of what makes place recoverable is available to an untrained network as well.</div>
</section>""",

"c_layers": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">C &middot; depth</div>
  <h2 class="sh">Layer by layer on Akkadian: only the translation encoder builds depth</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">the same per-layer ridge probe as slide 7, now on the <strong>cleaned raw Akkadian fragments</strong>, plotted against normalised depth so models of different sizes are comparable. Four panels: <strong>year</strong> (Spearman, the ranking read-out dating needs) and <strong>place</strong> (R&sup2;, the paper's read-out for coordinates), each under <strong>last token</strong> and <strong>average</strong> pooling. Dashed = untrained controls, &#9733; = each arm's best layer. The English-translation panels are not repeated here; they are the cell B slides.</div>
  </div>
  <div class="fig-wrap">{{FIG:fig_cellC_layers.png}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>On the cleaned, balanced Akkadian text with average pooling, <strong>cuneiform-400M is the best arm by a clear margin on both year and place</strong>, and it is a <em>multilingual</em> translation model rather than an Akkadian-only one. Its best layers sit <strong>late</strong> in the network, whereas most other arms peak in the <strong>first few layers</strong>, which is what a representation that never deepens looks like: the probe is reading the input surface, not something the network built. The encoders have no true causal last token and unsurprisingly do worse in the last-token panels. <strong>gpt-oss-120B is the odd one out</strong>: it collapses through the middle of the network and then climbs steeply over the final layers, a shape no other arm shows and a hint that it organises this input differently.</div>
</section>""",

"c_plsk": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">C &middot; dimensionality</div>
  <h2 class="sh">How many directions the Akkadian signal needs: far fewer than English, and it decays after</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">at each arm's best layer from the previous slide, refit with <strong>PLS using k = 1 to 64 components</strong>. Same four panels, same colours, same controls. The dash-dot line marks k = 16, which is where the English curves plateaued.</div>
  </div>
  <div class="fig-wrap">{{FIG:fig_cellC_plsk.png}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>The signal is exhausted much sooner here than in English.</strong> Almost every arm peaks between <strong>k = 5 and k = 16</strong> and then <em>declines</em>, sharply so by k = 32 and 64, where the place panels fall below zero. On English the same probes kept gaining to k = 16 and held their value afterwards. A representation that cannot use more than a handful of directions, and that is actively hurt by being given more, is not a rich geometry: it is a thin surface feature. That is the dimensionality counterpart of the flat layer curves on the previous slide, and it is what a low-resource, obscure-entity regime looks like from the inside.</div>
</section>""",

"t9_knowledge": """<section class="slide slide-text">
  <div class="eyebrow">Rescue 1 &middot; is the knowledge simply absent?</div>
  <h2 class="sh">The models do know these kings and their dates, when you simply ask them in English</h2>
  <div class="cfg tight">
    <div class="cfg-k">Why this slide</div><div class="cfg-v">everything so far says the probe cannot read a date out of Akkadian. That has two possible explanations: the model never learned these rulers at all, or it knows them but the knowledge is not linearly available in the representation the probe reads. This slide rules out the first.</div>
    <div class="cfg-k">Setup</div><div class="cfg-v">no probe and no Akkadian. We ask the chat models two plain English questions: <em>&ldquo;When did {ruler} reign?&rdquo;</em> for each of the 8 rulers, answered as a start and end year, and <em>&ldquo;list the rulers of this period&rdquo;</em>. Names are matched ignoring diacritics.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v"><strong>reign accuracy</strong>: the answered window, widened by 10 years, must contain a true reign year. <strong>Recall</strong>: the share of the period's rulers the model names. Accuracy is identical at 50, 30 and 10 year tolerances, so the strictest is shown.</div>
  </div>
  <table class="rtbl compact"><thead><tr><th>model</th><th class="num">reign dates correct</th><th class="num">rulers recalled</th></tr></thead><tbody>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">8 / 8</td><td class="num">1.00</td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">6 / 8</td><td class="num">1.00</td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">8 / 8</td><td class="num">0.75</td></tr>
    <tr><td><span class="mdl">Qwen3-1.7B</span></td><td class="num">7 / 8</td><td class="num">0.50</td></tr>
  </tbody></table>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>The knowledge is there.</strong> gpt-oss-120B and Qwen3-8B date all eight rulers correctly to within ten years and the larger models name every ruler of the period. So when the probe finds nothing in the Akkadian representation, it is not because the model has never heard of Ashurbanipal. Declarative knowledge that a model can state in English is simply <em>not the same thing</em> as a linearly decodable axis in its activations over Akkadian text, and this deck is measuring the second.</div>
</section>""",

"ntp_finetune": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">Rescue 3 &middot; more Akkadian</div>
  <h2 class="sh">Training the LLMs further on our own Akkadian moves nothing at all</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">we continued pretraining <strong>Qwen3-1.7B, Qwen3-8B, Qwen3-32B and gpt-oss-120B</strong> on our Akkadian fragments with ordinary <strong>next-token prediction</strong>, the objective they were built with, using the training split of the same corpus the MLM was trained on (the test fragments are never touched). Each model was fine-tuned at <strong>four unfreezing depths</strong>, from the whole network down to only the top tenth, so the update could be placed where the probe actually reads. The tokenizer was left alone: the tokens worth adding are royal and divine names, which is exactly the leakage channel the cleaning removes.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">year Spearman &rho; on the cleaned Akkadian text, 200 balanced draws, re-probing every checkpoint with the identical protocol. Solid bar = base model, hatched = after fine-tuning, dotted line = the base level.</div>
  </div>
  <div class="fig-wrap">{{FIG:fig_finetune_ntp.png}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>Nothing moves.</strong> Across four models and four unfreezing depths the change is between &minus;.013 and +.002 &rho;, far inside the spread of the draws, and for the two largest models several arms are numerically identical to base because the layers the probe reads were frozen. More exposure to Akkadian, delivered through the objective these models were built with, does not create a temporal axis. The honest caveat is <strong>scale of data</strong>: our corpus is about 2.5M words, which is all the published Akkadian there is, but it is four orders of magnitude below these models' pretraining. So the fair claim is not that more Akkadian could never help; it is that <em>all the Akkadian that exists</em>, used this way, does not.</div>
</section>""",

"shuffle": """<section class="slide slide-text">
  <div class="eyebrow">Rescue 4 &middot; word order</div>
  <h2 class="sh">Scrambling the word order costs almost nothing, so the probe is reading a bag of words</h2>
  <div class="cfg tight">
    <div class="cfg-k">Idea</div><div class="cfg-v">if a probe is reading grammar, syntax or anything about how a sentence is built, destroying the word order should destroy the score. If it is reading which words are present, shuffling should change nothing. This is a direct test of what kind of signal the earlier tables actually contain.</div>
    <div class="cfg-k">Setup</div><div class="cfg-v">every fragment gets an <strong>exact twin with its words randomly permuted</strong>, one twin per fragment, same words and same length. Both are embedded with identical settings and probed identically. We run it on the <strong>cleaned Akkadian</strong> and on the <strong>English translation</strong> so the answer is not specific to one language.</div>
    <div class="cfg-k">Target &amp; metric</div><div class="cfg-v"><strong>year</strong>, Spearman &rho;, average pooling, 200 balanced draws over the 8 rulers, best layer per arm, <strong>PLS</strong> (Ridge agrees and is in the CSV).</div>
  </div>
  <table class="rtbl compact"><thead><tr><th rowspan="2">model</th><th colspan="3" class="num">cleaned Akkadian</th><th colspan="3" class="num">English translation</th></tr><tr><th class="num">in order</th><th class="num">shuffled</th><th class="num">cost</th><th class="num">in order</th><th class="num">shuffled</th><th class="num">cost</th></tr></thead><tbody>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.316</td><td class="num">.297</td><td class="num">.018</td><td class="num">.411</td><td class="num">.365</td><td class="num">.045</td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">.332</td><td class="num">.295</td><td class="num">.037</td><td class="num">.420</td><td class="num">.411</td><td class="num">.009</td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.339</td><td class="num">.277</td><td class="num">.062</td><td class="num">.397</td><td class="num">.384</td><td class="num">.013</td></tr>
    <tr><td><span class="mdl">Qwen3-1.7B</span></td><td class="num">.336</td><td class="num">.292</td><td class="num">.044</td><td class="num">.355</td><td class="num">.350</td><td class="num">.004</td></tr>
    <tr><td><span class="mdl">cuneiform-400M</span></td><td class="num">.390</td><td class="num">.396</td><td class="num">&minus;.006</td><td class="num">.388</td><td class="num">.361</td><td class="num">.027</td></tr>
    <tr><td><span class="mdl">AKK-300M</span></td><td class="num">.300</td><td class="num">.273</td><td class="num">.027</td><td class="num">.400</td><td class="num">.381</td><td class="num">.018</td></tr>
    <tr><td><span class="mdl">uMT5-base</span></td><td class="num">.278</td><td class="num">.278</td><td class="num">.000</td><td class="num">.343</td><td class="num">.335</td><td class="num">.009</td></tr>
    <tr><td><span class="mdl">MLM</span> (ours)</td><td class="num">.285</td><td class="num">.285</td><td class="num">.001</td><td class="num">&ndash;</td><td class="num">&ndash;</td><td class="num">&ndash;</td></tr>
    <tr class="rand"><td><span class="mdl">TF-IDF</span></td><td class="num">.266</td><td class="num">.266</td><td class="num">.000</td><td class="num">&ndash;</td><td class="num">&ndash;</td><td class="num">&ndash;</td></tr>
  </tbody></table>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>Word order carries almost none of it.</strong> Destroying the order of every word costs between .00 and .06 &rho;, in both languages and at every scale, and for cuneiform-400M on Akkadian the shuffled text scores <em>marginally higher</em>. TF-IDF is order-blind by construction and loses exactly nothing, which is the point of the comparison: these models are behaving like a bag of words for this task. Whatever chronological signal exists is carried by <em>which</em> words appear, not by how they are arranged, which is why the probe cannot be reading grammatical change over time.</div>
</section>""",

"conditions": """<section class="slide slide-text">
  <div class="eyebrow">Synthesis</div>
  <h2 class="sh">What a linear temporal world model needs in order to exist</h2>
  <div class="exp-config">Reading the ladder back down, the collapse is attributable, and the failed rescues say what is <em>not</em> responsible for it.</div>
  <div class="text-points">
    <div class="tp"><div class="tp-h">Condition 1: the language must be well represented in training</div><div class="tp-b">This is the load-bearing factor. Holding the entities fixed and moving from English to Akkadian drops every trained model <strong>onto its own untrained twin</strong>, while a character n-gram baseline beats them all. The geometry the paper found does not transfer to a language the model barely saw.</div></div>
    <div class="tp"><div class="tp-h">Condition 2: the entities must be salient enough to have been written about</div><div class="tp-b">Moving from famous names to obscure ones weakens the signal before any language change, and the n-gram baseline starts to lead on time. A linear timeline needs entities the model has read <em>about</em>, repeatedly and in dated context, not merely names it can spell.</div></div>
    <div class="tp"><div class="tp-h">Condition 3: the read-out must sit where the information is</div><div class="tp-b">Pooling was never a side detail. On bare names the paper's entity-last-token choice carries the result and averaging destroys it; on whole fragments the relation flips and averaging is worth about +.20. A claim about what a model represents is inseparable from where you read it.</div></div>
    <div class="tp"><div class="tp-h">What does not substitute for any of them</div><div class="tp-b"><strong>Scale</strong> (flat from 1.7B to 120B on Akkadian, and gpt-oss-120B lands mid-pack even on English) &middot; <strong>declarative knowledge</strong> (the models recite these kings' reigns in English) &middot; <strong>prompting</strong>, <strong>asking directly</strong>, <strong>continued pretraining on all the Akkadian there is</strong>, <strong>word order</strong>, <strong>curved and kernel probes</strong>, <strong>unsupervised geometry</strong>. Every rescue leaves the numbers where they were.</div></div>
  </div>
</section>""",

"winner": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">The positive result</div>
  <h2 class="sh">What does work: the 400M translation encoder beats every LLM at dating Akkadian</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">the full model set under the honest protocol: cleaned Akkadian, average pooling, 200 balanced draws, <strong>PLS</strong> with <span style="color:#6b7484">Ridge</span> alongside as the check. Year Spearman &rho;, best layer per arm.</div>
  </div>
  <div class="fig-wrap">{{IMG:4}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>cuneiform-400M ends the deck where the probes kept pointing:</strong> the highest balanced year score of every source, above all the Qwen scales, the 120B model, our own Akkadian-only MLM, and the n-gram baseline, and it is the <strong>only arm that clears both of its controls</strong> on this task. PLS and Ridge agree on the ordering, so the win is not an artifact of the dimensionality reduction. A 400M translation encoder, not a 120B language model, is the system that actually dates Akkadian.</div>
</section>""",

"translation_line": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">Mechanism &middot; the objective, isolated</div>
  <h2 class="sh">Why it works: translation finetuning, and multilingual translation most of all</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">the three encoders of the same family, layer by layer, so the objective is the only thing that varies: <strong>uMT5-base</strong> (multilingual pretraining, no translation finetune), <strong>AKK-300M</strong> (translation finetuned on Akkadian alone) and <strong>cuneiform-400M</strong> (translation finetuned on the multilingual cuneiform family). The dashed line is the untrained Qwen3-8B, the smallest untrained control we have (no 1.7B-scale random twin exists). All four panels are the <strong>cleaned Akkadian text</strong>: YEAR = Spearman, PLACE = R&sup2;, each under last-token and average pooling. The English-translation counterpart shows the same ordering and lives with the cell B slides.</div>
  </div>
  <div class="fig-wrap">{{FIG:fig_encoders_translation.png}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>The ordering is the argument: <strong>no translation finetune &lt; Akkadian-only translation &lt; multilingual translation</strong>, on year and on place, most clearly under average pooling on the Akkadian text, where cuneiform-400M rises with depth to its late-layer peak while uMT5 decays and the untrained control stays flat. Training a model to map a low-resource language onto meaning is what builds the recoverable structure, and training it across a <em>family</em> of related low-resource languages helps beyond Akkadian exposure alone, consistent with <a href="https://scholar.google.com/citations?view_op=view_citation&amp;hl=iw&amp;user=2dBD8o8AAAAJ&amp;citation_for_view=2dBD8o8AAAAJ:d1gkVwhDpl0C">Stanovsky et al. (2022)</a> on multilingual transfer for low-resource ancient languages.</div>
</section>""",

"tokenizer": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">Mechanism &middot; ruling out the tokenizer</div>
  <h2 class="sh">It is not the tokenizer: the winner has the worst one</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">tokens per Akkadian word for every model's tokenizer, over our corpora. If tokenizer fit drove dating performance, the most efficient tokenizer should win.</div>
  </div>
  <div class="fig-wrap">{{IMG:8}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>The ranking runs the wrong way for the tokenizer story: <strong>cuneiform-400M splits Akkadian into more pieces than any other model</strong> (6.22 tokens per word, the least efficient of all) and wins anyway, while gpt-oss-120B has the most efficient tokenizer (4.43) and sits mid-pack. Whatever the encoder learned, it learned it <em>through</em> a poor segmentation of the text, which rules the tokenizer out as the cause and leaves the training objective as the live explanation.</div>
</section>""",

"explorer_eng": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">B &middot; looking inside the embedding space</div>
  <h2 class="sh">Inside the best English-side embedding: the year gradient is visible, and so are its confounds</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">the 1,202 fragments of the corpus, embedded by the best English-side model (<strong>Qwen3-32B</strong> on the English translation, year &rho; .437, rank 1 of 9) and projected to 2-D with <strong>supervised PLS</strong>; we also built t-SNE, UMAP and PCA views of every arm. The same map is drawn six times, coloured by our metadata: year, ruler, period, sub-genre, find-spot, and text length.</div>
  </div>
  <div class="fig-wrap">{{FIG:e6_clusters/embedding_panels/engtier0/pls/qwen3_32b.png}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>The six colourings of one map are the whole story of this deck in one picture: the <strong>year panel shows a real gradient</strong>, but the ruler and period panels show the same regions, the find-spot panel shows them again, and the <strong>length panel shows a gradient of its own</strong>. Date, king, place, genre and preservation are woven together in this corpus, which is exactly why every number in this deck had to be balanced, name-stripped and length-controlled before it could be believed. An interactive explorer with every model, reduction and colouring ships with the repo at <span class="ph-slot" style="display:inline;padding:2px 7px">v_1/src/stress_tests/e6_clusters/embedding_panels/index.html</span> (open next to this deck; interactive.html is the self-contained viewer).</div>
</section>""",

"explorer_akk": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">C &middot; looking inside the embedding space</div>
  <h2 class="sh">Inside the winner's Akkadian embedding: the same gradient, from the raw language</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">the same six-way view for the best Akkadian-side model: <strong>cuneiform-400M</strong> on the cleaned, name-stripped Akkadian (year &rho; .391, rank 1 of 10), supervised PLS projection at its best layer.</div>
  </div>
  <div class="fig-wrap">{{FIG:e6_clusters/embedding_panels/maximal/pls/thalesian_cunei400m.png}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>This is what the one surviving arm's space looks like on the raw language, with the names stripped: a coarser but <strong>still-visible year gradient</strong> (Neo-Babylonian mass separating from the Neo-Assyrian core), organised less by ruler identity than the English map and with the length gradient weakened by the truncation. Read against the untrained and n-gram panels in the same folder, this is the difference between an embedding that merely <em>sorts surface form</em> and one that has begun to order the language in time. It is a beginning, not a solved problem, and that is the honest place to end.</div>
</section>""",

"contributions": """<section class="slide slide-text">
  <div class="eyebrow">Main point and takeaways</div>
  <h2 class="sh">The boundary condition: where the linear world model ends, and the one way past it</h2>
  <div class="text-points" style="gap:10px">
    <div class="tp"><div class="tp-h">Main point</div><div class="tp-b" style="font-size:14px;line-height:1.5">Prior work recovered linear spatial and temporal geometry at entity-token sites, for salient entities, in a high-resource language. We confirm that result is real: it reproduces on our models within .02 R&sup2;, scales across a second family, and towers over the controls it never ran (.905 against an untrained twin at .170). Moving it one factor at a time: obscure entities <em>weaken</em> the geometry (.701 vs .457, a quarter of the original margin); the low-resource language <em>kills</em> it. What probes recover from LLM activations over Akkadian is not the model's knowledge but token identity and shallow surface statistics. The knowledge is real, but keyed to names and accessed through the high-resource language; it does not exist as document-level structure that scale, prompting, or finetuning can surface. The one lever that installs a genuine document-level increment is the training configuration: <strong>cross-lingual translation supervision, not size</strong>. The contribution is this boundary condition, established with random-init twins, matched n-gram baselines and confounder-controlled probing. Read cautiously, it also says something about robustness: when the input moves even somewhat out of distribution, in entity familiarity or in language, the world model does not come along; these linear maps appear more tightly tied to the training distribution than the term suggests, and how far they generalize is an empirical question per regime, not a property one can assume.</div></div>
    <div class="tp"><div class="tp-h">1. The &ldquo;internal timeline&rdquo; (<a href="https://arxiv.org/abs/2310.02207">Gurnee &amp; Tegmark, ICLR 2024</a>) does not travel</div><div class="tp-b" style="font-size:13.5px">At their own pooling site, on obscure entities in Akkadian, an untrained network recovers the same signal: what probes read in low-resource text is name morphology and bag-of-tokens statistics, flat from 1.7B to 120B.</div></div>
    <div class="tp"><div class="tp-h">2. Geography (<a href="https://aclanthology.org/2024.lrec-main.1087.pdf">Godey et al., LREC 2024</a>) has no scaling law at the document level</div><div class="tp-b" style="font-size:13.5px">Their smooth spatial scaling lives at templated entity-token sites for salient places. At our document site a random-initialized network carries most of the signal, and richer probe geometry does not rescue it.</div></div>
    <div class="tp"><div class="tp-h">3. The models genuinely know the facts, but the knowledge is name-gated, not document-structured</div><div class="tp-b" style="font-size:13.5px">Asked in English they state reign windows exactly, and dating from translations beats every activation probe; the access route runs through restored surface forms, not through temporal structure built from the document.</div></div>
    <div class="tp"><div class="tp-h">4. What helps is how the model was trained, not how big it is or how you use it</div><div class="tp-b" style="font-size:13.5px">Scale, prompting, direct questioning and continued pretraining on all the published Akkadian all leave trained models at their controls. A 400M model with <strong>multilingual translation supervision</strong> is the one exception, and the ablation triangle (Akkadian-only translation fails, untranslated multilingual base fails) isolates the combination as the cause; consistent with <a href="https://arxiv.org/pdf/2205.04086">Malkin et al. (NAACL 2022)</a> on configuration over multilinguality in cross-lingual transfer.</div></div>
  </div>

</section>""",
}

# eyebrow rewrites so reused slides announce their act
EYEBROW_PATCHES = {
    24: "Cell A &middot; the paper's own setting &middot; salient entities, English",
    29: "Cell A &middot; depth",
    30: "Cell A &middot; dimensionality",
    25: "Cell B &middot; fragment level &middot; English gloss &middot; balanced Monte-Carlo",
    26: "Cell B &middot; fragment level &middot; English gloss &middot; by-site geo",
    15: "Cell C &middot; entity level &middot; the king-name token",
    27: "Cell C &middot; fragment level &middot; raw Akkadian &middot; balanced Monte-Carlo",
    28: "Cell C &middot; fragment level &middot; raw Akkadian &middot; by-site geo",
    31: "Cell C &middot; depth &middot; the Akkadian mirror",
    32: "Cell C &middot; dimensionality &middot; the Akkadian mirror",
    13:  "Rescue 1 &middot; does it already know?",
    7:   "Rescue 3 &middot; scale and continued pretraining",
    19:  "Rescue 4 &middot; word order",
    20:  "Rescue 5 &middot; non-linear probes",
    21:  "Rescue 6 &middot; supervision",
    4:   "The positive result",
    9:   "Mechanism &middot; the objective, isolated",
    8:   "Mechanism &middot; ruling out the tokenizer",
}


EXTRA_CSS = """
/* --- added by build_story_deck.py --- */
.two-col{display:grid;grid-template-columns:1fr 1.15fr;gap:16px;flex:1;min-height:0;overflow:hidden;}
.two-col .text-points{gap:8px;overflow:auto;padding-right:4px;}
.two-col .tp-h{font-size:13.5px;margin-bottom:2px;}
.two-col .tp-b{font-size:12.5px;line-height:1.42;}
.two-col .fig-wrap{min-height:0;height:100%;margin:0;}
.two-col .fig-wrap img{max-width:100%;max-height:100%;object-fit:contain;border:none;}
.arch{border-left:1px solid var(--border);padding-left:18px;font-size:10.5px;
      display:flex;flex-direction:column;justify-content:center;}
.arch-cap{font-size:9px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;
          color:var(--green);margin-bottom:5px;}
.arch-row{display:flex;gap:5px;justify-content:center;margin:1px 0;}
.arch-ar{text-align:center;color:var(--ink-light);font-size:11px;line-height:1;}
.ab{border:1px solid var(--border);border-radius:5px;padding:4px 8px;text-align:center;
    background:#fbfcfd;color:var(--ink-mid);line-height:1.3;}
.ab.in{background:#eef1f6;}
.ab.torso{background:var(--green-bg);border-color:var(--green-mid);color:var(--green);
          font-weight:700;flex:1;}
.ab.head{background:#fff;border-color:var(--green-mid);color:var(--green);font-weight:700;}
.ab.head.off{opacity:.35;text-decoration:line-through;font-weight:400;}
.ab.probe{background:#fdfaf0;border-color:#e3d7a8;color:#7c5e00;font-weight:700;flex:1;}
.ab-sub{font-weight:400;font-size:9px;color:var(--ink-light);}
.arch-inner{display:flex;flex-direction:column;gap:3px;margin-top:6px;}
.ab.sub-b{background:#fff;border-color:var(--border);color:var(--ink-mid);
          font-weight:500;font-size:9.5px;padding:3px 6px;}
/* dense: the generated entity/fragment tables, which grew to 17 rows when the OLMo
   arm and its twin joined; at the compact size the takeaway fell off the slide. */
.rtbl.compact.dense{font-size:9.8px;}
.rtbl.compact.dense th,.rtbl.compact.dense td{padding:1.4px 7px;}
.rtbl.compact.wide{font-size:9.5px;}
.rtbl.compact.wide th,.rtbl.compact.wide td{padding:1.5px 3px;}
.rtbl.compact.wide td i{font-style:normal;color:var(--ink-light);}
.rtbl.compact.wide thead tr:first-child th{border-bottom:none;padding-bottom:0;}
.rtbl.matrix{font-size:15px;margin-top:6px;}
.rtbl.matrix th,.rtbl.matrix td{padding:13px 18px;vertical-align:top;line-height:1.45;}
.rtbl.matrix thead th{font-size:12px;letter-spacing:.05em;}
.rtbl.matrix td:first-child{white-space:nowrap;}
.takeaway.tight{font-size:12.5px;line-height:1.45;padding:9px 15px;}
.takeaway.tight .tk-label{font-size:9px;}
.cfg.tight{font-size:11.5px;gap:3px 11px;margin-bottom:8px;}
.cfg.tight .cfg-k{font-size:9px;}
/* figure-first slides: shrink every non-figure block so the plot gets the page */
.slide.fig-major{padding:26px 30px 30px;}
.slide.fig-major .sh{font-size:19.5px;line-height:1.15;margin-bottom:5px;}
.slide.fig-major .eyebrow{font-size:9.5px;margin-bottom:4px;}
.slide.fig-major .cfg.tight{font-size:10px;line-height:1.35;gap:2px 9px;margin-bottom:5px;}
.slide.fig-major .cfg.tight .cfg-k{font-size:8px;}
.slide.fig-major .fig-wrap{margin:0 0 5px;}
.slide.fig-major .fig-wrap img{max-height:100%;width:auto;border:none;}
.slide.fig-major .takeaway.tight{font-size:10.5px;line-height:1.38;padding:6px 12px;}
.slide.fig-major .takeaway.tight .tk-label{font-size:8px;margin-right:7px;}
.cellmap{position:absolute;top:15px;right:18px;display:grid;
         grid-template-columns:auto 34px 34px;grid-auto-rows:auto;gap:2px;
         font-family:var(--sans);z-index:5;}
.cm-h{font-size:7.5px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;
      color:var(--ink-light);text-align:center;padding-bottom:1px;}
.cm-r{font-size:7.5px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;
      color:var(--ink-light);text-align:right;padding-right:3px;align-self:center;}
.cm-c{width:34px;height:22px;border:1px solid var(--border);border-radius:3px;
      display:flex;align-items:center;justify-content:center;
      font-size:11px;font-weight:800;color:var(--ink-light);background:#fbfcfd;}
.cm-c.on{background:var(--green);border-color:var(--green);color:#fff;
         box-shadow:0 1px 5px rgba(26,92,58,.35);}
.cm-c.done{background:var(--green-bg);border-color:var(--green-mid);color:var(--green);}
.cm-c.na{background:#f4f4f6;color:#c3c7d0;border-style:dashed;}
</style>"""


WM = os.path.abspath(os.path.join(HERE, "..", "..", "world_models"))
DATASETS = [("world_place", "World"), ("us_place", "USA"), ("nyc_place", "NYC"),
            ("historical_figure", "Figures"), ("art", "Art"), ("headline", "Headlines")]
TABLE_ROWS = [
    ("llama2_70b", "Llama-2-70B", False), ("llama2_13b", "Llama-2-13B", False),
    ("llama2_7b", "Llama-2-7B", False), ("olmo2_7b", "OLMo-2-7B", False),
    ("gpt_oss_120b", "gpt-oss-120B", False),
    ("qwen3_32b", "Qwen3-32B", False), ("qwen3_8b", "Qwen3-8B", False),
    ("qwen3_1b7", "Qwen3-1.7B", False), ("umt5_base", "uMT5-base", False),
    ("thalesian_cunei400m", "cuneiform-400M", False),
    ("thalesian_akk300m", "AKK-300M", False),
    ("tfidf", "TF-IDF", True),
    ("llama2_70b_random", "Llama-2-70B random*", True),
    ("llama2_13b_random", "Llama-2-13B random*", True),
    ("llama2_7b_random", "Llama-2-7B random*", True),
    ("olmo2_7b_random", "OLMo-2-7B random*", True),
    ("random", "random Qwen3-8B*", True),
]


def _fmt(v):
    if v is None or v != v:
        return "&ndash;"
    t = f"{v:.3f}"
    return t.lstrip("0") if t.startswith("0.") else t


def cellA_table():
    """Best-layer English results: every dataset x {R2, Spearman} x {Ridge | PLS}.
    Built from the committed CSV/JSON so the slide cannot drift from the results."""
    import csv
    import json as _json

    def load(name):
        with open(os.path.join(WM, "results", name)) as f:
            return {r["method"]: r for r in csv.DictReader(f)}
    r2, rho = load("summary_best_layer_r2.csv"), load("summary_best_layer_spearman.csv")

    pls = {}
    pls_dir = os.path.join(WM, "results", "eng_pls")
    for arm in os.listdir(pls_dir) if os.path.isdir(pls_dir) else []:
        pls[arm] = {}
        for ds, _ in DATASETS:
            f = os.path.join(pls_dir, arm, f"{ds}.last.json")
            if not os.path.exists(f):
                continue
            at = _json.load(open(f))["pls_at_best_layer"]
            pls[arm][ds] = (max(v["test_r2"] for v in at.values()),
                            max(v["test_spearman"] for v in at.values()))

    head = ('<table class="rtbl compact wide"><thead><tr><th rowspan="2">model</th>'
            + "".join(f'<th colspan="2" class="num">{lab}</th>' for _, lab in DATASETS)
            + '</tr><tr>'
            + "".join('<th class="num">R&sup2;</th><th class="num">&rho;</th>'
                      for _ in DATASETS) + '</tr></thead><tbody>')
    body = []
    for arm, label, ctrl in TABLE_ROWS:
        cells = []
        for ds, _ in DATASETS:
            gr = float(r2[arm][ds]) if arm in r2 else None
            gs = float(rho[arm][ds]) if arm in rho else None
            pr, ps = pls.get(arm, {}).get(ds, (None, None))
            cells.append(f'<td class="num">{_fmt(gr)}<i>|{_fmt(pr)}</i></td>'
                         f'<td class="num">{_fmt(gs)}<i>|{_fmt(ps)}</i></td>')
        tr = '<tr class="rand">' if ctrl else '<tr>'
        body.append(tr + f'<td><span class="mdl">{label}</span></td>'
                    + "".join(cells) + '</tr>')
    paper = ('<tr><td><span class="mdl">Llama-2-70B (paper)</span></td>'
             + "".join(f'<td class="num">{v}</td><td class="num">&ndash;</td>'
                       for v in [".911", ".864", ".359", ".835", ".885", ".746"])
             + '</tr>')
    return head + paper + "".join(body) + "</tbody></table>"



AKK = os.path.join(WM, "akkadian", "results")
ROWS_B = [
    ("llama2_70b", "Llama-2-70B", False), ("llama2_13b", "Llama-2-13B", False),
    ("llama2_7b", "Llama-2-7B", False), ("olmo2_7b", "OLMo-2-7B", False),
    ("gpt_oss_120b", "gpt-oss-120B", False),
    ("qwen3_32b", "Qwen3-32B", False), ("qwen3_8b", "Qwen3-8B", False),
    ("qwen3_1b7", "Qwen3-1.7B", False), ("umt5_base", "uMT5-base", False),
    ("thalesian_cunei400m", "cuneiform-400M", False),
    ("thalesian_akk300m", "AKK-300M", False),
    ("tfidf", "TF-IDF", True),
    ("llama2_70b_random", "Llama-2-70B random", True),
    ("llama2_13b_random", "Llama-2-13B random", True),
    ("llama2_7b_random", "Llama-2-7B random", True),
    ("olmo2_7b_random", "OLMo-2-7B random", True),
    ("random", "random Qwen3-8B", True),
]


def _rp(ridge, pls):
    """One cell: ridge value, then the PLS value in grey."""
    return f'<td class="num">{_fmt(ridge)}<i>|{_fmt(pls)}</i></td>'


def entity_table():
    """Cell B, entity level: rulers -> year and find-spots -> lat/lon, the paper's
    pooling only. The average-over-the-name read-out is reported in the takeaway,
    since it never separates from the controls."""
    import csv
    rows = {}
    with open(os.path.join(AKK, "summary_entity_best.csv")) as f:
        for r in csv.DictReader(f):
            rows[(r["arm"], r["entity_type"], r["site"], r["rows"])] = r

    def cell(arm, et):
        site = "text" if arm == "tfidf" else "ent_last"
        r = rows.get((arm, et, site, "bare"))
        if not r:
            return None, None
        return float(r["ridge_mc_rho"]), float(r["pls5_mc_rho"])

    head = ('<table class="rtbl compact dense"><thead>'
            '<tr><th>model</th>'
            '<th class="num">YEAR &nbsp;&middot;&nbsp; 34 rulers &rarr; year</th>'
            '<th class="num">PLACE &nbsp;&middot;&nbsp; 25 find-spots &rarr; latitude, longitude</th>'
            '</tr></thead><tbody>')
    body = []
    for arm, label, ctrl in ROWS_B:
        tr = '<tr class="rand">' if ctrl else '<tr>'
        body.append(tr + f'<td><span class="mdl">{label}</span></td>'
                    + _rp(*cell(arm, "assyrian_ruler"))
                    + _rp(*cell(arm, "mesopotamian_place")) + '</tr>')
    return head + "".join(body) + "</tbody></table>"


def frag_table(variant, target):
    """Cell B/C, fragment level: both poolings x {Spearman, R2}, balanced draws."""
    import json as _json

    def read(arm, site):
        if target == "year":
            p = os.path.join(AKK, "probes", arm, f"{variant}.r8.year.{site}.ridge.json")
            key = "mc"
        else:
            p = os.path.join(AKK, "probes_geosite", arm,
                             f"{variant}.{site}.geo_site.json")
            key = "mc_site"
        if not os.path.exists(p):
            return None, None
        m = _json.load(open(p))[key]
        return m.get("spearman_mean"), m.get("r2_mean")

    def tfidf_year():
        import csv
        with open(os.path.join(AKK, "summary_ALL_modes_full.csv")) as f:
            for r in csv.DictReader(f):
                if (r["arm"] == "tfidf" and r["variant"] == variant
                        and r["rulers"] == "r8" and r["target"] == "year"):
                    return float(r["mc_rho"]), float(r["mc_r2"])
        return None, None

    head = ('<table class="rtbl compact dense"><thead>'
            '<tr><th rowspan="2">model</th>'
            '<th colspan="3" class="num">Spearman &rho;</th>'
            '<th colspan="3" class="num">R&sup2;</th></tr>'
            '<tr><th class="num">last token</th><th class="num">average</th>'
            '<th class="num">difference</th>'
            '<th class="num">last token</th><th class="num">average</th>'
            '<th class="num">difference</th></tr></thead><tbody>')
    body = []
    for arm, label, ctrl in ROWS_B:
        if arm == "tfidf":
            if target == "year":
                rho, r2 = tfidf_year()
            else:
                rho, r2 = read("tfidf", "text")
            tr = '<tr class="rand">'
            body.append(
                tr + f'<td><span class="mdl">{label}</span></td>'
                f'<td class="num" colspan="2">{_fmt(rho)}</td>'
                '<td class="num">&ndash;</td>'
                f'<td class="num" colspan="2">{_fmt(r2)}</td>'
                '<td class="num">&ndash;</td></tr>')
            continue
        rl, r2l = read(arm, "last")
        rm, r2m = read(arm, "mean")

        def d(a, b):
            if a is None or b is None:
                return "&ndash;"
            v = b - a
            return f'{"+" if v > 0 else ""}{_fmt(v)}'
        tr = '<tr class="rand">' if ctrl else '<tr>'
        body.append(tr + f'<td><span class="mdl">{label}</span></td>'
                    f'<td class="num">{_fmt(rl)}</td><td class="num">{_fmt(rm)}</td>'
                    f'<td class="num">{d(rl, rm)}</td>'
                    f'<td class="num">{_fmt(r2l)}</td><td class="num">{_fmt(r2m)}</td>'
                    f'<td class="num">{d(r2l, r2m)}</td></tr>')
    return head + "".join(body) + "</tbody></table>"


CELL_ORDER = ["A", "B", "C"]


def cellmap(active):
    """Small 2x2 position marker: green = the cell this slide is in, pale green =
    cells already covered, dashed grey = D, which has no honest filler."""
    done = set(CELL_ORDER[:CELL_ORDER.index(active)]) if active in CELL_ORDER else set()

    def cls(c):
        if c == "D":
            return "cm-c na"
        if c == active:
            return "cm-c on"
        return "cm-c done" if c in done else "cm-c"
    return (
        '<div class="cellmap">'
        '<div></div><div class="cm-h">Eng</div><div class="cm-h">Akk</div>'
        f'<div class="cm-r">salient</div><div class="{cls("A")}">A</div>'
        f'<div class="{cls("D")}">&#10005;</div>'
        f'<div class="cm-r">obscure</div><div class="{cls("B")}">B</div>'
        f'<div class="{cls("C")}">C</div>'
        '</div>')


def inject_css(head):
    return head.replace("</style>", EXTRA_CSS, 1)


def figure_tag(name, alt=""):
    """Inline a figure as base64: plain names come from world_models/results/figs,
    names containing '/' resolve from the stress_tests directory (e.g. the
    embedding-panel PNGs)."""
    import base64
    if "/" in name:
        path = os.path.join(os.path.dirname(HERE), name)
    else:
        path = os.path.join(WM, "results", "figs", name)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}">'


def image_of(source_slides, idx, alt=None):
    """The <img> tag from a source slide, so a rewritten slide keeps its figure.
    The old alt text is replaced (it still carried the old slide's title)."""
    m = re.search(r'<img[^>]*>', source_slides[idx])
    if not m:
        raise SystemExit(f"slide {idx} has no <img> to reuse")
    tag = re.sub(r'\s*alt="[^"]*"', '', m.group(0))
    return tag.replace('<img', f'<img alt="{alt or "figure"}"', 1)


def parse_slides(html):
    out = {}
    for m in re.finditer(
            r'<section class="slide[^"]*" data-index="(\d+)">.*?</section>', html, re.S):
        out[int(m.group(1))] = m.group(0)
    return out


def strip_index(section):
    return re.sub(r'(<section class="slide[^"]*)" data-index="\d+"', r'\1"', section, count=1)


def set_eyebrow(section, text):
    if '<div class="eyebrow">' in section:
        return re.sub(r'<div class="eyebrow">.*?</div>',
                      f'<div class="eyebrow">{text}</div>', section, count=1, flags=re.S)
    # slides that never had one: insert after the opening tag
    return re.sub(r'(<section class="slide[^"]*">\n)',
                  rf'\1  <div class="eyebrow">{text}</div>\n', section, count=1)


def build(html):
    old = parse_slides(html)
    missing = [ref for kind, ref, _ in SPINE
               if (kind == "old" and ref not in old)
               or (kind == "new" and ref not in NEW_SLIDES)]
    if missing:
        raise SystemExit(f"spine references unknown slides: {missing}")

    bodies, titles = [], []
    for i, (kind, ref, title) in enumerate(SPINE):
        sec = strip_index(old[ref]) if kind == "old" else NEW_SLIDES[ref]
        for fig in set(re.findall(r'\{\{FIG:([^}]+)\}\}', sec)):
            sec = sec.replace("{{FIG:%s}}" % fig, figure_tag(fig, alt=title))
        if "{{TABLE:cellA}}" in sec:
            sec = sec.replace("{{TABLE:cellA}}", cellA_table())
        if "{{TABLE:entity}}" in sec:
            sec = sec.replace("{{TABLE:entity}}", entity_table())
        for m in set(re.findall(r'\{\{TABLE:frag:([a-z0-9_]+):([a-z]+)\}\}', sec)):
            sec = sec.replace("{{TABLE:frag:%s:%s}}" % m, frag_table(*m))
        for tok in set(re.findall(r'\{\{IMG:(\d+)\}\}', sec)):
            sec = sec.replace("{{IMG:%s}}" % tok, image_of(old, int(tok), alt=title))
        if kind == "old" and ref in EYEBROW_PATCHES:
            sec = set_eyebrow(sec, EYEBROW_PATCHES[ref])
        if kind == "old" and ref in SLIDE_TRANSFORMS:
            sec = SLIDE_TRANSFORMS[ref](sec)
        sec = re.sub(r'(<section class="slide[^"]*)"',
                     rf'\1" data-index="{i}"', sec, count=1)
        if i in CELLMAP_AT:
            sec = re.sub(r'(<section[^>]*>)', r'\1\n  ' + cellmap(CELLMAP_AT[i]),
                         sec, count=1)
        bodies.append(sec)
        titles.append(title)

    head = inject_css(html[:html.index('<section')])
    tail = html[html.rindex('</section>') + len('</section>'):]
    tail = re.sub(r'const TOTAL = \d+;', f'const TOTAL = {len(SPINE)};', tail)
    titles_js = ", ".join(
        '"' + t.replace("\\", "\\\\").replace('"', '\\"') + '"' for t in titles)
    tail = re.sub(r'const TITLES = \[.*?\];', f'const TITLES = [{titles_js}];',
                  tail, flags=re.S)
    return head + "\n".join(bodies) + tail


def check(html):
    n_sec = html.count("<section")
    idx = [int(x) for x in re.findall(r'data-index="(\d+)"', html)]
    total = int(re.search(r'const TOTAL = (\d+)', html).group(1))
    titles = re.search(r'const TITLES = \[(.*?)\];', html, re.S).group(1)
    n_titles = len(re.findall(r'"(?:[^"\\]|\\.)*"', titles))
    ok = (n_sec == total == n_titles == len(idx)) and idx == list(range(len(idx)))
    print(f"sections={n_sec} indices={'contiguous 0..%d' % (len(idx)-1) if idx == list(range(len(idx))) else idx} "
          f"TOTAL={total} titles={n_titles}  ->  {'OK' if ok else 'MISMATCH'}")
    return ok


def read_source(path=None):
    """The 33-slide deck the spine refers to: a file if given, else the pinned blob."""
    if path:
        with open(path) as f:
            return f.read()
    return subprocess.run(["git", "cat-file", "blob", SOURCE_BLOB],
                          cwd=HERE, capture_output=True, text=True,
                          check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--source", default=None,
                    help="33-slide source deck (default: the pinned git blob)")
    ap.add_argument("--check", action="store_true", help="only verify the current deck")
    args = ap.parse_args()

    if args.check:
        with open(DECK) as f:
            raise SystemExit(0 if check(f.read()) else 1)

    new = build(read_source(args.source))
    with open(args.out, "w") as f:
        f.write(new)
    print(f"[write] {args.out}  ({len(new)/1e6:.2f} MB, {len(SPINE)} slides)")
    check(new)


if __name__ == "__main__":
    main()
