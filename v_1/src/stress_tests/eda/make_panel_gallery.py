"""make_panel_gallery.py — self-contained index.html viewer for the embedding
panels, in the seal_eda GUI style: MODEL dropdown + CLEANING / REDUCTION
selectors, one large six-view panel below, the model's year-probe Spearman in
the caption, and a live per-cleaning leaderboard (top-right) so you can see
where the current model ranks. No dependencies: open index.html in a browser.

The Spearman shown is the full-dim ACTIVATION year probe (PLS best-k; Ridge for
TF-IDF where PLS~0) from results/csv/ — a property of the model x cleaning, the
SAME across all four 2-D reductions (the maps only visualize that space).

Usage:  python v_1/src/stress_tests/eda/make_panel_gallery.py
"""
import csv
import json
from pathlib import Path

ST = Path(__file__).resolve().parents[1]
ROOT = ST / "e6_clusters" / "embedding_panels"
CSVD = ST / "results" / "csv"
CLEANINGS = ["maximal", "engtier0"]
REDUCTIONS = ["tsne", "pca", "umap", "pls"]
CONTROLS = {"tfidf", "random"}
MODEL_LABEL = {
    "thalesian_cunei400m": "cunei-400m", "thalesian_akk300m": "Thal-AKK-300m",
    "qwen3_32b": "Qwen3-32B", "qwen3_8b": "Qwen3-8B", "qwen3_1b7": "Qwen3-1.7B",
    "gpt_oss_120b": "gpt-oss-120B", "umt5_base": "uMT5-base", "mlm": "MLM",
    "tfidf": "TF-IDF*", "random": "random*"}
MODEL_ORDER = ["thalesian_cunei400m", "qwen3_32b", "qwen3_8b", "qwen3_1b7",
               "gpt_oss_120b", "thalesian_akk300m", "umt5_base", "mlm",
               "tfidf", "random"]


def _f(v):
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


# --- year-probe Spearman per model x cleaning (same metric family as deck Table 1)
scores = {"maximal": {}, "engtier0": {}}
for r in csv.DictReader(open(CSVD / "p1_year_mc.csv")):
    if r["site"] == "mean_maximal":
        scores["maximal"][r["model"]] = _f(r["pls_spearman_mean"])
for r in csv.DictReader(open(CSVD / "translation_mc.csv")):
    if r["cleaning"] == "engtier0":
        scores["engtier0"][r["model"]] = _f(r["year_pls_spearman"])
for r in csv.DictReader(open(CSVD / "tfidf_baseline.csv")):
    if r["cleaning"] in scores:
        scores[r["cleaning"]]["tfidf"] = _f(r["year_ridge_spearman"])

# --- available[cleaning][reduction] = model stems that have a png
avail = {}
for cl in CLEANINGS:
    for red in REDUCTIONS:
        d = ROOT / cl / red
        if not d.is_dir():
            continue
        models = [p.stem for p in d.glob("*.png")]
        avail.setdefault(cl, {})[red] = sorted(
            models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)

DATA = json.dumps(avail)
LABELS = json.dumps(MODEL_LABEL)
SCORES = json.dumps(scores)
CONTROLS_JS = json.dumps(sorted(CONTROLS))
RLAB = json.dumps({"tsne": "t-SNE", "pca": "PCA", "umap": "UMAP", "pls": "PLS (supervised)"})
CLAB = json.dumps({"maximal": "Akkadian maximal", "engtier0": "English tier0"})

html = """<!doctype html><html><head><meta charset="utf-8">
<title>Embedding panel viewer</title>
<style>
 :root{--g:#0a7}
 *{box-sizing:border-box}
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:14px 22px;z-index:9;
   display:flex;justify-content:space-between;gap:24px;align-items:flex-start;flex-wrap:wrap}
 .hl h1{margin:0 0 3px;font-size:17px}
 .hl .sub{font-size:12.5px;color:#666;margin-bottom:12px;max-width:70ch}
 .bar{display:flex;flex-wrap:wrap;gap:22px;align-items:flex-end}
 .ctl{display:flex;flex-direction:column;gap:4px}
 .ctl label{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#888;font-weight:600}
 select{font-size:14px;padding:7px 10px;border:1px solid #ccc;border-radius:7px;background:#fff;min-width:190px}
 .seg{display:flex;border:1px solid #ccc;border-radius:7px;overflow:hidden}
 .seg button{border:0;background:#fff;padding:8px 14px;font-size:13.5px;cursor:pointer;color:#333}
 .seg button.on{background:var(--g);color:#fff}
 .seg button+button{border-left:1px solid #ccc}
 .board{min-width:250px;border:1px solid #e2e2e2;border-radius:9px;background:#fafafa;padding:8px 10px}
 .board h3{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#666}
 .board table{border-collapse:collapse;width:100%;font-size:12.5px}
 .board td{padding:2px 6px;white-space:nowrap}
 .board td.r{text-align:right;font-variant-numeric:tabular-nums;color:#333}
 .board tr.cur{background:var(--g);color:#fff;border-radius:4px}
 .board tr.cur td{color:#fff}
 .board tr.ctrl td:first-child{color:#a15}
 .board tr{cursor:pointer}
 main{padding:18px 22px 40px;text-align:center}
 #cap{font-size:14px;color:#444;margin:0 0 10px;font-weight:600}
 #cap .rho{color:var(--g)}
 #panel{max-width:100%;max-height:80vh;border:1px solid #e0e0e0;border-radius:10px;background:#fff}
 #miss{display:none;color:#a00;padding:40px;font-size:15px}
 .hint{font-size:12px;color:#888;margin-top:10px}
</style></head><body>
<header>
 <div class="hl">
  <h1>Embedding panel viewer &mdash; six views (year / ruler / period / sub-genre / provenance / length)</h1>
  <div class="sub">ORCC royal inscriptions. &rho; = full-dim activation year-probe Spearman (PLS; Ridge for TF-IDF) &mdash; a property of the model &times; cleaning, the SAME across all four 2-D reductions; the maps just visualize that space.</div>
  <div class="bar">
   <div class="ctl"><label>Model</label><select id="model"></select></div>
   <div class="ctl"><label>Cleaning</label><div class="seg" id="cleaning"></div></div>
   <div class="ctl"><label>Dim reduction</label><div class="seg" id="reduction"></div></div>
  </div>
 </div>
 <div class="board"><h3 id="btitle">Year-probe &rho; leaderboard</h3>
  <table><tbody id="brows"></tbody></table></div>
</header>
<main>
 <p id="cap"></p>
 <img id="panel" alt="panel">
 <div id="miss"></div>
 <p class="hint">Click a leaderboard row to jump to that model. Missing combos (MLM = Akkadian only; some controls lack a projection) are handled automatically.</p>
</main>
<script>
 const AVAIL=__DATA__, MLAB=__LABELS__, SCORES=__SCORES__, CONTROLS=new Set(__CONTROLS__),
       RLAB=__RLAB__, CLAB=__CLAB__;
 const state={cleaning:"maximal", reduction:"tsne", model:null};
 const rho=(cl,m)=> (SCORES[cl]&&SCORES[cl][m]!=null)?SCORES[cl][m]:null;

 function modelsFor(cl,red){return (AVAIL[cl]&&AVAIL[cl][red])||[];}
 function ranked(cl){
   return Object.keys(SCORES[cl]).filter(m=>rho(cl,m)!=null)
          .sort((a,b)=>rho(cl,b)-rho(cl,a));
 }
 function buildSeg(id, keys, labMap, cur){
   const box=document.getElementById(id); box.innerHTML="";
   keys.forEach(k=>{const b=document.createElement("button");
     b.textContent=labMap[k]||k; b.dataset.k=k;
     if(k===cur) b.classList.add("on");
     b.onclick=()=>{state[id]=k; sync();}; box.appendChild(b);});
 }
 function syncSeg(id,cur){document.querySelectorAll("#"+id+" button").forEach(b=>
   b.classList.toggle("on", b.dataset.k===cur));}

 function refreshModels(){
   const sel=document.getElementById("model");
   const list=modelsFor(state.cleaning,state.reduction);
   const prev=state.model; sel.innerHTML="";
   list.forEach(m=>{const o=document.createElement("option");
     o.value=m; const s=rho(state.cleaning,m);
     o.textContent=(MLAB[m]||m)+(s!=null?"  (ρ "+s.toFixed(3)+")":""); sel.appendChild(o);});
   state.model = list.includes(prev)?prev:(list[0]||null);
   sel.value=state.model||"";
 }
 function board(){
   const cl=state.cleaning, rk=ranked(cl), tb=document.getElementById("brows");
   document.getElementById("btitle").innerHTML="Year-probe &rho; &mdash; "+CLAB[cl];
   tb.innerHTML="";
   rk.forEach((m,i)=>{const tr=document.createElement("tr");
     if(m===state.model) tr.className="cur";
     else if(CONTROLS.has(m)) tr.className="ctrl";
     tr.innerHTML="<td>"+(i+1)+". "+(MLAB[m]||m)+"</td><td class='r'>"+rho(cl,m).toFixed(3)+"</td>";
     tr.onclick=()=>{ if(modelsFor(cl,state.reduction).includes(m)){state.model=m; sync();} };
     tb.appendChild(tr);});
 }
 function show(){
   const img=document.getElementById("panel"), miss=document.getElementById("miss"),
         cap=document.getElementById("cap");
   const list=modelsFor(state.cleaning,state.reduction), cl=state.cleaning, m=state.model;
   if(!m || !list.includes(m)){
     img.style.display="none"; miss.style.display="block";
     miss.textContent="No "+RLAB[state.reduction]+" panel for "+(MLAB[m]||m)+" · "+CLAB[cl];
     cap.textContent=""; return;
   }
   miss.style.display="none"; img.style.display="inline-block";
   img.src=cl+"/"+state.reduction+"/"+m+".png";
   const s=rho(cl,m), rk=ranked(cl), pos=rk.indexOf(m)+1;
   let tail="";
   if(s!=null) tail=" · <span class='rho'>year-probe ρ = "+s.toFixed(3)+
                     "</span> (rank "+pos+"/"+rk.length+" on "+CLAB[cl]+")";
   cap.innerHTML=(MLAB[m]||m)+" · "+CLAB[cl]+" · "+RLAB[state.reduction]+tail;
 }
 function sync(){
   syncSeg("cleaning",state.cleaning); syncSeg("reduction",state.reduction);
   refreshModels(); board(); show();
 }
 document.getElementById("model").onchange=e=>{state.model=e.target.value; board(); show();};
 buildSeg("cleaning",["maximal","engtier0"],CLAB,state.cleaning);
 buildSeg("reduction",["tsne","pca","umap","pls"],RLAB,state.reduction);
 sync();
</script>
</body></html>"""

html = (html.replace("__DATA__", DATA).replace("__LABELS__", LABELS)
        .replace("__SCORES__", SCORES).replace("__CONTROLS__", CONTROLS_JS)
        .replace("__RLAB__", RLAB).replace("__CLAB__", CLAB))
out = ROOT / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}")

# --- also emit a fully self-contained standalone.html (all PNGs base64-embedded) ---
import base64
emb = {}
for p in sorted(ROOT.rglob("*.png")):
    emb[p.relative_to(ROOT).as_posix()] = ("data:image/png;base64,"
                                            + base64.b64encode(p.read_bytes()).decode())
sa = html.replace("<script>\n", "<script>\nconst EMB=" + json.dumps(emb) + ";\n", 1)
sa = sa.replace('img.src=cl+"/"+state.reduction+"/"+m+".png";',
                'img.src=EMB[cl+"/"+state.reduction+"/"+m+".png"]||"";')
sa = sa.replace("<title>Embedding panel viewer</title>",
                "<title>Embedding panel viewer (standalone)</title>")
sap = ROOT / "standalone.html"
sap.write_text(sa, encoding="utf-8")
print(f"wrote {sap}  ({sap.stat().st_size/1e6:.1f} MB, {len(emb)} images embedded, fully portable)")
