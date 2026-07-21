"""make_interactive_viewer.py — a fully self-contained INTERACTIVE embedding
viewer (like the seal_eda GUI) for the stress-test maps. Points are drawn live
on a canvas, so browser zoom / wheel-zoom / pan keep everything crisp and the
legend is real HTML text (no blurry baked-in labels).

Controls: Model dropdown + Cleaning / Reduction / Color-by selectors + the
per-cleaning year-probe leaderboard. Wheel = zoom, drag = pan, hover = tooltip
with that fragment's metadata. Continuous color-bys (year, length) get a
gradient colorbar; categorical ones (ruler, period, sub-genre, provenance) get
a top-8 legend.

Coordinates come from the already-computed viz JSONs (stress_coords = tsne/pca,
stress_umap_coords = umap, pls3d_coords = pls first-2-components); metadata from
the ORCC corpus parquet. Everything is embedded -> one portable file.

Output: e6_clusters/embedding_panels/interactive.html
Usage:  python v_1/src/stress_tests/eda/make_interactive_viewer.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ST = Path(__file__).resolve().parents[1]
VIZ = ST.parents[1] / "src" / "viz"
ROOT = ST / "e6_clusters" / "embedding_panels"
CSVD = ST / "results" / "csv"
CORPUS = ST.parents[1] / "data" / "evaluation" / "corpora" / "orcc_corpus.parquet"

MODEL_LABEL = {
    "thalesian_cunei400m": "cunei-400m", "thalesian_akk300m": "Thal-AKK-300m",
    "qwen3_32b": "Qwen3-32B", "qwen3_8b": "Qwen3-8B", "qwen3_1b7": "Qwen3-1.7B",
    "gpt_oss_120b": "gpt-oss-120B", "umt5_base": "uMT5-base", "mlm": "MLM",
    "tfidf": "TF-IDF*", "random": "random*"}
MODEL_ORDER = list(MODEL_LABEL)


def _f(v):
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


def best_layer_pick(embkeys, cl, model, proj):
    cands = [k for k in embkeys if k.startswith(f"{cl}__{model}__") and k.endswith(f"__{proj}")]
    return sorted(cands, key=lambda k: k.split("__")[2])[-1] if cands else None


df = pd.read_parquet(CORPUS)
FIDS = df["fragment_id"].astype(str).tolist()
n = len(df)


def norm_xy(arr):
    """arr: list of [x,y] or None -> normalized to [0,1]^2 with None -> null."""
    a = np.array([[np.nan, np.nan] if (v is None or v[0] is None) else v[:2] for v in arr],
                 dtype=float)
    m = np.isfinite(a).all(1)
    if m.sum() < 2:
        return [None] * len(a)
    lo = np.nanmin(a[m], 0); hi = np.nanmax(a[m], 0); rng = np.where(hi > lo, hi - lo, 1)
    out = []
    for i in range(len(a)):
        if m[i]:
            xy = (a[i] - lo) / rng
            out.append([round(float(xy[0]), 4), round(float(xy[1]), 4)])
        else:
            out.append(None)
    return out


# ---- coordinates: {cleaning: {reduction: {model: [[x,y]|null ...]}}} ----
sc = json.loads((VIZ / "stress_coords.json").read_text())
um = json.loads((VIZ / "stress_umap_coords.json").read_text())
pl = json.loads((VIZ / "pls3d_coords.json").read_text())
sources = [(sc, ["tsne", "pca"]), (um, ["umap"]), (pl, ["pls3d"])]

DATA = {"maximal": {}, "engtier0": {}}
for src, projs in sources:
    keys = src["embeddings"]
    align = src["fragment_ids"] == FIDS
    for cl in DATA:
        for proj in projs:
            out_proj = "pls" if proj == "pls3d" else proj
            models = sorted({k.split("__")[1] for k in keys
                             if k.startswith(f"{cl}__") and k.endswith(f"__{proj}")},
                            key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
            for m in models:
                key = best_layer_pick(keys, cl, m, proj)
                if not key:
                    continue
                vec = keys[key] if align else None
                if vec is None:
                    continue
                DATA[cl].setdefault(out_proj, {})[m] = norm_xy(vec)

# tfidf PLS computed here (its own supervised projection), both cleanings
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cross_decomposition import PLSRegression
tr = pd.read_parquet(ST / "translation" / "translations.parquet").set_index("fragment_id")
year = df["year"].to_numpy(dtype=float); ok = np.isfinite(year)
for cl, texts in [("maximal", df["text_maximal"].fillna("").astype(str).tolist()),
                  ("engtier0", tr["eng_tier0"].reindex(FIDS).fillna("").astype(str).tolist())]:
    X = TruncatedSVD(512, random_state=0).fit_transform(
        normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).fit_transform(texts)))
    Z = np.full((n, 2), np.nan)
    Z[ok] = PLSRegression(3).fit(X[ok], year[ok]).transform(X[ok])[:, :2]
    DATA[cl].setdefault("pls", {})["tfidf"] = norm_xy([None if not np.isfinite(r).all() else list(r) for r in Z])

# ---- per-fragment metadata (aligned to FIDS) ----
def top_codes(series, topn=8):
    top = series.value_counts().head(topn).index.tolist()
    code = {v: i for i, v in enumerate(top)}
    labels = top + ["other"]
    idx = [code.get(v, topn) for v in series.astype(str)]
    return labels, idx


META = {"year": [None if not np.isfinite(y) else int(y) for y in year],
        "length": [round(float(np.log10(max(1, w))), 2) for w in df["word_count"].to_numpy(dtype=float)]}
CATS = {}
for col in ["ruler", "period", "sub_genre", "provenance"]:
    labels, idx = top_codes(df[col].astype(str))
    CATS[col] = {"labels": labels, "idx": idx}

# ---- year-probe leaderboard (same as the static viewer) ----
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

payload = {"DATA": DATA, "META": META, "CATS": CATS, "SCORES": scores,
           "MLAB": MODEL_LABEL,
           "RLAB": {"tsne": "t-SNE", "pca": "PCA", "umap": "UMAP", "pls": "PLS (supervised)"},
           "CLAB": {"maximal": "Akkadian maximal", "engtier0": "English tier0"},
           "CONTROLS": ["tfidf", "random"]}
PAYLOAD = json.dumps(payload, separators=(",", ":"))

HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Interactive embedding viewer</title>
<style>
 :root{--g:#0a7}
 *{box-sizing:border-box}
 html,body{margin:0;height:100%;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1d1d1f;background:#f5f5f7}
 header{background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;display:flex;gap:26px;flex-wrap:wrap;align-items:flex-end;justify-content:space-between}
 h1{margin:0 0 8px;font-size:16px}
 .bar{display:flex;gap:20px;flex-wrap:wrap;align-items:flex-end}
 .ctl{display:flex;flex-direction:column;gap:4px}
 .ctl label{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:#888;font-weight:700}
 select{font-size:14px;padding:7px 9px;border:1px solid #ccc;border-radius:7px;background:#fff}
 .seg{display:flex;border:1px solid #ccc;border-radius:7px;overflow:hidden}
 .seg button{border:0;background:#fff;padding:7px 12px;font-size:13px;cursor:pointer;color:#333}
 .seg button.on{background:var(--g);color:#fff}
 .seg button+button{border-left:1px solid #ccc}
 .board{min-width:230px;border:1px solid #e2e2e2;border-radius:9px;background:#fafafa;padding:6px 9px;font-size:12px}
 .board h3{margin:0 0 4px;font-size:11px;text-transform:uppercase;color:#666}
 .board table{border-collapse:collapse;width:100%}
 .board td{padding:1px 5px;white-space:nowrap}.board td.r{text-align:right;font-variant-numeric:tabular-nums}
 .board tr.cur{background:var(--g);color:#fff}.board tr.cur td{color:#fff}
 .board tr.ctrl td:first-child{color:#a15}.board tr{cursor:pointer}
 main{display:flex;height:calc(100vh - 88px)}
 #wrap{flex:1;position:relative}
 canvas{width:100%;height:100%;display:block;cursor:grab;background:#fff}
 canvas.drag{cursor:grabbing}
 #side{width:230px;border-left:1px solid #ddd;background:#fff;padding:12px 14px;overflow:auto}
 #cap{font-size:12.5px;color:#333;font-weight:600;margin:0 0 10px}
 #cap .rho{color:var(--g)}
 .lg{font-size:12.5px}
 .lg .row{display:flex;align-items:center;gap:7px;margin:3px 0}
 .lg .sw{width:12px;height:12px;border-radius:3px;flex:none}
 .cbar{height:16px;border-radius:4px;margin:6px 0 3px;background:linear-gradient(90deg,var(--c0),var(--c1),var(--c2))}
 .cbl{display:flex;justify-content:space-between;font-size:11px;color:#666}
 #tip{position:absolute;pointer-events:none;background:#111;color:#fff;font-size:11.5px;padding:6px 8px;border-radius:6px;display:none;max-width:240px;line-height:1.35;z-index:5}
 .hint{font-size:11px;color:#999;margin-top:14px}
 .reset{margin-top:10px;font-size:12px;padding:5px 9px;border:1px solid #ccc;border-radius:6px;background:#fff;cursor:pointer}
</style></head><body>
<header>
 <div style="flex:1;min-width:420px">
  <h1>Interactive embedding viewer &mdash; wheel to zoom, drag to pan, hover a point for its fragment</h1>
  <div class="bar">
   <div class="ctl"><label>Model</label><select id="model"></select></div>
   <div class="ctl"><label>Cleaning</label><div class="seg" id="cleaning"></div></div>
   <div class="ctl"><label>Reduction</label><div class="seg" id="reduction"></div></div>
   <div class="ctl"><label>Color by</label><div class="seg" id="colorby"></div></div>
  </div>
 </div>
 <div class="board"><h3 id="btitle">Year-probe &rho;</h3><table><tbody id="brows"></tbody></table></div>
</header>
<main>
 <div id="wrap"><canvas id="cv"></canvas><div id="tip"></div></div>
 <div id="side">
  <p id="cap"></p>
  <div class="lg" id="legend"></div>
  <button class="reset" id="reset">Reset view</button>
  <p class="hint">Points are drawn live &mdash; browser zoom and wheel-zoom stay crisp, and the legend is real text. &rho; = full-dim year-probe (same across reductions).</p>
 </div>
</main>
<script>
const P=__PAYLOAD__;
const {DATA,META,CATS,SCORES,MLAB,RLAB,CLAB}=P, CONTROLS=new Set(P.CONTROLS);
const CONT={year:{lab:"Year BCE",stops:["#0d0887","#cc4778","#f0f921"],rev:true,
                  vals:META.year, fmt:v=>v},
            length:{lab:"Length (log10 words)",stops:["#440154","#21918c","#fde725"],rev:false,
                  vals:META.length, fmt:v=>v}};
const TAB=["#4e79a7","#f28e2b","#59a14f","#e15759","#b07aa1","#9c755f","#edc948","#76b7b2","#bab0ac"];
const state={cleaning:"maximal",reduction:"tsne",model:null,colorby:"year"};
const view={s:1,ox:0,oy:0};

const rho=(cl,m)=>(SCORES[cl]&&SCORES[cl][m]!=null)?SCORES[cl][m]:null;
const modelsFor=(cl,red)=>Object.keys((DATA[cl]&&DATA[cl][red])||{});
const coords=()=> (DATA[state.cleaning]&&DATA[state.cleaning][state.reduction]&&
                   DATA[state.cleaning][state.reduction][state.model])||null;

function seg(id,keys,lab,cur){const b=document.getElementById(id);b.innerHTML="";
  keys.forEach(k=>{const t=document.createElement("button");t.textContent=lab[k]||k;t.dataset.k=k;
    if(k===cur)t.classList.add("on");t.onclick=()=>{state[id]=k;resetView();sync();};b.appendChild(t);});}
function syncSeg(id,cur){document.querySelectorAll("#"+id+" button").forEach(b=>b.classList.toggle("on",b.dataset.k===cur));}

function refreshModels(){const sel=document.getElementById("model");
  const list=modelsFor(state.cleaning,state.reduction).sort((a,b)=>{
    const oa=Object.keys(MLAB).indexOf(a),ob=Object.keys(MLAB).indexOf(b);return oa-ob;});
  const prev=state.model;sel.innerHTML="";
  list.forEach(m=>{const o=document.createElement("option");o.value=m;const s=rho(state.cleaning,m);
    o.textContent=(MLAB[m]||m)+(s!=null?"  (ρ "+s.toFixed(3)+")":"");sel.appendChild(o);});
  state.model=list.includes(prev)?prev:(list[0]||null);sel.value=state.model||"";}

function board(){const cl=state.cleaning,tb=document.getElementById("brows");
  document.getElementById("btitle").innerHTML="Year-probe &rho; &mdash; "+CLAB[cl];
  const rk=Object.keys(SCORES[cl]).filter(m=>rho(cl,m)!=null).sort((a,b)=>rho(cl,b)-rho(cl,a));
  tb.innerHTML="";rk.forEach((m,i)=>{const tr=document.createElement("tr");
    if(m===state.model)tr.className="cur";else if(CONTROLS.has(m))tr.className="ctrl";
    tr.innerHTML="<td>"+(i+1)+". "+(MLAB[m]||m)+"</td><td class='r'>"+rho(cl,m).toFixed(3)+"</td>";
    tr.onclick=()=>{if(modelsFor(cl,state.reduction).includes(m)){state.model=m;sync();}};tb.appendChild(tr);});}

// color mapping
function lerp(a,b,t){return a+(b-a)*t;}
function hex(c){return[parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)];}
function ramp(stops,t){const c=[hex(stops[0]),hex(stops[1]),hex(stops[2])];
  t=Math.max(0,Math.min(1,t));const seg=t<.5?0:1,tt=t<.5?t*2:(t-.5)*2;
  const a=c[seg],b=c[seg+1];return"rgb("+Math.round(lerp(a[0],b[0],tt))+","+Math.round(lerp(a[1],b[1],tt))+","+Math.round(lerp(a[2],b[2],tt))+")";}
function colorFor(i){const cb=state.colorby;
  if(CONT[cb]){const V=CONT[cb].vals,v=V[i];if(v==null)return null;
    let t=(v-CONT[cb].lo)/(CONT[cb].hi-CONT[cb].lo||1);if(CONT[cb].rev)t=1-t;return ramp(CONT[cb].stops,t);}
  const idx=CATS[cb].idx[i];return idx>=8?"#d8d8d8":TAB[idx];}

function computeRange(){for(const k in CONT){const V=CONT[k].vals.filter(v=>v!=null);
  CONT[k].lo=Math.min(...V);CONT[k].hi=Math.max(...V);}}

// canvas
const cv=document.getElementById("cv"),ctx=cv.getContext("2d"),wrap=document.getElementById("wrap"),tip=document.getElementById("tip");
function fit(){const r=wrap.getBoundingClientRect(),dpr=devicePixelRatio||1;
  cv.width=r.width*dpr;cv.height=r.height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
function px(xy){const r=wrap.getBoundingClientRect(),pad=30,W=r.width-2*pad,H=r.height-2*pad;
  return[pad+(view.ox+xy[0]*view.s)*W, pad+(1-(view.oy+xy[1]*view.s))*H];}
function draw(){const C=coords();const r=wrap.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);
  if(!C){ctx.fillStyle="#a00";ctx.font="15px sans-serif";ctx.fillText("no coords for this combination",30,40);return;}
  // draw greyed 'other' first for categoricals so colored points sit on top
  const cb=state.colorby,catGrey=!CONT[cb];
  for(let pass=0;pass<2;pass++){for(let i=0;i<C.length;i++){const xy=C[i];if(!xy)continue;
    const col=colorFor(i);if(col==null)continue;const isGrey=(col==="#d8d8d8");
    if(catGrey&&((pass===0)!==isGrey))continue; if(!catGrey&&pass===1)continue;
    const p=px(xy);ctx.fillStyle=col;ctx.globalAlpha=isGrey?0.5:0.85;
    ctx.beginPath();ctx.arc(p[0],p[1],2.6,0,6.283);ctx.fill();}}
  ctx.globalAlpha=1;}
function resetView(){view.s=1;view.ox=0;view.oy=0;}
document.getElementById("reset").onclick=()=>{resetView();draw();};

// zoom / pan
cv.addEventListener("wheel",e=>{e.preventDefault();const r=wrap.getBoundingClientRect(),pad=30;
  const mx=(e.clientX-r.left-pad)/(r.width-2*pad), my=1-(e.clientY-r.top-pad)/(r.height-2*pad);
  const f=Math.exp(-e.deltaY*0.0015);const ns=Math.max(0.4,Math.min(40,view.s*f));
  view.ox=mx-(mx-view.ox)*(ns/view.s);view.oy=my-(my-view.oy)*(ns/view.s);view.s=ns;draw();},{passive:false});
let drag=null;
cv.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,ox:view.ox,oy:view.oy};cv.classList.add("drag");});
addEventListener("mouseup",()=>{drag=null;cv.classList.remove("drag");});
addEventListener("mousemove",e=>{if(!drag){hover(e);return;}const r=wrap.getBoundingClientRect(),pad=30;
  view.ox=drag.ox+(e.clientX-drag.x)/(r.width-2*pad);view.oy=drag.oy-(e.clientY-drag.y)/(r.height-2*pad);draw();});
function hover(e){const C=coords();if(!C){tip.style.display="none";return;}
  const r=wrap.getBoundingClientRect();let best=-1,bd=1e9;
  for(let i=0;i<C.length;i++){const xy=C[i];if(!xy)continue;const p=px(xy);
    const d=(p[0]-(e.clientX-r.left))**2+(p[1]-(e.clientY-r.top))**2;if(d<bd){bd=d;best=i;}}
  if(best<0||bd>90){tip.style.display="none";return;}
  const yr=META.year[best],lab=x=>CATS[x].labels[CATS[x].idx[best]];
  tip.innerHTML="<b>#"+best+"</b> · year "+(yr==null?"?":yr+" BCE")+
    "<br>ruler: "+lab("ruler")+"<br>period: "+lab("period")+
    "<br>site: "+lab("provenance")+"<br>type: "+lab("sub_genre");
  tip.style.display="block";tip.style.left=(e.clientX-r.left+12)+"px";tip.style.top=(e.clientY-r.top+12)+"px";}

function legend(){const el=document.getElementById("legend"),cb=state.colorby;let h="";
  if(CONT[cb]){const lo=CONT[cb].lo,hi=CONT[cb].hi,s=CONT[cb].stops;
    h='<div style="font-weight:600;margin-bottom:2px">'+CONT[cb].lab+'</div>'+
      '<div class="cbar" style="--c0:'+s[0]+';--c1:'+s[1]+';--c2:'+s[2]+'"></div>'+
      '<div class="cbl"><span>'+(CONT[cb].rev?hi:lo)+'</span><span>'+(CONT[cb].rev?lo:hi)+'</span></div>'+
      (CONT[cb].rev?'<div style="font-size:10.5px;color:#999">(older &rarr; left)</div>':'');
  }else{const L=CATS[cb].labels;h='<div style="font-weight:600;margin-bottom:4px">'+cb+'</div>';
    L.forEach((name,i)=>{const c=i>=8?"#d8d8d8":TAB[i];
      const cnt=CATS[cb].idx.filter(x=>x===i).length;
      h+='<div class="row"><span class="sw" style="background:'+c+'"></span>'+name+' <span style="color:#999">('+cnt+')</span></div>';});}
  el.innerHTML=h;}

function cap(){const cl=state.cleaning,m=state.model,s=rho(cl,m);
  const rk=Object.keys(SCORES[cl]).filter(x=>rho(cl,x)!=null).sort((a,b)=>rho(cl,b)-rho(cl,a));
  let t=(MLAB[m]||m)+" · "+CLAB[cl]+" · "+RLAB[state.reduction];
  if(s!=null)t+=" · <span class='rho'>year-ρ "+s.toFixed(3)+"</span> (rank "+(rk.indexOf(m)+1)+"/"+rk.length+")";
  document.getElementById("cap").innerHTML=t;}

function sync(){syncSeg("cleaning",state.cleaning);syncSeg("reduction",state.reduction);syncSeg("colorby",state.colorby);
  refreshModels();board();legend();cap();draw();}
document.getElementById("model").onchange=e=>{state.model=e.target.value;board();cap();draw();};
addEventListener("resize",fit);

computeRange();
seg("cleaning",["maximal","engtier0"],CLAB,state.cleaning);
seg("reduction",["tsne","pca","umap","pls"],RLAB,state.reduction);
seg("colorby",["year","ruler","period","sub_genre","provenance","length"],
    {year:"Year",ruler:"Ruler",period:"Period",sub_genre:"Sub-genre",provenance:"Provenance",length:"Length"},state.colorby);
refreshModels();fit();sync();
</script></body></html>"""

out = ROOT / "interactive.html"
out.write_text(HTML.replace("__PAYLOAD__", PAYLOAD), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, self-contained interactive)")
