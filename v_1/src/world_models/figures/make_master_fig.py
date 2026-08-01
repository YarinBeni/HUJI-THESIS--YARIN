import os, pandas as pd, numpy as np, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
from _style import COL, LAB
# Resolution/format policy lives in figures/lib/_save.py (300 dpi PNG + vector PDF)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'lib'))
from _save import save as _save_fig  # noqa: E402


ARMS=["llama2_70b","gpt_oss_120b","llama2_13b","qwen3_32b","llama2_7b","qwen3_8b","qwen3_1b7",
      "thalesian_cunei400m","thalesian_akk300m","umt5_base",
      "llama2_70b_random","llama2_13b_random","llama2_7b_random"]

# (cell, level, salience, cleaning, pooling, row-label, tokens, highlight-idx, note)
ROWS=[
 ("A","entity","salient","historical_figure","last",
  "A · salient entity · EN\nlast token of the name",
  ["George","Washington"],[1],"paper's entity string; probe the FINAL name token"),
 ("A","entity","salient","historical_figure","mean",
  "A · salient entity · EN\nmean over name",
  ["George","Washington"],[0,1],"average of all name tokens"),
 ("B","entity","obscure","rows_bare","ent_last",
  "B · obscure entity · EN\nname alone, last token",
  ["Ashur","bani","pal"],[2],"bare ruler name, final token"),
 ("B","entity","obscure","rows_bare","ent_mean",
  "B · obscure entity · EN\nname alone, mean",
  ["Ashur","bani","pal"],[0,1,2],"bare ruler name, averaged"),
 ("B","entity","obscure","rows_all","ent_last",
  "B · obscure entity · EN\nin sentence, ENTITY-last",
  ["This","tablet","dates","to","the","reign","of","Ashur","bani","pal","."],[9],
  "carrier sentence; probe last token OF THE NAME"),
 ("B","entity","obscure","rows_all","last",
  "B · obscure entity · EN\nin sentence, SENTENCE-last",
  ["This","tablet","dates","to","the","reign","of","Ashur","bani","pal","."],[10],
  "carrier sentence; probe the final token of the SENTENCE"),
 ("B'","fragment","obscure","tier0","last",
  "B′ · fragment · EN gloss (tier0)\nlast token",
  ["warrior","smite","with","weapon","ox","sheep","…","herald","of"],[8],
  "whole translated fragment; final token"),
 ("B'","fragment","obscure","tier0","mean",
  "B′ · fragment · EN gloss (tier0)\nMEAN over fragment",
  ["warrior","smite","with","weapon","ox","sheep","…","herald","of"],list(range(9)),
  "whole translated fragment; averaged"),
 ("C","fragment","obscure","maximal","last",
  "C · fragment · AKKADIAN (maximal)\nlast token",
  ["lu-qu-ra-di-šu","u-ra-si-bu","ina","ṣe-e-ni","šal-la-su","…","bal"],[6],
  "whole Akkadian fragment; final token"),
 ("C","fragment","obscure","maximal","mean",
  "C · fragment · AKKADIAN (maximal)\nMEAN over fragment",
  ["lu-qu-ra-di-šu","u-ra-si-bu","ina","ṣe-e-ni","šal-la-su","…","bal"],list(range(7)),
  "whole Akkadian fragment; averaged"),
]
CELLC={"A":"#1b7837","B":"#762a83","B'":"#2166ac","C":"#b2182b"}

def build(csv, outfile):
    d=pd.read_csv(csv)
    d=d[(d.metric=="spearman")&(d.target=="year")&(d.probe=="ridge")]
    n=len(ROWS)
    fig=plt.figure(figsize=(16.5,2.35*n))
    gs=fig.add_gridspec(n,2,width_ratios=[1.05,2.6],hspace=.55,wspace=.06,
                        left=.035,right=.985,top=.925,bottom=.05)
    xs=np.arange(len(ARMS))
    for ri,(cell,lev,sal,cl,pool,rlab,toks,hi,note) in enumerate(ROWS):
        sub=d[(d.level==lev)&(d.salience==sal)&(d.cleaning==cl)&(d.pooling==pool)]
        ref=sub[sub.arm=="random"].value
        ref=float(ref.iloc[0]) if len(ref) else np.nan
        # ---- left: the stimulus illustration
        ax=fig.add_subplot(gs[ri,0]); ax.axis("off")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.add_patch(Rectangle((0,0),.012,1,color=CELLC[cell],transform=ax.transAxes,clip_on=False))
        ax.text(.035,.88,rlab,fontsize=8.6,va="top",fontweight="bold",transform=ax.transAxes)
        x=.038; y=.46
        for i,t in enumerate(toks):
            on=i in hi
            w=max(.052,.0175*len(t)+.022)
            if x+w>1.0: x=.038; y-=.20
            ax.add_patch(Rectangle((x,y-.055),w,.135,transform=ax.transAxes,
                                   fc=CELLC[cell] if on else "#f0f0f0",
                                   ec=CELLC[cell] if on else "#cccccc",
                                   alpha=.95 if on else .9,lw=.8,clip_on=False))
            ax.text(x+w/2,y+.012,t,fontsize=6.4,ha="center",va="center",
                    color="white" if on else "#444",fontweight="bold" if on else "normal",
                    transform=ax.transAxes)
            x+=w+.006
        ax.text(.038,.035,note,fontsize=6.2,style="italic",color="#555",transform=ax.transAxes)
        # ---- right: the gap bars
        ax=fig.add_subplot(gs[ri,1])
        for i,m in enumerate(ARMS):
            v=sub[sub.arm==m].value
            if not len(v) or not np.isfinite(ref): continue
            g=float(v.iloc[0])-ref
            ax.bar(i,g,.72,color=COL.get(m,"#888"),edgecolor="k",linewidth=.4,
                   hatch="//" if m.endswith("random") else None,
                   label=LAB.get(m,m) if ri==0 else None)
        tf=sub[sub.arm=="tfidf"]
        if not len(tf):
            tf=d[(d.level==lev)&(d.salience==sal)&(d.cleaning==cl)&(d.pooling=="text")]
        if len(tf) and np.isfinite(ref):
            ax.axhline(float(tf.value.iloc[0])-ref,color="k",ls=":",lw=1.6,
                       label="TF-IDF floor" if ri==0 else None)
        ax.axhline(0,color="k",lw=1.1)
        ax.set_xticks(xs)
        ax.set_xticklabels([LAB.get(m,m) for m in ARMS] if ri==n-1 else [],
                           rotation=40,ha="right",fontsize=6.8)
        ax.grid(alpha=.22,axis="y"); ax.set_axisbelow(True)
        ax.set_ylabel("Δρ vs rand-Qwen",fontsize=7.2)
        ax.tick_params(axis="y",labelsize=6.8)
        ax.text(.995,.93,f"random-Qwen3-8B ρ = {ref:.3f}",transform=ax.transAxes,
                ha="right",va="top",fontsize=6.4,color="#666",
                bbox=dict(fc="w",ec="#ccc",alpha=.85,pad=1.6))
    h,l=fig.axes[1].get_legend_handles_labels()
    fig.legend(h,l,loc="upper center",ncol=8,fontsize=7.2,frameon=False,bbox_to_anchor=(.5,1.0))
    fig.suptitle("Learned chronology above the random-init control, per stimulus × pooling\n"
                 "target = YEAR · metric = Spearman ρ at each arm's best layer · ridge probe · "
                 "hatched = random-init controls · dotted = TF-IDF floor",
                 fontsize=11.5,fontweight="bold",y=.962)
    _save_fig(fig, outfile); plt.close(fig)
    print("saved",outfile)
