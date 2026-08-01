import numpy as np
COL={"qwen3_1b7":"#9ecae1","qwen3_8b":"#4292c6","qwen3_32b":"#08519c","gpt_oss_120b":"#08306b",
     "llama2_7b":"#a1d99b","llama2_13b":"#41ab5d","llama2_70b":"#00602a",
     "thalesian_akk300m":"#fd8d3c","thalesian_cunei400m":"#d94801","umt5_base":"#756bb1",
     "random":"#6baed6","llama2_7b_random":"#a1d99b","llama2_13b_random":"#41ab5d",
     "llama2_70b_random":"#00602a","tfidf":"#000000"}
LAB={"qwen3_1b7":"Qwen3-1.7B","qwen3_8b":"Qwen3-8B","qwen3_32b":"Qwen3-32B","gpt_oss_120b":"gpt-oss-120B",
     "llama2_7b":"Llama-2-7B","llama2_13b":"Llama-2-13B","llama2_70b":"Llama-2-70B",
     "thalesian_akk300m":"AKK-300M","thalesian_cunei400m":"cuneiform-400M","umt5_base":"uMT5-base",
     "random":"random Qwen*","llama2_7b_random":"Llama-7B rand*","llama2_13b_random":"Llama-13B rand*",
     "llama2_70b_random":"Llama-70B rand*","tfidf":"TF-IDF*"}
ENC={"thalesian_akk300m","thalesian_cunei400m","umt5_base"}
ORDER=["llama2_70b","llama2_13b","llama2_7b","gpt_oss_120b","qwen3_32b","qwen3_8b","qwen3_1b7",
       "thalesian_cunei400m","thalesian_akk300m","umt5_base",
       "llama2_70b_random","llama2_13b_random","llama2_7b_random","random"]
def isr(m): return m.endswith("random") or m=="random"
def sty(m): return dict(color=COL.get(m,"#888"),ls="--" if isr(m) else "-",lw=1.1 if isr(m) else 2.0)
def star(ax,x,y,m):
    y=np.asarray(y,dtype=float)
    if not np.isfinite(y).any(): return
    i=int(np.nanargmax(y))
    ax.plot(np.asarray(x)[i],y[i],marker="*",ms=13,color=COL.get(m,"#888"),mec="k",mew=.5,zorder=6)
