import numpy as np, pandas as pd, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="v_1/src/world_models/manifold/results"

def six_panel(tag, proj, title, outfile, max_cats=8):
    z=np.load(f"{R}/{tag}.coords.npz")
    P=z[proj]
    if P.shape[0]==0: print("no",proj,"for",tag); return False
    fp=f"{R}/{tag}.facets.csv.gz"
    F=pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()
    cols=list(F.columns)[:6]
    if len(cols)<6 and "y" in z.files:
        F=F.copy(); F["target"]=z["y"]; cols=(cols+["target"])[:6]
    X,Y=P[:,0],P[:,1]
    n=len(cols)
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    for ax,c in zip(axes.ravel(),cols):
        v=F[c].values[:len(X)]
        num=pd.api.types.is_numeric_dtype(F[c])
        if num:
            vv=pd.to_numeric(v,errors="coerce")
            s=ax.scatter(X,Y,c=vv,s=6,cmap="viridis",alpha=.75)
            plt.colorbar(s,ax=ax,fraction=.046)
        else:
            vs=pd.Series(v).astype(str)
            top=vs.value_counts().head(max_cats).index.tolist()
            cmap=plt.get_cmap("tab10")
            other=~vs.isin(top)
            ax.scatter(X[other.values],Y[other.values],s=4,c="#DDD",alpha=.5)
            for i,t in enumerate(top):
                m=(vs==t).values
                ax.scatter(X[m],Y[m],s=7,color=cmap(i%10),alpha=.85,
                           label=f"{t[:18]} ({m.sum()})")
            ax.legend(fontsize=5.5,loc="best",framealpha=.85)
        ax.set_title(c,fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]: ax.axis("off")
    fig.suptitle(title,fontweight="bold")
    fig.tight_layout(); fig.savefig(outfile,dpi=120); plt.close(fig)
    print("saved",outfile); return True
