import numpy as np, os, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"v_1/src/world_models/manifold")
import manifold_lib as ML
# Resolution/format policy lives in figures/lib/_save.py (300 dpi PNG + vector PDF)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from _save import save as _save_fig  # noqa: E402

R="v_1/src/world_models/manifold/results"

def get(tag):
    f=f"{R}/{tag}.coords.npz"
    if not os.path.exists(f): return None,None
    z=np.load(f); return z["pca"], z["y"]

def pc_grid(tags_titles, outfile, pairs=((0,1),(1,2),(2,3),(3,4)), cmap="viridis"):
    """Their 'Years of the 20th Century' figure: PC pairs coloured by the target.
    Rows = surface, cols = consecutive PC pairs (their years arc lived in PC3-PC4)."""
    rows=len(tags_titles)
    fig,axes=plt.subplots(rows,len(pairs),figsize=(4.1*len(pairs),4.0*rows),squeeze=False)
    for ri,(tag,title) in enumerate(tags_titles):
        P,y=get(tag)
        for ci,(a,b) in enumerate(pairs):
            ax=axes[ri][ci]
            if P is None or P.shape[1]<=max(a,b):
                ax.axis("off"); continue
            s=ax.scatter(P[:,a],P[:,b],c=y,s=7,cmap=cmap,alpha=.8,linewidths=0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"PCA axis {a+1}",fontsize=8); ax.set_ylabel(f"PCA axis {b+1}",fontsize=8)
            if ci==0: ax.set_title(title,fontsize=9,loc="left")
            if ci==len(pairs)-1: plt.colorbar(s,ax=ax,fraction=.046)
    fig.suptitle("Where does the chronology curve live? PCA axis pairs coloured by year "
                 "(Engels et al.: the years arc appears in PC3–PC4, not PC1–PC2)",fontweight="bold")
    fig.tight_layout(); _save_fig(fig, outfile); plt.close(fig); print("saved",outfile)

def distance_figs(tags_titles, outfile, k=10, rank=4, cap=700):
    """Modell et al.'s two diagnostics, drawn as THEY draw them:
       left  : squared feature distance vs cosine similarity, Chatterjee xi
       right : feature distance vs graph-geodesic distance,   Pearson rho
    Points coloured by the year of point i."""
    rows=len(tags_titles)
    fig,axes=plt.subplots(rows,2,figsize=(11,4.6*rows),squeeze=False)
    for ri,(tag,title) in enumerate(tags_titles):
        P,y=get(tag)
        if P is None:
            axes[ri][0].axis("off"); axes[ri][1].axis("off"); continue
        if len(P)>cap:
            sel=np.random.RandomState(0).choice(len(P),cap,replace=False); sel.sort()
            P,y=P[sel],y[sel]
        X=P[:,:rank].astype(np.float64)
        n=np.linalg.norm(X,axis=1); m=n>1e-9
        X=X[m]/n[m][:,None]; yy=y[m].astype(float)
        D=np.abs(yy[:,None]-yy[None,:])
        cos=ML.cosine_similarity_matrix(X)
        Dm,keep,_=ML.manifold_distance(X,k)
        iu=np.triu_indices(len(X),1)
        cy=np.repeat(yy,len(yy)).reshape(len(yy),len(yy))[iu]
        # left: cosine vs squared distance
        ax=axes[ri][0]
        xi=ML.chatterjee_corr(D[iu],cos[iu])
        ax.scatter(D[iu]**2,cos[iu],c=cy,s=.4,alpha=.12,cmap="viridis",linewidths=0)
        ax.set_xlabel("Squared distance between years"); ax.set_ylabel("Cosine similarity")
        ax.set_title(f"{title}",fontsize=9,loc="left")
        ax.text(.97,.06,f"$\\xi = {xi:.3f}$",transform=ax.transAxes,ha="right",
                fontsize=11,bbox=dict(fc="w",ec="k"))
        # right: geodesic vs distance
        ax=axes[ri][1]
        Dk=D[np.ix_(keep,keep)]; iu2=np.triu_indices(keep.sum(),1)
        yk=yy[keep]; cy2=np.repeat(yk,len(yk)).reshape(len(yk),len(yk))[iu2]
        fin=np.isfinite(Dm[iu2])
        rho=ML.pearson(Dk[iu2][fin],Dm[iu2][fin])
        ax.scatter(Dk[iu2][fin],Dm[iu2][fin],c=cy2[fin],s=.4,alpha=.12,cmap="viridis",linewidths=0)
        ax.set_xlabel("Distance between years"); ax.set_ylabel("Manifold (graph-geodesic) distance")
        ax.set_title("isometry check",fontsize=9,loc="left")
        ax.text(.97,.06,f"$\\rho = {rho:.3f}$",transform=ax.transAxes,ha="right",
                fontsize=11,bbox=dict(fc="w",ec="k"))
    fig.suptitle("Modell et al. representation-manifold diagnostics on our data "
                 f"(top-{rank} PCA directions, re-normalised; k={k} graph)",fontweight="bold")
    fig.tight_layout(); _save_fig(fig, outfile); plt.close(fig); print("saved",outfile)
