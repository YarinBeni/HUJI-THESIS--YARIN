# P10 — reduce-then-kernel: does pre-reducing help?

gkpls Spearman(year) under balanced-MC, per reducer×norm. **raw** = the P9/P8 anchor (no reduction). Δ = best reduce+norm − raw.

## tfidf · maximal  (raw=0.276)
best: **umap/none** gkpls=0.299 (Δ=+0.022), dial_pred=0.234

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.276 |    0.018 | 0.278 |
| pca       |  0.27  |    0.244 | 0.271 |
| pls       |  0.239 |    0.218 | 0.259 |
| umap      |  0.299 |    0.291 | 0.221 |

