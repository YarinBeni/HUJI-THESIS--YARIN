# P10 — reduce-then-kernel: does pre-reducing help?

gkpls Spearman(year) under balanced-MC, per reducer×norm. **raw** = the P9/P8 anchor (no reduction). Δ = best reduce+norm − raw.

## gpt_oss_120b · engtier0  (raw=0.242)
best: **raw/zscore** gkpls=0.319 (Δ=+0.077), dial_pred=0.247

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.242 |    0.319 | 0.253 |
| pca       |  0.163 |    0.123 | 0.148 |
| pls       |  0.182 |    0.174 | 0.164 |
| umap      |  0.25  |    0.235 | 0.199 |

## gpt_oss_120b · maximal  (raw=0.279)
best: **raw/zscore** gkpls=0.292 (Δ=+0.013), dial_pred=0.209

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.279 |    0.292 | 0.283 |
| pca       |  0.22  |    0.23  | 0.21  |
| pls       |  0.248 |    0.225 | 0.27  |
| umap      |  0.266 |    0.263 | 0.216 |

## mlm · maximal  (raw=0.187)
best: **pls/l2** gkpls=0.268 (Δ=+0.081), dial_pred=0.265

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.187 |    0.168 | 0.224 |
| pca       |  0.148 |    0.131 | 0.139 |
| pls       |  0.248 |    0.233 | 0.268 |
| umap      |  0.201 |    0.187 | 0.15  |

## qwen3_1b7 · engtier0  (raw=0.288)
best: **umap/none** gkpls=0.311 (Δ=+0.022), dial_pred=0.250

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.288 |    0.299 | 0.288 |
| pca       |  0.163 |    0.141 | 0.098 |
| pls       |  0.26  |    0.242 | 0.219 |
| umap      |  0.311 |    0.308 | 0.272 |

## qwen3_1b7 · maximal  (raw=0.231)
best: **pls/l2** gkpls=0.302 (Δ=+0.071), dial_pred=0.291

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.231 |    0.284 | 0.259 |
| pca       |  0.151 |    0.181 | 0.225 |
| pls       |  0.265 |    0.298 | 0.302 |
| umap      |  0.227 |    0.221 | 0.202 |

## qwen3_32b · engtier0  (raw=0.296)
best: **raw/zscore** gkpls=0.398 (Δ=+0.102), dial_pred=0.354

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.296 |    0.398 | 0.346 |
| pca       |  0.243 |    0.301 | 0.201 |
| pls       |  0.371 |    0.383 | 0.349 |
| umap      |  0.255 |    0.246 | 0.204 |

## qwen3_32b · maximal  (raw=0.283)
best: **raw/l2** gkpls=0.302 (Δ=+0.019), dial_pred=0.252

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.283 |    0.288 | 0.302 |
| pca       |  0.293 |    0.301 | 0.298 |
| pls       |  0.28  |    0.233 | 0.299 |
| umap      |  0.272 |    0.272 | 0.225 |

## qwen3_8b · engtier0  (raw=0.237)
best: **raw/l2** gkpls=0.289 (Δ=+0.052), dial_pred=0.210

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.237 |    0.283 | 0.289 |
| pca       |  0.134 |    0.122 | 0.118 |
| pls       |  0.213 |    0.213 | 0.19  |
| umap      |  0.246 |    0.25  | 0.176 |

## qwen3_8b · maximal  (raw=0.264)
best: **pls/l2** gkpls=0.292 (Δ=+0.028), dial_pred=0.195

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.264 |    0.266 | 0.281 |
| pca       |  0.223 |    0.255 | 0.253 |
| pls       |  0.264 |    0.245 | 0.292 |
| umap      |  0.259 |    0.246 | 0.219 |

## random · engtier0  (raw=0.173)
best: **raw/l2** gkpls=0.287 (Δ=+0.114), dial_pred=0.255

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.173 |    0.181 | 0.287 |
| pca       |  0.22  |    0.206 | 0.181 |
| pls       |  0.182 |    0.193 | 0.173 |
| umap      |  0.231 |    0.224 | 0.191 |

## random · maximal  (raw=0.275)
best: **raw/l2** gkpls=0.311 (Δ=+0.036), dial_pred=0.224

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.275 |    0.269 | 0.311 |
| pca       |  0.204 |    0.171 | 0.174 |
| pls       |  0.265 |    0.272 | 0.227 |
| umap      |  0.225 |    0.227 | 0.174 |

## tfidf · maximal  (raw=0.276)
best: **umap/none** gkpls=0.299 (Δ=+0.022), dial_pred=0.234

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.276 |    0.018 | 0.278 |
| pca       |  0.27  |    0.244 | 0.271 |
| pls       |  0.239 |    0.218 | 0.259 |
| umap      |  0.299 |    0.291 | 0.221 |

## thalesian_akk300m · engtier0  (raw=0.339)
best: **pls/none** gkpls=0.345 (Δ=+0.006), dial_pred=0.326

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.339 |    0.34  | 0.342 |
| pca       |  0.238 |    0.242 | 0.228 |
| pls       |  0.345 |    0.336 | 0.301 |
| umap      |  0.305 |    0.291 | 0.265 |

## thalesian_akk300m · maximal  (raw=0.245)
best: **pls/l2** gkpls=0.283 (Δ=+0.038), dial_pred=0.236

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.245 |    0.241 | 0.245 |
| pca       |  0.181 |    0.201 | 0.233 |
| pls       |  0.265 |    0.257 | 0.283 |
| umap      |  0.24  |    0.227 | 0.195 |

## thalesian_cunei400m · engtier0  (raw=0.362)
best: **raw/l2** gkpls=0.366 (Δ=+0.004), dial_pred=0.248

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.362 |    0.359 | 0.366 |
| pca       |  0.33  |    0.329 | 0.256 |
| pls       |  0.357 |    0.351 | 0.289 |
| umap      |  0.303 |    0.287 | 0.254 |

## thalesian_cunei400m · maximal  (raw=0.300)
best: **pls/l2** gkpls=0.383 (Δ=+0.082), dial_pred=0.351

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.3   |    0.301 | 0.297 |
| pca       |  0.263 |    0.234 | 0.252 |
| pls       |  0.361 |    0.37  | 0.383 |
| umap      |  0.296 |    0.286 | 0.261 |

## umt5_base · engtier0  (raw=0.338)
best: **raw/l2** gkpls=0.346 (Δ=+0.009), dial_pred=0.213

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.338 |    0.34  | 0.346 |
| pca       |  0.256 |    0.255 | 0.243 |
| pls       |  0.342 |    0.344 | 0.293 |
| umap      |  0.305 |    0.295 | 0.263 |

## umt5_base · maximal  (raw=0.220)
best: **pls/l2** gkpls=0.267 (Δ=+0.048), dial_pred=0.227

| reducer   |   none |   zscore |    l2 |
|:----------|-------:|---------:|------:|
| raw       |  0.22  |    0.23  | 0.243 |
| pca       |  0.178 |    0.187 | 0.214 |
| pls       |  0.254 |    0.242 | 0.267 |
| umap      |  0.2   |    0.202 | 0.16  |

