# The dated inscriptions, read out on the thesis protocol

Ridge on frozen features of the 1,193 dated royal inscriptions, ruler-grouped folds, SLA §7: `gkf` is the POOLED Spearman over the held-out docs (per-fold rho is undefined when 39 of 40 rulers carry one year), `mc` the mean over the frozen balanced draws. Same protocol and same folds as `EMIN2_RESULT.md`, so the rows are directly comparable; the only thing that changes between rows is which representation the ridge sees. No SSL run ever trained on these documents.

| representation | n docs | mc rho | gkf rho (pooled) |
|---|---|---|---|
| `Thalesian/cuneiformBase-400m::L12::mean` | 1,176 | 0.448 | 0.419 |
| `ssl::ssl_byol_cunei400m-s0::L0::h` | 1,176 | 0.470 | 0.393 |
| `ssl::ssl_infonce_cunei400m-s0::L0::h` | 1,176 | 0.381 | 0.380 |
| `ssl::ssl_jepa_cunei400m-s0::L0::h` | 1,176 | 0.494 | 0.376 |
| `ssl::ssl_infonce_llama2_7b-s0::L0::h` | 1,176 | 0.417 | 0.374 |
| `Thalesian/cuneiformBase-400m::L6::mean` | 1,176 | 0.426 | 0.361 |
| `Qwen/Qwen3-8B::L27::mean` | 1,176 | 0.353 | 0.337 |
| `ssl_e2e::e2e_jepa_XL-s0::L0::h` | 1,176 | 0.403 | 0.333 |
| `ssl_e2e::e2e_jepa_L-s0::L0::h` | 1,176 | 0.367 | 0.324 |
| `ssl::ssl_barlow_cunei400m-s0::L0::h` | 1,176 | 0.419 | 0.322 |
| `ssl::ssl_infonce_qwen3_8b-s0::L0::h` | 1,176 | 0.385 | 0.292 |
| `Thalesian/AKK_300m::L4::mean` | 1,176 | 0.310 | 0.287 |
| `ssl_e2e::e2e_jepa_M-s0::L0::h` | 1,176 | 0.365 | 0.285 |
| `ssl::ssl_jepa_akk300m-s0::L0::h` | 1,176 | 0.389 | 0.282 |
| `ssl::ssl_jepa_qwen3_8b-s0::L0::h` | 1,176 | 0.248 | 0.279 |
| `ssl::ssl_infonce_akk300m-s0::L0::h` | 1,176 | 0.396 | 0.275 |
| `ssl::ssl_jepa_llama2_7b-s0::L0::h` | 1,176 | 0.316 | 0.269 |
| `ssl::ssl_byol_akk300m-s0::L0::h` | 1,176 | 0.377 | 0.269 |
| `Thalesian/AKK_300m::L8::mean` | 1,176 | 0.338 | 0.265 |
| `Qwen/Qwen3-8B::L18::mean` | 1,176 | 0.261 | 0.262 |
| `ssl_e2e::e2e_barlow_M-s0::L0::h` | 1,176 | 0.364 | 0.255 |
| `ssl::ssl_barlow_akk300m-s0::L0::h` | 1,176 | 0.340 | 0.240 |
| `ssl::ssl_byol_llama2_7b-s0::L0::h` | 1,176 | 0.285 | 0.228 |
| `ssl_e2e::e2e_barlow_L-s0::L0::h` | 1,176 | 0.307 | 0.219 |
| `ssl::ssl_barlow_qwen3_8b-s0::L0::h` | 1,176 | 0.251 | 0.217 |
| `ssl::ssl_barlow_llama2_7b-s0::L0::h` | 1,176 | 0.292 | 0.217 |
| `ssl_e2e::e2e_jepa_S-s0::L0::h` | 1,176 | 0.383 | 0.209 |
| `ssl::ssl_byol_qwen3_8b-s0::L0::h` | 1,176 | 0.190 | 0.135 |
| `ssl_e2e::e2e_barlow_S-s0::L0::h` | 1,176 | 0.298 | 0.133 |
| `ssl_e2e::e2e_barlow_XL-s0::L0::h` | 1,176 | 0.260 | 0.130 |
