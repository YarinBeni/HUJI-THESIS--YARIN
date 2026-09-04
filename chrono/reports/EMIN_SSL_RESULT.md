# The dated inscriptions, read out on the thesis protocol

Ridge on frozen features of the 1,193 dated royal inscriptions, ruler-grouped folds, SLA §7: `gkf` is the POOLED Spearman over the held-out docs (per-fold rho is undefined when 39 of 40 rulers carry one year), `mc` the mean over the frozen balanced draws. Same protocol and same folds as `EMIN2_RESULT.md`, so the rows are directly comparable; the only thing that changes between rows is which representation the ridge sees. No SSL run ever trained on these documents.


**Sanity check against the published run.** `Thalesian/cuneiformBase-400m::L12` reads mc .448 here; the same encoder, folds and probe read .45 +- .02 as the ridge baseline of E-MIN v2. The protocol reproduces, so the differences between rows are about the representation and not about the harness.

**Reference points from `EMIN2_RESULT.md` (Akkadian arm, mc rho):** ridge on the frozen cuneiformBase-400m .45, on Llama-2-7B .35, on Qwen3-8B .26; the SUPERVISED Chrono-Barlow head on the same features .61 / .54 / .43. Nothing below reaches the supervised head.

| representation | n docs | mc rho | gkf rho (pooled) |
|---|---|---|---|
| `Thalesian/cuneiformBase-400m::L12::mean` | 1,176 | 0.448 | 0.419 |
| `ssl::ssl_barlow_cunei400m_wdated-s2::L0::h` | 1,176 | 0.471 | 0.417 |
| `ssl::ssl_barlow_cunei400m_wdated-s1::L0::h` | 1,176 | 0.422 | 0.415 |
| `ssl::ssl_barlow_cunei400m_leace_wdated-s0::L0::h` | 1,176 | 0.482 | 0.414 |
| `ssl::ssl_byol_cunei400m_wdated-s2::L0::h` | 1,176 | 0.499 | 0.413 |
| `ssl::ssl_barlow_cunei400m_wdated-s0::L0::h` | 1,176 | 0.491 | 0.410 |
| `ssl::ssl_jepa_cunei400m_leace_wdated-s0::L0::h` | 1,176 | 0.473 | 0.401 |
| `ssl::ssl_byol_cunei400m_wdated-s0::L0::h` | 1,176 | 0.519 | 0.401 |
| `ssl::ssl_byol_cunei400m-s0::L0::h` | 1,176 | 0.470 | 0.393 |
| `ssl::ssl_infonce_cunei400m_wdated-s0::L0::h` | 1,176 | 0.425 | 0.388 |
| `ssl::ssl_byol_cunei400m_wdated-s1::L0::h` | 1,176 | 0.457 | 0.384 |
| `ssl::ssl_infonce_cunei400m-s0::L0::h` | 1,176 | 0.381 | 0.380 |
| `ssl::ssl_jepa_cunei400m-s0::L0::h` | 1,176 | 0.494 | 0.376 |
| `ssl::ssl_infonce_llama2_7b-s0::L0::h` | 1,176 | 0.417 | 0.374 |
| `ssl::ssl_jepa_cunei400m_wdated-s0::L0::h` | 1,176 | 0.473 | 0.369 |
| `Thalesian/cuneiformBase-400m::L6::mean` | 1,176 | 0.426 | 0.361 |
| `ssl::ssl_barlow_cunei400m_leopard-s0::L0::h` | 1,176 | 0.447 | 0.354 |
| `ssl::ssl_jepa_cunei400m_leace-s0::L0::h` | 1,176 | 0.437 | 0.352 |
| `ssl::ssl_byol_cunei400m_leace_wdated-s0::L0::h` | 1,176 | 0.437 | 0.342 |
| `Qwen/Qwen3-8B::L27::mean` | 1,176 | 0.353 | 0.337 |
| `ssl_e2e::e2e_jepa_XL-s0::L0::h` | 1,176 | 0.403 | 0.333 |
| `ssl_e2e::e2e_jepa_L-s0::L0::h` | 1,176 | 0.367 | 0.324 |
| `ssl::ssl_barlow_cunei400m-s0::L0::h` | 1,176 | 0.419 | 0.322 |
| `ssl::ssl_barlow_cunei400m_adv-s0::L0::h` | 1,176 | 0.419 | 0.313 |
| `ssl::ssl_barlow_cunei400m_leace-s0::L0::h` | 1,176 | 0.340 | 0.297 |
| `ssl_hyb::hyb_barlow_S_thalesian_cunei400m-s0::L0::h` | 1,176 | 0.386 | 0.294 |
| `ssl::ssl_infonce_qwen3_8b-s0::L0::h` | 1,176 | 0.385 | 0.292 |
| `Thalesian/AKK_300m::L4::mean` | 1,176 | 0.310 | 0.287 |
| `ssl_e2e::e2e_jepa_M-s0::L0::h` | 1,176 | 0.365 | 0.285 |
| `ssl::ssl_jepa_akk300m-s0::L0::h` | 1,176 | 0.389 | 0.282 |
| `ssl::ssl_jepa_qwen3_8b-s0::L0::h` | 1,176 | 0.248 | 0.279 |
| `ssl::ssl_infonce_akk300m-s0::L0::h` | 1,176 | 0.396 | 0.275 |
| `ssl::ssl_jepa_llama2_7b-s0::L0::h` | 1,176 | 0.316 | 0.269 |
| `ssl::ssl_byol_akk300m-s0::L0::h` | 1,176 | 0.377 | 0.269 |
| `Thalesian/AKK_300m::L8::mean` | 1,176 | 0.338 | 0.265 |
| `ssl::ssl_jepa_cunei400m_leopard-s0::L0::h` | 1,176 | 0.334 | 0.265 |
| `Qwen/Qwen3-8B::L18::mean` | 1,176 | 0.261 | 0.262 |
| `ssl_e2e::e2e_barlow_M-s0::L0::h` | 1,176 | 0.364 | 0.255 |
| `ssl::ssl_jepa_cunei400m_both-s0::L0::h` | 1,176 | 0.199 | 0.247 |
| `ssl::ssl_barlow_akk300m-s0::L0::h` | 1,176 | 0.340 | 0.240 |
| `ssl::ssl_byol_llama2_7b-s0::L0::h` | 1,176 | 0.285 | 0.228 |
| `ssl_e2e::e2e_barlow_L-s0::L0::h` | 1,176 | 0.307 | 0.219 |
| `ssl::ssl_barlow_qwen3_8b-s0::L0::h` | 1,176 | 0.251 | 0.217 |
| `ssl::ssl_barlow_llama2_7b-s0::L0::h` | 1,176 | 0.292 | 0.217 |
| `ssl_e2e::e2e_jepa_S-s0::L0::h` | 1,176 | 0.383 | 0.209 |
| `ssl_hyb::hyb_jepa_S_thalesian_cunei400m-s0::L0::h` | 1,176 | 0.291 | 0.209 |
| `ssl::ssl_barlow_cunei400m_both-s0::L0::h` | 1,176 | 0.151 | 0.198 |
| `ssl_hyb::hyb_barlow_M_llama2_7b-s0::L0::h` | 1,176 | 0.283 | 0.198 |
| `ssl_hyb::hyb_barlow_S_llama2_7b-s0::L0::h` | 1,176 | 0.183 | 0.177 |
| `ssl_hyb::hyb_barlow_M_thalesian_cunei400m-s0::L0::h` | 1,176 | 0.203 | 0.166 |
| `ssl_hyb::hyb_jepa_S_llama2_7b-s0::L0::h` | 1,176 | 0.204 | 0.164 |
| `ssl::ssl_jepa_cunei400m_adv-s0::L0::h` | 1,176 | 0.314 | 0.149 |
| `ssl::ssl_byol_qwen3_8b-s0::L0::h` | 1,176 | 0.190 | 0.135 |
| `ssl_e2e::e2e_barlow_S-s0::L0::h` | 1,176 | 0.298 | 0.133 |
| `ssl_hyb::hyb_jepa_M_llama2_7b-s0::L0::h` | 1,176 | 0.150 | 0.131 |
| `ssl_e2e::e2e_barlow_XL-s0::L0::h` | 1,176 | 0.260 | 0.130 |
| `ssl_hyb::hyb_jepa_M_thalesian_cunei400m-s0::L0::h` | 1,176 | 0.194 | 0.113 |

## By family

| family | cells | best mc | median mc | best gkf | median gkf |
|---|---|---|---|---|---|
| adapter on a frozen encoder | 35 | 0.519 | 0.417 | 0.417 | 0.322 |
| from scratch (raw signs) | 8 | 0.403 | 0.364 | 0.333 | 0.237 |
| frozen encoder, no SSL | 6 | 0.448 | 0.346 | 0.419 | 0.312 |
| hybrid (frozen states -> fresh transformer) | 8 | 0.386 | 0.204 | 0.294 | 0.171 |
