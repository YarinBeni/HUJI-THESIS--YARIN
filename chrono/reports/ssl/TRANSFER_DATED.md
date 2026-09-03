# Transfer to the dated royal inscriptions

Fit on 6,328 undated texts (period midpoint as the target), evaluated on 1,176 dated royal inscriptions with a known year that no SSL run ever saw. `rho` is Spearman between the predicted year and the true year — the number to read. `acc` is balanced accuracy over the periods with >= 20 test and >= 50 training documents, with per-class recall beside it; earlier tables scored classes with as little as one test document and are superseded here.

`rho within NA` repeats it over the 924 Neo-Assyrian inscriptions alone — the same question the thesis asks, with the easy between-period contrast removed.

| model | transfer rho | rho within NA | classes | bal acc | per-class recall |
|---|---|---|---|---|---|
| `ssl::ssl_barlow_cunei400m-s0::L0::h` | +0.269 | -0.063 | 2 | 0.817 | Middle Babylonian 0.82 (n=28), Neo-Assyrian 0.81 (n=924) |
| `Thalesian/cuneiformBase-400m::L12::mean` | +0.268 | -0.048 | 2 | 0.650 | Middle Babylonian 0.57 (n=28), Neo-Assyrian 0.73 (n=924) |
| `ssl::ssl_infonce_cunei400m-s0::L0::h` | +0.252 | -0.064 | 2 | 0.622 | Middle Babylonian 0.43 (n=28), Neo-Assyrian 0.81 (n=924) |
| `ssl::ssl_jepa_cunei400m-s0::L0::h` | +0.246 | -0.062 | 2 | 0.713 | Middle Babylonian 0.89 (n=28), Neo-Assyrian 0.53 (n=924) |
| `ssl::ssl_byol_cunei400m-s0::L0::h` | +0.243 | -0.066 | 2 | 0.737 | Middle Babylonian 0.54 (n=28), Neo-Assyrian 0.94 (n=924) |
| `Thalesian/cuneiformBase-400m::L6::mean` | +0.227 | -0.040 | 2 | 0.756 | Middle Babylonian 0.61 (n=28), Neo-Assyrian 0.90 (n=924) |
| `ssl_e2e::e2e_barlow_S-s0::L0::h` | +0.201 | -0.096 | 2 | 0.460 | Middle Babylonian 0.46 (n=28), Neo-Assyrian 0.46 (n=924) |
| `ssl_e2e::e2e_jepa_XL-s0::L0::h` | +0.182 | -0.137 | 2 | 0.624 | Middle Babylonian 0.54 (n=28), Neo-Assyrian 0.71 (n=924) |
| `ssl_e2e::e2e_barlow_L-s0::L0::h` | +0.174 | -0.153 | 2 | 0.543 | Middle Babylonian 0.50 (n=28), Neo-Assyrian 0.59 (n=924) |
| `ssl::ssl_jepa_akk300m-s0::L0::h` | +0.160 | -0.072 | 2 | 0.802 | Middle Babylonian 0.71 (n=28), Neo-Assyrian 0.89 (n=924) |
| `ssl::ssl_infonce_llama2_7b-s0::L0::h` | +0.139 | -0.116 | 2 | 0.505 | Middle Babylonian 0.25 (n=28), Neo-Assyrian 0.76 (n=924) |
| `ssl_e2e::e2e_barlow_M-s0::L0::h` | +0.131 | -0.145 | 2 | 0.547 | Middle Babylonian 0.46 (n=28), Neo-Assyrian 0.63 (n=924) |
| `ssl::ssl_barlow_llama2_7b-s0::L0::h` | +0.125 | -0.164 | 2 | 0.475 | Middle Babylonian 0.25 (n=28), Neo-Assyrian 0.70 (n=924) |
| `ssl::ssl_infonce_akk300m-s0::L0::h` | +0.123 | -0.150 | 2 | 0.667 | Middle Babylonian 0.50 (n=28), Neo-Assyrian 0.83 (n=924) |
| `Thalesian/AKK_300m::L4::mean` | +0.123 | -0.163 | 2 | 0.653 | Middle Babylonian 0.57 (n=28), Neo-Assyrian 0.73 (n=924) |
| `ssl_e2e::e2e_barlow_XL-s0::L0::h` | +0.122 | -0.220 | 2 | 0.747 | Middle Babylonian 0.79 (n=28), Neo-Assyrian 0.71 (n=924) |
| `ssl_e2e::e2e_jepa_L-s0::L0::h` | +0.116 | -0.251 | 2 | 0.548 | Middle Babylonian 0.39 (n=28), Neo-Assyrian 0.70 (n=924) |
| `ssl::ssl_barlow_akk300m-s0::L0::h` | +0.109 | -0.186 | 2 | 0.872 | Middle Babylonian 0.86 (n=28), Neo-Assyrian 0.89 (n=924) |
| `Thalesian/AKK_300m::L8::mean` | +0.096 | -0.166 | 2 | 0.754 | Middle Babylonian 0.61 (n=28), Neo-Assyrian 0.90 (n=924) |
| `ssl_e2e::e2e_jepa_S-s0::L0::h` | +0.096 | -0.118 | 2 | 0.482 | Middle Babylonian 0.36 (n=28), Neo-Assyrian 0.61 (n=924) |
| `ssl::ssl_infonce_qwen3_8b-s0::L0::h` | +0.092 | -0.116 | 2 | 0.376 | Middle Babylonian 0.29 (n=28), Neo-Assyrian 0.47 (n=924) |
| `ssl_e2e::e2e_jepa_M-s0::L0::h` | +0.090 | -0.210 | 2 | 0.539 | Middle Babylonian 0.43 (n=28), Neo-Assyrian 0.65 (n=924) |
| `ssl::ssl_byol_akk300m-s0::L0::h` | +0.081 | -0.108 | 2 | 0.890 | Middle Babylonian 0.86 (n=28), Neo-Assyrian 0.92 (n=924) |
| `ssl::ssl_byol_qwen3_8b-s0::L0::h` | +0.080 | -0.160 | 2 | 0.431 | Middle Babylonian 0.14 (n=28), Neo-Assyrian 0.72 (n=924) |
| `ssl::ssl_byol_llama2_7b-s0::L0::h` | +0.069 | -0.225 | 2 | 0.540 | Middle Babylonian 0.32 (n=28), Neo-Assyrian 0.76 (n=924) |
| `Qwen/Qwen3-8B::L27::mean` | +0.065 | -0.170 | 2 | 0.545 | Middle Babylonian 0.50 (n=28), Neo-Assyrian 0.59 (n=924) |
| `Qwen/Qwen3-8B::L18::mean` | +0.063 | -0.179 | 2 | 0.372 | Middle Babylonian 0.36 (n=28), Neo-Assyrian 0.39 (n=924) |
| `ssl::ssl_barlow_qwen3_8b-s0::L0::h` | +0.061 | -0.158 | 2 | 0.385 | Middle Babylonian 0.29 (n=28), Neo-Assyrian 0.48 (n=924) |
| `ssl::ssl_jepa_llama2_7b-s0::L0::h` | +0.056 | -0.117 | 2 | 0.455 | Middle Babylonian 0.18 (n=28), Neo-Assyrian 0.73 (n=924) |
| `ssl::ssl_jepa_qwen3_8b-s0::L0::h` | +0.018 | -0.177 | 2 | 0.437 | Middle Babylonian 0.29 (n=28), Neo-Assyrian 0.59 (n=924) |
