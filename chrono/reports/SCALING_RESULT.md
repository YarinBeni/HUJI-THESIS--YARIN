# Representation sweep — S1 frozen vs S2 adapters vs S2 from-scratch

Balanced accuracy unless noted; period chance ≈ .17 (6 classes ≥ 30 docs), source chance ≈ .17. A high SOURCE probe with a low WITHIN-source period probe means the model learned corpora, not time.

## Main table

| kind | model | lin period | mlp period | lin source | lin genre | lin provenance | silhouette_period | knn10_purity_period |
|---|---|---|---|---|---|---|---|---|
| frozen encoder | `Qwen/Qwen3-8B::L18::mean` | 0.838 | 0.816 | 0.963 | 0.234 | 0.633 | 0.052 | 0.957 |
| frozen encoder | `Qwen/Qwen3-8B::L27::mean` | 0.856 | 0.798 | 0.963 | 0.249 | 0.655 | 0.080 | 0.964 |
| frozen encoder | `Thalesian/AKK_300m::L4::mean` | 0.841 | 0.825 | 0.961 | 0.273 | 0.660 | 0.096 | 0.980 |
| frozen encoder | `Thalesian/AKK_300m::L8::mean` | 0.865 | 0.816 | 0.939 | 0.261 | 0.650 | 0.113 | 0.969 |
| frozen encoder | `Thalesian/cuneiformBase-400m::L12::mean` | 0.890 | 0.823 | 0.963 | 0.273 | 0.674 | 0.092 | 0.981 |
| frozen encoder | `Thalesian/cuneiformBase-400m::L6::mean` | 0.901 | 0.843 | 0.967 | 0.278 | 0.689 | 0.138 | 0.983 |

## Period probe within source / with a source held out (linear)

| model | within_oracc | within_seal |
|---|---|---|
| `Qwen/Qwen3-8B::L18::mean` | 0.988 | 0.697 |
| `Qwen/Qwen3-8B::L27::mean` | 0.990 | 0.761 |
| `Thalesian/AKK_300m::L4::mean` | 0.992 | 0.795 |
| `Thalesian/AKK_300m::L8::mean` | 0.991 | 0.793 |
| `Thalesian/cuneiformBase-400m::L12::mean` | 0.993 | 0.775 |
| `Thalesian/cuneiformBase-400m::L6::mean` | 0.995 | 0.768 |

## From-scratch family — quick linear period probe during training

| run | params | steps seen | first | best | last |
|---|---|---|---|---|---|
| `ssl_e2e::e2e_barlow_L-s0` | 50.5 M | 56,000 | 0.850 | 0.907 | 0.899 |
| `ssl_e2e::e2e_barlow_M-s0` | 23.6 M | 98,000 | 0.852 | 0.895 | 0.862 |
| `ssl_e2e::e2e_barlow_S-s0` | 9.4 M | 152,000 | 0.856 | 0.922 | 0.887 |
| `ssl_e2e::e2e_barlow_XL-s0` | 104.5 M | 62,000 | 0.877 | 0.885 | 0.884 |
| `ssl_e2e::e2e_jepa_L-s0` | 50.5 M | 96,000 | 0.882 | 0.882 | 0.844 |
| `ssl_e2e::e2e_jepa_M-s0` | 23.6 M | 138,000 | 0.842 | 0.856 | 0.824 |
| `ssl_e2e::e2e_jepa_S-s0` | 9.4 M | 270,000 | 0.848 | 0.883 | 0.839 |
| `ssl_e2e::e2e_jepa_XL-s0` | 104.5 M | 96,000 | 0.841 | 0.850 | 0.832 |

| run | steps | hours | final loss |
|---|---|---|---|
| `ssl_e2e::e2e_barlow_S-s0` | 153,723 | 5.01 | 1.9214 |
| `ssl_e2e::e2e_barlow_M-s0` | 99,922 | 5.01 | 1.9245 |
| `ssl_e2e::e2e_barlow_L-s0` | 56,414 | 5.01 | 2.1461 |
| `ssl_e2e::e2e_barlow_XL-s0` | 63,826 | 5.01 | 2.8620 |
| `ssl_e2e::e2e_jepa_S-s0` | 270,000 | 5.01 | 0.1495 |
| `ssl_e2e::e2e_jepa_M-s0` | 139,130 | 5.01 | 0.1132 |
| `ssl_e2e::e2e_jepa_L-s0` | 96,410 | 5.01 | 0.1009 |
| `ssl_e2e::e2e_jepa_XL-s0` | 96,173 | 5.01 | 0.1452 |
