# Representation sweep — S1 frozen vs S2 adapters vs S2 from-scratch

Balanced accuracy unless noted; period chance ≈ .17 (6 classes ≥ 30 docs), source chance ≈ .17. A high SOURCE probe with a low WITHIN-source period probe means the model learned corpora, not time.

## How to read this (2026-09-03)

**Within the SSL corpora the period is nearly free, and it is nearly the same question as the source.** Cells read the period at .81-.90 and the SOURCE at .92-.98, because here one implies the other (Old Babylonian = Archibab, Late Babylonian = the letters, Hellenistic = ORACC). The high period numbers, the k-NN purity and the UMAP silhouette are therefore not evidence that anything chronological was learned.

**The `HELD-OUT dated` column is NOT usable; it is kept so the mistake stays on the record.** It scores balanced accuracy over the periods the dated royal inscriptions share with the undated pool — Neo-Assyrian (924 test documents), Middle Babylonian (28) and Hellenistic (ONE) — while the pool training the probe holds 52 Middle Babylonian and 5 Neo-Babylonian texts. Averaging three such classes is what produced ".10-.20, below chance": an artefact of the class filter, not a finding about time. The 216 Neo-Babylonian inscriptions, the second largest group in the test set, were dropped from it entirely. The read-out that replaces it is C17 (`ssl/TRANSFER_DATED.md`): fit against an approximate period midpoint on the undated corpora, then Spearman against the true year of the dated inscriptions.

## Main table

| kind | model | lin period | mlp period | HELD-OUT dated | lin source | lin genre | lin provenance | silhouette_period | knn10_purity_period |
|---|---|---|---|---|---|---|---|---|---|
| adapter (SSL on frozen) | `ssl::ssl_barlow_cunei400m-s0::L0::h` | 0.901 | 0.882 | 0.161 | 0.971 | 0.262 | 0.676 | 0.091 | 0.984 |
| adapter (SSL on frozen) | `ssl::ssl_barlow_llama2_7b-s0::L0::h` | 0.862 | 0.839 | 0.115 | 0.978 | 0.228 | 0.613 | 0.195 | 0.979 |
| adapter (SSL on frozen) | `ssl::ssl_barlow_qwen3_8b-s0::L0::h` | 0.820 | 0.801 | 0.096 | 0.964 | 0.206 | 0.586 | 0.197 | 0.966 |
| adapter (SSL on frozen) | `ssl::ssl_byol_akk300m-s0::L0::h` | 0.885 | 0.848 | 0.143 | 0.935 | 0.188 | 0.492 | 0.302 | 0.974 |
| adapter (SSL on frozen) | `ssl::ssl_byol_cunei400m-s0::L0::h` | 0.840 | 0.815 | 0.159 | 0.918 | 0.129 | 0.391 | 0.192 | 0.903 |
| adapter (SSL on frozen) | `ssl::ssl_jepa_cunei400m-s0::L0::h` | 0.809 | 0.766 | 0.146 | 0.924 | 0.136 | 0.447 | 0.175 | 0.890 |
| from-scratch | `ssl_e2e::e2e_barlow_M-s0::L0::h` | 0.886 | 0.866 | 0.125 | 0.972 | 0.261 | 0.634 | 0.222 | 0.981 |
| from-scratch | `ssl_e2e::e2e_barlow_S-s0::L0::h` | 0.871 | 0.885 | 0.203 | 0.966 | 0.245 | 0.649 | 0.190 | 0.981 |
| from-scratch | `ssl_e2e::e2e_jepa_L-s0::L0::h` | 0.870 | 0.839 | 0.104 | 0.951 | 0.278 | 0.664 | 0.060 | 0.959 |
| from-scratch | `ssl_e2e::e2e_jepa_S-s0::L0::h` | 0.848 | 0.824 | 0.121 | 0.900 | 0.241 | 0.616 | 0.019 | 0.920 |
| from-scratch | `ssl_e2e::e2e_jepa_XL-s0::L0::h` | 0.855 | 0.822 | 0.106 | 0.946 | 0.265 | 0.618 | 0.068 | 0.949 |
| frozen encoder | `Qwen/Qwen3-8B::L18::mean` | 0.838 | 0.816 |  | 0.963 | 0.234 | 0.633 | 0.052 | 0.957 |
| frozen encoder | `Qwen/Qwen3-8B::L27::mean` | 0.856 | 0.798 |  | 0.963 | 0.249 | 0.655 | 0.080 | 0.964 |
| frozen encoder | `Thalesian/AKK_300m::L4::mean` | 0.841 | 0.825 |  | 0.961 | 0.273 | 0.660 | 0.096 | 0.980 |
| frozen encoder | `Thalesian/AKK_300m::L8::mean` | 0.865 | 0.816 |  | 0.939 | 0.261 | 0.650 | 0.113 | 0.969 |
| frozen encoder | `Thalesian/cuneiformBase-400m::L12::mean` | 0.890 | 0.823 |  | 0.963 | 0.273 | 0.674 | 0.092 | 0.981 |
| frozen encoder | `Thalesian/cuneiformBase-400m::L6::mean` | 0.901 | 0.843 |  | 0.967 | 0.278 | 0.689 | 0.138 | 0.983 |

## Period probe within source / with a source held out (linear)

| model | within_oracc | within_seal | heldout_orcc | heldout_seal |
|---|---|---|---|---|
| `ssl::ssl_barlow_cunei400m-s0::L0::h` | 0.992 | 0.802 | 0.161 | 0.676 |
| `ssl::ssl_barlow_llama2_7b-s0::L0::h` | 0.996 | 0.722 | 0.115 | 0.502 |
| `ssl::ssl_barlow_qwen3_8b-s0::L0::h` | 0.977 | 0.653 | 0.096 | 0.597 |
| `ssl::ssl_byol_akk300m-s0::L0::h` | 0.992 | 0.791 | 0.143 | 0.664 |
| `ssl::ssl_byol_cunei400m-s0::L0::h` | 0.980 | 0.807 | 0.159 | 0.643 |
| `ssl::ssl_jepa_cunei400m-s0::L0::h` | 0.964 | 0.700 | 0.146 | 0.661 |
| `ssl_e2e::e2e_barlow_M-s0::L0::h` | 0.988 | 0.817 | 0.125 | 0.501 |
| `ssl_e2e::e2e_barlow_S-s0::L0::h` | 0.989 | 0.793 | 0.203 | 0.622 |
| `ssl_e2e::e2e_jepa_L-s0::L0::h` | 0.991 | 0.765 | 0.104 | 0.603 |
| `ssl_e2e::e2e_jepa_S-s0::L0::h` | 0.959 | 0.725 | 0.121 | 0.502 |
| `ssl_e2e::e2e_jepa_XL-s0::L0::h` | 0.988 | 0.677 | 0.106 | 0.514 |
| `Qwen/Qwen3-8B::L18::mean` | 0.988 | 0.697 |  |  |
| `Qwen/Qwen3-8B::L27::mean` | 0.990 | 0.761 |  |  |
| `Thalesian/AKK_300m::L4::mean` | 0.992 | 0.795 |  |  |
| `Thalesian/AKK_300m::L8::mean` | 0.991 | 0.793 |  |  |
| `Thalesian/cuneiformBase-400m::L12::mean` | 0.993 | 0.775 |  |  |
| `Thalesian/cuneiformBase-400m::L6::mean` | 0.995 | 0.768 |  |  |

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
