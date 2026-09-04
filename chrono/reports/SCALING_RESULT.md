# Representation sweep — S1 frozen vs S2 adapters vs S2 from-scratch

Balanced accuracy unless noted; period chance ≈ .17 (6 classes ≥ 30 docs), source chance ≈ .17. A high SOURCE probe with a low WITHIN-source period probe means the model learned corpora, not time.

## How to read this (2026-09-03)

**Within the SSL corpora the period is nearly free, and it is nearly the same question as the source.** Cells read the period at .81-.90 and the SOURCE at .92-.98, because here one implies the other (Old Babylonian = Archibab, Late Babylonian = the letters, Hellenistic = ORACC). The high period numbers, the k-NN purity and the UMAP silhouette are therefore not evidence that anything chronological was learned.

**The `HELD-OUT dated` column is NOT usable; it is kept so the mistake stays on the record.** It scores balanced accuracy over the periods the dated royal inscriptions share with the undated pool — Neo-Assyrian (924 test documents), Middle Babylonian (28) and Hellenistic (ONE) — while the pool training the probe holds 52 Middle Babylonian and 5 Neo-Babylonian texts. Averaging three such classes is what produced ".10-.20, below chance": an artefact of the class filter, not a finding about time. The 216 Neo-Babylonian inscriptions, the second largest group in the test set, were dropped from it entirely. The read-out that replaces it is C17 (`ssl/TRANSFER_DATED.md`): fit against an approximate period midpoint on the undated corpora, then Spearman against the true year of the dated inscriptions.

## Main table

| kind | model | lin period | mlp period | HELD-OUT dated | lin source | lin genre | lin provenance | silhouette_period | knn10_purity_period |
|---|---|---|---|---|---|---|---|---|---|
| adapter (SSL on frozen) | `ssl::ssl_barlow_akk300m-s0::L0::h` | 0.871 | 0.865 | 0.205 | 0.947 | 0.258 | 0.623 | 0.086 | 0.979 |
| adapter (SSL on frozen) | `ssl::ssl_barlow_cunei400m-s0::L0::h` | 0.901 | 0.882 | 0.161 | 0.971 | 0.262 | 0.676 | 0.091 | 0.984 |
| adapter (SSL on frozen) | `ssl::ssl_barlow_llama2_7b-s0::L0::h` | 0.862 | 0.839 | 0.115 | 0.978 | 0.228 | 0.613 | 0.195 | 0.979 |
| adapter (SSL on frozen) | `ssl::ssl_barlow_qwen3_8b-s0::L0::h` | 0.820 | 0.801 | 0.096 | 0.964 | 0.206 | 0.586 | 0.197 | 0.966 |
| adapter (SSL on frozen) | `ssl::ssl_byol_akk300m-s0::L0::h` | 0.885 | 0.848 | 0.143 | 0.935 | 0.188 | 0.492 | 0.302 | 0.974 |
| adapter (SSL on frozen) | `ssl::ssl_byol_cunei400m-s0::L0::h` | 0.840 | 0.815 | 0.159 | 0.918 | 0.129 | 0.391 | 0.192 | 0.903 |
| adapter (SSL on frozen) | `ssl::ssl_byol_llama2_7b-s0::L0::h` | 0.836 | 0.825 | 0.141 | 0.962 | 0.179 | 0.502 | 0.336 | 0.956 |
| adapter (SSL on frozen) | `ssl::ssl_byol_qwen3_8b-s0::L0::h` | 0.797 | 0.781 | 0.098 | 0.939 | 0.157 | 0.469 | 0.239 | 0.939 |
| adapter (SSL on frozen) | `ssl::ssl_infonce_akk300m-s0::L0::h` | 0.879 | 0.841 | 0.145 | 0.948 | 0.249 | 0.603 | 0.018 | 0.972 |
| adapter (SSL on frozen) | `ssl::ssl_infonce_cunei400m-s0::L0::h` | 0.891 | 0.870 | 0.101 | 0.960 | 0.247 | 0.637 | 0.025 | 0.976 |
| adapter (SSL on frozen) | `ssl::ssl_infonce_llama2_7b-s0::L0::h` | 0.852 | 0.829 | 0.120 | 0.973 | 0.246 | 0.620 | -0.056 | 0.973 |
| adapter (SSL on frozen) | `ssl::ssl_infonce_qwen3_8b-s0::L0::h` | 0.852 | 0.823 | 0.094 | 0.968 | 0.234 | 0.632 | -0.025 | 0.975 |
| adapter (SSL on frozen) | `ssl::ssl_jepa_akk300m-s0::L0::h` | 0.879 | 0.872 | 0.148 | 0.943 | 0.203 | 0.515 | 0.308 | 0.980 |
| adapter (SSL on frozen) | `ssl::ssl_jepa_cunei400m-s0::L0::h` | 0.809 | 0.766 | 0.146 | 0.924 | 0.136 | 0.447 | 0.175 | 0.890 |
| adapter (SSL on frozen) | `ssl::ssl_jepa_llama2_7b-s0::L0::h` | 0.848 | 0.838 | 0.105 | 0.969 | 0.201 | 0.544 | 0.277 | 0.969 |
| adapter (SSL on frozen) | `ssl::ssl_jepa_qwen3_8b-s0::L0::h` | 0.809 | 0.802 | 0.067 | 0.949 | 0.167 | 0.493 | 0.082 | 0.942 |
| from-scratch | `ssl_e2e::e2e_barlow_L-s0::L0::h` | 0.878 | 0.851 | 0.123 | 0.972 | 0.265 | 0.672 | 0.233 | 0.983 |
| from-scratch | `ssl_e2e::e2e_barlow_M-s0::L0::h` | 0.886 | 0.866 | 0.125 | 0.972 | 0.261 | 0.634 | 0.222 | 0.981 |
| from-scratch | `ssl_e2e::e2e_barlow_S-s0::L0::h` | 0.871 | 0.885 | 0.203 | 0.966 | 0.245 | 0.649 | 0.190 | 0.981 |
| from-scratch | `ssl_e2e::e2e_barlow_XL-s0::L0::h` | 0.856 | 0.852 | 0.282 | 0.972 | 0.260 | 0.659 | 0.163 | 0.982 |
| from-scratch | `ssl_e2e::e2e_jepa_L-s0::L0::h` | 0.870 | 0.839 | 0.104 | 0.951 | 0.278 | 0.664 | 0.060 | 0.959 |
| from-scratch | `ssl_e2e::e2e_jepa_M-s0::L0::h` | 0.863 | 0.817 | 0.188 | 0.952 | 0.279 | 0.653 | 0.052 | 0.953 |
| from-scratch | `ssl_e2e::e2e_jepa_S-s0::L0::h` | 0.848 | 0.824 | 0.121 | 0.900 | 0.241 | 0.616 | 0.019 | 0.920 |
| from-scratch | `ssl_e2e::e2e_jepa_XL-s0::L0::h` | 0.855 | 0.822 | 0.106 | 0.946 | 0.265 | 0.618 | 0.068 | 0.949 |
| frozen encoder | `Qwen/Qwen3-8B::L18::mean` | 0.838 | 0.816 |  | 0.963 | 0.234 | 0.633 | 0.052 | 0.957 |
| frozen encoder | `Qwen/Qwen3-8B::L27::mean` | 0.856 | 0.798 |  | 0.963 | 0.249 | 0.655 | 0.080 | 0.964 |
| frozen encoder | `Thalesian/AKK_300m::L4::mean` | 0.841 | 0.825 |  | 0.961 | 0.273 | 0.660 | 0.096 | 0.980 |
| frozen encoder | `Thalesian/AKK_300m::L8::mean` | 0.865 | 0.816 |  | 0.939 | 0.261 | 0.650 | 0.113 | 0.969 |
| frozen encoder | `Thalesian/cuneiformBase-400m::L12::mean` | 0.890 | 0.823 | 0.179 | 0.963 | 0.273 | 0.674 | 0.092 | 0.981 |
| frozen encoder | `Thalesian/cuneiformBase-400m::L6::mean` | 0.901 | 0.843 |  | 0.967 | 0.278 | 0.689 | 0.138 | 0.983 |

## Period probe within source / with a source held out (linear)

| model | within_oracc | within_seal | heldout_orcc | heldout_seal |
|---|---|---|---|---|
| `ssl::ssl_barlow_akk300m-s0::L0::h` | 0.996 | 0.765 | 0.205 | 0.662 |
| `ssl::ssl_barlow_cunei400m-s0::L0::h` | 0.992 | 0.802 | 0.161 | 0.676 |
| `ssl::ssl_barlow_llama2_7b-s0::L0::h` | 0.996 | 0.722 | 0.115 | 0.502 |
| `ssl::ssl_barlow_qwen3_8b-s0::L0::h` | 0.977 | 0.653 | 0.096 | 0.597 |
| `ssl::ssl_byol_akk300m-s0::L0::h` | 0.992 | 0.791 | 0.143 | 0.664 |
| `ssl::ssl_byol_cunei400m-s0::L0::h` | 0.980 | 0.807 | 0.159 | 0.643 |
| `ssl::ssl_byol_llama2_7b-s0::L0::h` | 0.975 | 0.707 | 0.141 | 0.626 |
| `ssl::ssl_byol_qwen3_8b-s0::L0::h` | 0.931 | 0.682 | 0.098 | 0.623 |
| `ssl::ssl_infonce_akk300m-s0::L0::h` | 0.994 | 0.761 | 0.145 | 0.642 |
| `ssl::ssl_infonce_cunei400m-s0::L0::h` | 0.993 | 0.767 | 0.101 | 0.634 |
| `ssl::ssl_infonce_llama2_7b-s0::L0::h` | 0.993 | 0.753 | 0.120 | 0.422 |
| `ssl::ssl_infonce_qwen3_8b-s0::L0::h` | 0.990 | 0.722 | 0.094 | 0.451 |
| `ssl::ssl_jepa_akk300m-s0::L0::h` | 0.988 | 0.810 | 0.148 | 0.643 |
| `ssl::ssl_jepa_cunei400m-s0::L0::h` | 0.964 | 0.700 | 0.146 | 0.661 |
| `ssl::ssl_jepa_llama2_7b-s0::L0::h` | 0.981 | 0.712 | 0.105 | 0.573 |
| `ssl::ssl_jepa_qwen3_8b-s0::L0::h` | 0.954 | 0.679 | 0.067 | 0.477 |
| `ssl_e2e::e2e_barlow_L-s0::L0::h` | 0.987 | 0.828 | 0.123 | 0.360 |
| `ssl_e2e::e2e_barlow_M-s0::L0::h` | 0.988 | 0.817 | 0.125 | 0.501 |
| `ssl_e2e::e2e_barlow_S-s0::L0::h` | 0.989 | 0.793 | 0.203 | 0.622 |
| `ssl_e2e::e2e_barlow_XL-s0::L0::h` | 0.987 | 0.782 | 0.282 | 0.378 |
| `ssl_e2e::e2e_jepa_L-s0::L0::h` | 0.991 | 0.765 | 0.104 | 0.603 |
| `ssl_e2e::e2e_jepa_M-s0::L0::h` | 0.991 | 0.749 | 0.188 | 0.552 |
| `ssl_e2e::e2e_jepa_S-s0::L0::h` | 0.959 | 0.725 | 0.121 | 0.502 |
| `ssl_e2e::e2e_jepa_XL-s0::L0::h` | 0.988 | 0.677 | 0.106 | 0.514 |
| `Qwen/Qwen3-8B::L18::mean` | 0.988 | 0.697 |  |  |
| `Qwen/Qwen3-8B::L27::mean` | 0.990 | 0.761 |  |  |
| `Thalesian/AKK_300m::L4::mean` | 0.992 | 0.795 |  |  |
| `Thalesian/AKK_300m::L8::mean` | 0.991 | 0.793 |  |  |
| `Thalesian/cuneiformBase-400m::L12::mean` | 0.993 | 0.775 | 0.179 | 0.668 |
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
