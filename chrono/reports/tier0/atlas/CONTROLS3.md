# Controls wave 3 — mask-only alignment, order-test diagnostics

Stored view conditions (augs x lang x count):
```
a                         lang
                          akk     2386
                          eng     2386
mask_ruler                akk     2386
                          eng     2386
mask_ruler,crop16         akk     2386
                          eng     2386
mask_ruler,crop32         akk     2386
                          eng     2386
mask_ruler,drop_span      akk     2386
                          eng     2386
mask_ruler,strip_formula  akk     2386
                          eng     2386
orthonorm                 akk     2386
                          eng     2386
strip_formula             akk     2386
                          eng     2386
```

## V1b — language alignment on [mask_ruler], full text

| model | layer | ARI orig | ARI mask-only | shuffle null |
|---|---|---|---|---|
| `Llama-2-7b-hf` | 20 | 0.903 | 0.855 | -0.000 |
| `Qwen3-8B` | 27 | 0.532 | 0.134 | -0.002 |
| `cuneiformBase-400m` | 4 | 0.308 | 0.014 | +0.001 |
| `OLMo-2-1124-7B` | 9 | 0.875 | 0.843 | +0.008 |

## V3b — order-test diagnostics per century

Only centuries with >= 60 docs and >= 5 distinct years were tested in wave 2. What do they hold?

| model | century (BC) | n docs | distinct years | majority-year share | rho |
|---|---|---|---|---|---|
| `Llama-2-7b-hf` | 7 | 240 | 8 | 60% | 0.493 |
| `Llama-2-7b-hf` | 6 | 728 | 17 | 37% | 0.659 |
| `Llama-2-7b-hf` | 5 | 170 | 5 | 51% | 0.743 |
| `Qwen3-8B` | 7 | 240 | 8 | 60% | 0.534 |
| `Qwen3-8B` | 6 | 732 | 17 | 37% | 0.613 |
| `Qwen3-8B` | 5 | 171 | 5 | 51% | 0.703 |
| `cuneiformBase-400m` | 7 | 240 | 8 | 60% | 0.586 |
| `cuneiformBase-400m` | 6 | 732 | 17 | 37% | 0.709 |
| `cuneiformBase-400m` | 5 | 171 | 5 | 51% | 0.808 |
| `OLMo-2-1124-7B` | 7 | 240 | 8 | 60% | 0.538 |
| `OLMo-2-1124-7B` | 6 | 728 | 17 | 37% | 0.659 |
| `OLMo-2-1124-7B` | 5 | 170 | 5 | 51% | 0.757 |
