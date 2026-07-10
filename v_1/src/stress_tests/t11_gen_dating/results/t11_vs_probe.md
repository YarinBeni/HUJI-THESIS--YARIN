# T11 — generated-answer dating vs activation probe (mean site)

Answer = Spearman of the model's parsed year answers over the SAME 200
balanced 8x21 draws as every probe. Probe = PLS best-k MC Spearman on the
mean-pooled activations (\*maxking probe uses its own 5x9 StratifiedKFold
protocol — indicative only). named/unnamed = full-corpus Spearman
conditioned on the answer text naming the true ruler.

| model | cleaning | answer MC | probe MC | scoreable | declined | named rate | ρ named (n) | ρ unnamed (n) |
|---|---|---|---|---|---|---|---|---|
| qwen3_1b7 | tier0 | — | 0.352 | 15/1202 | 0.99 | 0.000 | — (0) | 0.344 (15) |
| qwen3_1b7 | maximal | — | 0.334 | 0/1202 | 1.00 | — | — (0) | — (0) |
| qwen3_1b7 | maxking | — | 0.720* | 0/1202 | 1.00 | — | — (0) | — (0) |
| qwen3_1b7 | engtier0 | 0.436±0.19 | 0.368 | 76/1202 | 0.94 | 0.000 | — (0) | 0.550 (76) |
| qwen3_8b | tier0 | -0.266±0.06 | 0.348 | 1160/1202 | 0.01 | 0.087 | -0.182 (101) | -0.201 (1059) |
| qwen3_8b | maximal | -0.264±0.06 | 0.339 | 946/1202 | 0.21 | 0.128 | -0.405 (121) | -0.255 (825) |
| qwen3_8b | maxking | -0.265±0.05 | 0.715* | 1010/1202 | 0.15 | 0.164 | -0.393 (166) | -0.158 (844) |
| qwen3_8b | engtier0 | 0.520±0.07 | 0.416 | 996/1202 | 0.15 | 0.453 | 0.876 (451) | 0.169 (545) |
| qwen3_32b | tier0 | -0.232±0.26 | 0.381 | 113/1202 | 0.87 | 0.035 | — (4) | 0.008 (109) |
| qwen3_32b | maximal | — | 0.332 | 18/1202 | 0.98 | 0.000 | — (0) | -0.303 (18) |
| qwen3_32b | maxking | PENDING | | | | | | |
| qwen3_32b | engtier0 | PENDING | | | | | | |
| gpt_oss_120b | tier0 | PENDING | | | | | | |
| gpt_oss_120b | maximal | PENDING | | | | | | |
| gpt_oss_120b | maxking | PENDING | | | | | | |
| gpt_oss_120b | engtier0 | PENDING | | | | | | |
