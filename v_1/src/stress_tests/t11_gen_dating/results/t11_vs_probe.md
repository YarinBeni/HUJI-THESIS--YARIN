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
| qwen3_32b | maxking | — | 0.760* | 20/1202 | 0.98 | 0.000 | — (0) | -0.364 (20) |
| qwen3_32b | engtier0 | 0.733±0.06 | 0.437 | 746/1202 | 0.37 | 0.665 | 0.870 (496) | 0.360 (250) |
| gpt_oss_120b | tier0 | -0.024±0.07 | 0.388 | 876/1202 | 0.25 | 0.039 | 0.662 (34) | -0.045 (842) |
| gpt_oss_120b | maximal | 0.026±0.08 | 0.316 | 618/1202 | 0.48 | 0.055 | 0.622 (34) | -0.058 (584) |
| gpt_oss_120b | maxking | -0.007±0.07 | 0.781* | 680/1202 | 0.42 | 0.051 | 0.903 (35) | -0.142 (645) |
| gpt_oss_120b | engtier0 | 0.537±0.07 | 0.366 | 1013/1202 | 0.14 | 0.587 | 0.845 (595) | 0.160 (418) |
