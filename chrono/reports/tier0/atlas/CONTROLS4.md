# Length control, model-level — 'aaaa' texts through the encoder

Every non-space character replaced by 'a'; only length and word boundaries survive. Clustered at each model's Fig-B peak layer.

| model | layer | ARI real (orig) | ARI on a-texts | shuffle null | ARI(a, real) akk | eng |
|---|---|---|---|---|---|---|
| `Llama-2-7b-hf` | 20 | 0.903 | 0.318 | -0.001 | 0.347 | -0.063 |
| `Qwen3-8B` | 27 | 0.532 | 0.027 | -0.003 | 0.247 | 0.277 |
| `cuneiformBase-400m` | 4 | 0.308 | 0.210 | -0.001 | 0.323 | 0.632 |
| `OLMo-2-1124-7B` | 9 | 0.875 | -0.008 | -0.002 | -0.008 | 0.087 |

## Label-level control, bin sweep

ARI between akk and eng length-quantile bins of the same documents.

| q bins | ARI |
|---|---|
| 4 | 0.805 |
| 8 | 0.653 |
| 16 | 0.469 |
