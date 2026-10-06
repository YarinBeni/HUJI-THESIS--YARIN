# Length control, model-level — 'aaaa' texts through the encoder

Every non-space character replaced by 'a'; only length and word boundaries survive. Clustered at each model's Fig-B peak layer.

| model | layer | ARI real (orig) | ARI on a-texts | shuffle null | ARI(a, real) akk | eng |
|---|---|---|---|---|---|---|

## Label-level control, bin sweep

ARI between akk and eng length-quantile bins of the same documents.

| q bins | ARI |
|---|---|
| 4 | 0.805 |
| 8 | 0.653 |
| 16 | 0.469 |
