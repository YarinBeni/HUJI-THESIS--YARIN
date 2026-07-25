# Cell B — entity-level results (obscure entities, English)

Protocol: 200-draw Monte-Carlo over **entity-level** splits (20% of
entities held out per draw); ridge, best layer. `bare` = the entity
string alone (paper-faithful); `all` = plus five carrier sentences.
A score witnesses learning only if it beats **both** the TF-IDF floor
**and** the arm's own random-init twin.

## assyrian_ruler — rows=bare

| arm | site | best layer | MC R2 | +/- | MC rho | +/- |
|---|---|--:|--:|--:|--:|--:|
| *tfidf* | text | 0 | -0.345 | 1.461 | 0.344 | 0.388 |

## assyrian_ruler — rows=all

| arm | site | best layer | MC R2 | +/- | MC rho | +/- |
|---|---|--:|--:|--:|--:|--:|
| *tfidf* | text | 0 | -0.293 | 1.478 | 0.340 | 0.374 |

## mesopotamian_place — rows=bare

| arm | site | best layer | MC R2 | +/- | MC rho | +/- |
|---|---|--:|--:|--:|--:|--:|
| *tfidf* | text | 0 | -0.620 | 1.399 | 0.296 | 0.304 |

## mesopotamian_place — rows=all

| arm | site | best layer | MC R2 | +/- | MC rho | +/- |
|---|---|--:|--:|--:|--:|--:|
| *tfidf* | text | 0 | -0.531 | 0.945 | 0.257 | 0.285 |
