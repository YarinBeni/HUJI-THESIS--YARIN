# The language cliff at the NAME level (WB-akk, 2026-10-06)

The thesis showed LLMs date **entities** well and **documents** poorly. This
experiment separates two candidate causes of that cliff — the language/script
of the surface form vs. span/aggregation — by probing the SAME rulers under
the SAME templates and fold-matched splits, with only the name form changed:
"Sennacherib" vs. his attested transliterated spelling ("m-d-30-PAP-MEŠ-SU",
harvested from the titulary of his own inscriptions).

MC Spearman rho, bare name, ent_last, best layer. `eng25` = the English probe
restricted to exactly the rulers whose Akkadian spelling survived the harvest
(21 shared rulers), so eng25 vs. akk is language-only.

| arm | eng (40 rulers) | eng25 (matched) | akk names | **cliff (eng25 − akk)** |
|---|---|---|---|---|
| Llama-2-7B | .527 | .519 | .138 | **.38** |
| Qwen3-8B | .596 | .684 | .252 | **.43** |
| OLMo-2-7B | .526 | .643 | .129 | **.51** |
| cuneiformBase-400m | .456 | .415 | .451 | **−.04** |
| gpt-oss-120B | .663 | — | pending (C23b) | — |

## What it says

1. **Three independent LLM families lose 60–80 % of their entity-dating
   signal when the king's name is written in his own script.** Their
   chronological knowledge is bound to the English surface form.
2. **The domain encoder has no cliff at all** (−.04): cunei400m knows the
   ruler in either orthography — a lower ceiling, but script-invariant.
3. **The thesis's entity→document cliff therefore decomposes differently per
   model class**: for LLMs it is language + aggregation; for the domain
   encoder, aggregation only. This also explains why cunei400m beats the
   7B–8B LLMs on Akkadian *documents* (E-MIN v2, C18) while losing to them
   on English *names*.
4. Pooling note: with `ent_mean` over the name span the Akkadian numbers
   recover substantially (Llama .36, Qwen .44, cunei .51) — the last token
   of a long transliterated name is a poor summary; part of the cliff is
   read-out, not knowledge. The cliff survives but narrows.

## Caveats

- 25 rulers carry a surviving spelling (21 shared with the English split);
  MC sd in the akk cells is large (±.4–.5). The ranking is stable, the exact
  magnitudes are not.
- R² is negative throughout the akk cells (high-dim ridge, small n); rho is
  the readable statistic.
- Spellings were harvested from the rulers' own inscriptions (titulary =
  first PN per document); a residual minority of non-royal openings remains
  (e.g. 12 Teumman-opening documents under Ashurbanipal).
