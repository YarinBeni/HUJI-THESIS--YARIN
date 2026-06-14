# Pillar 1 — Thalesian autopsy (1a + 1b)

**Question.** Thalesian (`google/umt5-base` + cuneiform finetune) beats much larger
Qwen3 / gpt-oss on the **mean-balanced-maximal** PLS dating of ORCC. Which of
**T** (tokenizer) / **A** (architecture) / **O** (objective) / **F** (finetune) carries
the win — and what should we finetune next? Scope here is **1a + 1b** (no training).

## 1a — Tokenization audit (CPU) → isolates (T)
`tokenization_audit.py` → `results/tokenization_audit.{json,csv}`, `results/figures/*.png`.
Per tokenizer (Thalesian/uMT5, Qwen3-8B, gpt-oss-120b), **per corpus**
(orcc, seal, letters, archibab, oracc_1mill, ebl): fertility (tokens/Akkadian-word),
UNK rate, byte-fallback rate, isolated-special-char probe, ORCC orthographic-category
split (logogram/determinative/diacritic/index/plain) and tier0-vs-maximal detail.
Multi-corpus so a tokenizer advantage is shown to be *general*, not royal-specific.
Thalesian's vocab was **not** expanded vs uMT5, so the audit also *proves* their
tokenizers are identical — which is why (T) is held constant in the 1b F-comparison.

## 1b — Control-ladder probe (1 GPU to extract, then CPU) → isolates (F) and (A)+(T)
No training. Extract **vanilla `google/umt5-base`** encoder activations on ORCC
(tier0+maximal, **mean**, all layers) with the *same* seq2seq-encoder extractor that
produced the on-disk Thalesian activations
(`linear_probing/round2_phase3/extract_enc_activations.py`), then probe the whole
ladder under the **identical** maximal-balanced PLS via
`linear_probing/round2_phase0/run_mc_probes.py` (200 MC balanced ruler draws).
`build_ladder_table.py` → `results/ladder_table.csv` + decision reads.

| Comparison | Holds constant | Isolates |
|---|---|---|
| Thalesian vs vanilla uMT5 | tokenizer, architecture, objective-family | **(F)** the cuneiform finetune |
| vanilla uMT5 vs Qwen3-8B  | neither cuneiform-finetuned | **(A)+(T)** enc-dec/bidirectional + tokenizer bundle |
| 1a fertility | — | **(T)** descriptively, to split (A) from (T) |

**Decision regime = maximal + mean** (where Thalesian wins, Spearman≈0.41). tier0+mean
is a confirmatory robustness column; last-token pooling is not used (we mean-pool the
uMT5 encoder; the whole on-disk ladder is mean).

```
Thalesian ≈ uMT5   -> win is the BASE model (arch/tokenizer/pretraining), not the finetune
Thalesian >> uMT5  -> the cuneiform finetune does the work -> go to 1c (which objective?)
uMT5 >> Qwen       -> enc-dec/bidirectional+multilingual base matters (A/T)
uMT5 ≈ Qwen        -> base arch is not it; the story is the cuneiform finetune (F/O)
```

## Run (on the cluster — agent never SSHes)
```
cd ~/projects/HUJI-THESIS--YARIN && git pull && sbatch v_1/src/chronorank/autopsy/sbatch/P1a_tokenization.sbatch
cd ~/projects/HUJI-THESIS--YARIN && git pull && sbatch v_1/src/chronorank/autopsy/sbatch/P1b_umt5_probe.sbatch
```

## Status — COMPLETE (see FINDINGS_1ab.md)
- [x] P1a done — **(T) rejected**: uMT5/Thalesian tokenizers are the *least* efficient on Akkadian.
- [x] P1b done (job 9661) — **(A) rejected**: vanilla uMT5 (0.297) = random floor (0.301) < Qwen (0.366).
- [x] **Verdict: (F) the cuneiform finetune** — Thalesian 0.413 vs uMT5 0.297 (Δ+0.116, maximal).
      Next finetune = **seq2seq/translation objective** (1c), since Round-3 NTP finetune was flat.
- [ ] 1c objective ablation — ON HOLD (needs Akkadian→English translation data).

(1c objective ablation and 1d English diagnostic are ON HOLD — need Akkadian→English data.)
