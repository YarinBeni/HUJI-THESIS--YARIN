# W — world-models replication (Gurnee & Tegmark space/time probes on our ladder)

See **PLAN.md** for the full design (why, model ladder, protocol, risks). This file is
the operator's guide.

## What this section does

Re-runs the probing experiments of *"Language Models Represent Space and Time"*
(arXiv:2310.02207) — six English entity datasets, per-layer ridge probes for
coordinates / years — with the thesis model ladder (Qwen3 1.7B/8B/32B, gpt-oss-120B,
AKK-300m, cunei-400m, uMT5-base, TF-IDF, random) **plus** trained *and random-init*
Llama-2 7B/13B/70B. Trained Llama-2 anchors our harness to the paper's reported
numbers; the random-init arms are the control the paper never ran.

## Run order (cluster)

```bash
# once, before the first submission (logs/ is gitignored by the repo-wide **/logs/ rule
# and SLURM needs the --output dir to exist at job start):
mkdir -p v_1/src/world_models/logs

# 0. once: materialize random-init Llama-2-70B (seed 42, ~130GB cluster-local)
sbatch v_1/src/world_models/sbatch/W0_build_random_llama70b.sbatch

# 1. extraction (parallel; W1d task 1 needs W0 done first)
sbatch v_1/src/world_models/sbatch/W1_extract.sbatch          # 7 gpu:1 arms
sbatch v_1/src/world_models/sbatch/W1b_extract_gptoss.sbatch  # gpu:8
sbatch v_1/src/world_models/sbatch/W1c_extract_llama.sbatch   # 7B/13B ± random
sbatch --dependency=afterok:<W0-jobid> v_1/src/world_models/sbatch/W1d_extract_llama70b.sbatch

# 2. probes (CPU; per-arm as soon as its extraction landed, e.g. --array=10 for llama2_70b)
sbatch v_1/src/world_models/sbatch/W2_probe.sbatch
sbatch v_1/src/world_models/sbatch/W2b_tfidf.sbatch

# 3. tables + figures (rerunnable any time)
sbatch v_1/src/world_models/sbatch/W3_aggregate.sbatch
```

Local smoke test (no GPU / no downloads): `python extract_acts.py --method <m> --limit 500`
after models are cached, or see the offline path used in development (tiny from-config
model through the same wm_lib functions; pooling verified against manual forwards).

## Notes

* **Llama gating**: set `HF_TOKEN` for meta-llama, or do nothing — the loader falls
  back to the ungated NousResearch mirrors. `WM_LLAMA_ORG` overrides the org.
* **Disk**: npz activations are gitignored/cluster-local; the full trained set is
  ~450 GB (llama2_70b 122 GB, qwen3_32b 61 GB the biggest). W2 deletes the random
  arms' npz after probing (`CLEANUPS` array in W2_probe.sbatch — flip to taste).
* **Committed artifacts**: per-extraction `metadata.json`, probe JSONs
  (`results/probes/`), best-layer projections (`results/projections/`), summary
  CSVs + `results/RESULTS.md` + `results/figs/`.
* The `pythia_70m_test` registry entry is a debug arm (excluded from tables).
* Datasets are vendored under `data/entity_datasets/` (verify: `python fetch_data.py`);
  entity strings and the `is_test` split are byte-faithful ports of the paper repo.

## Reading the results

`results/RESULTS.md` — best-layer test R² per arm × dataset with the paper's
Llama-2-70B row on top. The three headline comparisons:
1. `llama2_70b` vs paper row (harness validation),
2. every `*_random` vs its trained twin vs `tfidf` (how much is *learned* world
   structure vs architecture prior vs surface form),
3. Qwen3/gpt-oss vs Llama-2 at matched scale, and the encoders vs `tfidf`
   (cross-lingual control for the thesis story).
