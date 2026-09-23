# Q7 Large-scale Efficiency Analysis

This directory contains the bounded, LinearRAG-only cold-index experiment. It uses deterministic prefixes of the local HotpotQA corpus as scale proxies; it does not reproduce the paper's ATLAS-Wiki data or its RAPTOR and HippoRAG baselines.

## Current results

The 25K, 50K, 250K, and 500K token runs completed. Their raw measurements are in:

- `data/output/experiment_results/q7_large_scale_efficiency/q7-design-check-25k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-design-check-50k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-formal-250k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-formal-500k-r2-20260921/`

The 1M-token attempt did not complete during NER and has no usable measurement. It does not establish performance at 1M tokens. See `project/agent/experiments/q7_large_scale_efficiency/RESULTS_2026-09-21_SCALE_SERIES.md` for the observed results and limits.

## Measurement design

Token counts use the local `all-mpnet-base-v2` tokenizer. Each completed invocation created an isolated cold cache, called the existing `LinearRAG.index()` implementation, recorded indexing and graph measurements, and guarded against generative LLM calls. Retrieval, answer generation, and answer evaluation are outside this experiment.

Run a small design check from the repository root with:

```powershell
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 25000 --experiment-id q7-design-check-25k
```

Use a new experiment ID for each run. Do not reuse the IDs listed above, which identify completed evidence.
