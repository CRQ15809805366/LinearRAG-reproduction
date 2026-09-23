# Q7 Large-scale Efficiency Analysis

## Status and evidence

Four isolated cold-index runs completed on 2026-09-21: 25K, 50K, 250K, and 500K input tokens. Their measurements and run metadata have `status=passed`, exact target token counts, isolated caches at execution time, and zero generative LLM calls. The temporary per-run caches have since been removed; measurements remain.

A 1M-token attempt did not complete during NER and has no retained output or usable measurement. It does not establish indexing performance at that scale.

Detailed comparison and interpretation: `RESULTS_2026-09-21_SCALE_SERIES.md`.

## Design and interpretation

This experiment uses deterministic prefixes of local HotpotQA as a scale proxy, with the local `all-mpnet-base-v2` tokenizer. It measures the real `LinearRAG.index()` path, including embeddings, NER, and graph construction, while preventing generative LLM calls. It does not use ATLAS-Wiki or run RAPTOR/HippoRAG comparisons. The four completed one-shot runs are local scale observations; they do not establish stable scalability or reproduce the paper's 5M/10M settings.

Code entry point: `experiments/q7_large_scale_efficiency/large_scale_efficiency.py`.

Completed raw measurement directories:

- `data/output/experiment_results/q7_large_scale_efficiency/q7-design-check-25k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-design-check-50k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-formal-250k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-formal-500k-r2-20260921/`
