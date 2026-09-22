# Q7 Bounded Scale-proxy Results - 25K to 500K Tokens

## Execution status

Four isolated cold-index runs completed on 2026-09-21:

- `q7-design-check-25k-20260921`;
- `q7-design-check-50k-20260921`;
- `q7-formal-250k-20260921`;
- `q7-formal-500k-r2-20260921`.

All four runs have `status=passed`, exact target token counts, isolated cold
caches, and zero generative LLM calls. A separate 1M-token attempt failed
during NER and produced no completed measurement.

## Raw evidence

- `data/output/experiment_results/q7_large_scale_efficiency/q7-design-check-25k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-design-check-50k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-formal-250k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-formal-500k-r2-20260921/`

## Fixed conditions

- Source corpus: local `hotpotqa/chunks.json`.
- Paper corpus: ATLAS-Wiki, not used in this local scale proxy.
- Subset rule: deterministic nested corpus prefixes; only the final passage may
  be truncated at a tokenizer boundary.
- Tokenizer and embedding model:
  `data/input/models/all-mpnet-base-v2`.
- Measured operation: `LinearRAG.index(passages)`, including embedding, NER,
  graph construction, and cache writes.
- Retrieval parameters: `max_iterations=3`, `passage_ratio=0.05`,
  `iteration_threshold=0.4`, and `top_k_sentence=1`.
- NER model: `en_core_web_trf`.
- Environment: Python `3.9.25`, Torch `2.8.0+cu128`, CUDA `12.8`, and NVIDIA
  GeForce RTX 4050 Laptop GPU.
- A fail-fast `NoLLM` object rejected generative-model calls.
- Retrieval, answer generation, and answer evaluation were out of scope.

## Commands

```powershell
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 25000 --experiment-id q7-design-check-25k-20260921
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 50000 --experiment-id q7-design-check-50k-20260921
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 250000 --experiment-id q7-formal-250k-20260921
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 500000 --experiment-id q7-formal-500k-r2-20260921
```

## Results

| Target tokens | Passages | Setup (s) | Index (s) | Seconds per million tokens | Cache bytes | CUDA peak bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 25,000 | 25 | 2.7670675 | 23.5984473 | 943.9378920 | 9,822,566 | 1,186,101,760 |
| 50,000 | 50 | 2.4411446 | 46.5257928 | 930.5158560 | 18,208,830 | 1,925,146,624 |
| 250,000 | 250 | 2.6718896 | 243.5780218 | 974.3120872 | 80,757,157 | 4,224,486,400 |
| 500,000 | 499 | 3.0402969 | 462.9087062 | 925.8174124 | 151,654,004 | 4,376,660,992 |

The 500K target used 499 passages because the deterministic prefix ended at
499 complete passages. The measured token count remained exactly `500000`.

| Target tokens | Entities | Sentences | Graph vertices | Graph edges |
|---:|---:|---:|---:|---:|
| 25,000 | 1,666 | 749 | 1,691 | 2,174 |
| 50,000 | 3,271 | 1,499 | 3,321 | 4,401 |
| 250,000 | 14,835 | 7,349 | 15,085 | 22,239 |
| 500,000 | 27,362 | 14,489 | 27,861 | 43,493 |

## Scaling observations

Observed indexing throughput remained in a narrow band from `925.8174124` to
`974.3120872` seconds per million input tokens, a relative spread of `5.238%`.

| Segment | Token factor | Index-time factor |
|---|---:|---:|
| 25K to 50K | 2.0000 | 1.9716 |
| 50K to 250K | 5.0000 | 5.2353 |
| 250K to 500K | 2.0000 | 1.9005 |
| 25K to 500K | 20.0000 | 19.6161 |

The 25K-to-500K comparison produced a `20x` token increase and a `19.6161x`
increase in observed index time. Entity, sentence, graph-vertex, graph-edge,
cache-size, and peak-CUDA-memory values also increased with corpus size.

These are single observations at each scale, not repeated timing measurements.
They support approximate linear scaling on this proxy series but do not
establish a stable scalability distribution.

## Failure boundary

The separate 1M-token attempt failed at NER progress `528/998` with
`numpy.core._exceptions._ArrayMemoryError: Unable to allocate 32.5 MiB`.
`LinearRAG.index()` writes `ner_results.json` only after the complete
`batch_ner()` call returns, so the failed attempt had no usable NER checkpoint.
It is excluded from the completed-results table and does not establish a 1M
result.

## Interpretation and limits

Executed evidence establishes that the Q7 harness can construct exact nested
token scales, run real isolated cold indexes, and capture indexing and graph
metrics without calling a generative LLM. The observed indexing time and stored
graph size roughly track input-token count over this bounded range.

This is a HotpotQA scale proxy, not the paper's ATLAS-Wiki 5M/10M experiment.
It does not reproduce the paper's indexing times, establish a statistically
stable linear-scaling claim, execute retrieval or generation, or compare
LinearRAG with RAPTOR and HippoRAG.
