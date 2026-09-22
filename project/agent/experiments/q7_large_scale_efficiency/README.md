# Q7 Large-scale Efficiency Analysis

## Status

The first-stage 25K/50K-token design check completed successfully on
2026-09-21. The one-shot bounded 250K and 500K runs also completed. The
one-shot 1M run failed during spaCy transformer NER with a host-memory
allocation error at 528/998 passages. This is a HotpotQA scale-proxy
experiment, not the paper's ATLAS-Wiki 5M/10M experiment.

Detailed results:
`RESULTS_2026-09-21_SCALE_SERIES.md`.

## Design

- Source corpus: local `hotpotqa/chunks.json`.
- Subset construction: deterministic nested prefixes; only the final passage
  may be truncated at a tokenizer boundary.
- Tokenizer: local `data/input/models/all-mpnet-base-v2` tokenizer.
- Measured operation: the real `LinearRAG.index(passages)` call, including
  embeddings, NER, graph construction, and cache writes.
- Each process used a new experiment ID and isolated cache.
- A fail-fast `NoLLM` object guarded against generative LLM calls.
- Retrieval, answer generation, and answer evaluation were intentionally out
  of scope.

## Commands

```powershell
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 25000 --experiment-id q7-design-check-25k-20260921
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 50000 --experiment-id q7-design-check-50k-20260921
```

## Raw evidence

- `data/output/experiment_results/q7_large_scale_efficiency/q7-design-check-25k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-design-check-50k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-formal-250k-20260921/`
- `data/output/experiment_results/q7_large_scale_efficiency/q7-formal-500k-r2-20260921/`

Both runs have `isolated_cache=true`, `resumed_cache=false`, exact target token
counts, and zero generative LLM calls.

| Target | Passages | Index seconds | Entities | Sentences | Graph vertices | Graph edges | Cache bytes | CUDA peak allocated bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25,000 | 25 | 23.5984473 | 1,666 | 749 | 1,691 | 2,174 | 9,822,566 | 1,186,101,760 |
| 50,000 | 50 | 46.5257928 | 3,271 | 1,499 | 3,321 | 4,401 | 18,208,830 | 1,925,146,624 |

The input scale doubled and observed index time increased by approximately
1.9716x. Seconds per million input tokens were 943.94 and 930.52. This is a
useful design signal but two small, single-run points do not establish stable
linear scalability.

## One-shot bounded execution

```powershell
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 500000 --experiment-id q7-formal-500k-r2-20260921
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 1000000 --experiment-id q7-formal-1m-20260921
```

| Target | Outcome | Passages | Index seconds | Entities | Sentences | Graph vertices | Graph edges | Cache bytes | CUDA peak allocated bytes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 500,000 | passed | 499 | 462.9087062 | 27,362 | 14,489 | 27,861 | 43,493 | 151,654,004 | 4,376,660,992 |
| 1,000,000 | failed | 998 planned | n/a | n/a | n/a | n/a | n/a | incomplete | n/a |

The successful 500K run wrote raw evidence to
`data/output/experiment_results/q7_large_scale_efficiency/q7-formal-500k-r2-20260921/`.
It used an isolated cold cache, made zero generative LLM calls, and consumed
zero prompt/completion tokens.

The 1M run failed at NER progress `528/998` with
`numpy.core._exceptions._ArrayMemoryError: Unable to allocate 32.5 MiB`.
`LinearRAG.index()` writes `ner_results.json` only after the full `batch_ner()`
return, so the failed directory contains no NER checkpoint and cannot resume
from passage 528. A PowerShell recursive removal request for the failed output
directory was rejected by the execution environment; the incomplete cache has
not been reused as evidence.

## Observed warning

The Hugging Face tokenizer printed a long-sequence warning while counting raw
passage tokens because some source chunks contain about 1,000 tokens whereas
the embedding model has a shorter inference limit. The indexing run did not
fail: SentenceTransformer applies its configured inference truncation in the
actual embedding path. The warning exposes an existing property of the source
chunks and does not indicate an LLM call.

## Evidence boundary

Executed evidence supports that the Q7 harness can construct exact nested
token scales, run real isolated cold indexes, and capture the expected metrics
without a generative model. It does not yet support a stable scaling claim,
the 1M result, ATLAS-Wiki equivalence, or comparisons with RAPTOR and HippoRAG.
