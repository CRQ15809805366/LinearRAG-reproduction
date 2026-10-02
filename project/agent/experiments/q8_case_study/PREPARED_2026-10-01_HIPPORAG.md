# Q8 HippoRAG supplement — prepared, not executed

The single Table 7 question `3a3c2efe0bdc11eba7f7acde48001122` is prepared for a matched comparison of default BFS LinearRAG and original HippoRAG 2024 v1.0.0. This is not the paper's HippoRAG2 baseline.

Both methods consume the same 658 ordered passages from `2wikimultihop/chunks.json`, formatted as in Q1. Preparation verified their text set against the existing LinearRAG passage cache. The marriage and Germany support passages are indices 28 and 27, located from fact text. Diagnostic entities and support indices do not participate in ranking or generation.

Code changes are confined to `experiments/q8_case_study/case_study.py`, `src/baselines/hipporag.py`, and `experiments/hipporag/official_runner.py`. No external official source or reproduced retrieval algorithm was changed. Existing LinearRAG BFS trace is retained. HippoRAG diagnostics reuse official query/link/PPR logs and add graph size, exact focus-node presence, one-hop edges around case/linked entities, support-passage extraction triples, and full-ranking positions/scores. Relation labels are pair metadata and can be overwritten by similarity edges; inspect extracted triples alongside them. No per-iteration PPR trace or full-graph export was added.

Configuration: `all-mpnet-base-v2`; generation and HippoRAG extraction `qwen3.8-flash`; Top-5; LinearRAG iterations 3, delta 0.4, lambda 0.05, one sentence; HippoRAG damping 0.5, similarity threshold 0.8, no document ensemble or DPR-only mode.

Executed preparation command:

```powershell
.venv\Scripts\python.exe -m experiments.q8_case_study.case_study --prepare-only
```

Evidence under `data/output/experiment_results/q8_case_study/q8-hipporag-case-20261001/`: `prepared.json` and `offline_validation.json`. Focused offline validation script: `data/output/verify_q8_preparation.py`. Passed syntax/function-docstring checks, missing-node and one-hop graph cases, support ranking outside Top-5, diagnostic query-cache separation without changing index identity, and identical Top-5 passages/scores with instrumentation enabled/disabled using simulated official ranking. No real model loading, official ranking, or paid calls were performed. This does not establish live end-to-end operation.

Formal command, NOT executed:

```powershell
.venv\Scripts\python.exe -m experiments.q8_case_study.case_study
```

A cold HippoRAG corpus requires 1,316 passage NER/OpenIE calls, one query-NER call, plus two answer-generation calls across both methods. Actual tokens/cost remain unknown until responses are recorded. Driver edits change the existing index fingerprint; do not assume older driver caches will be reused. Query observations have a separate query identity. The old `q8-paper-case-20260921` evidence remains untouched. LinearRAG saves `linearrag.json` before HippoRAG starts; completed comparison would save `result.json`. HippoRAG extraction/retrieval checkpoints support recovery; the Q8 entry reruns LinearRAG generation on another formal invocation.
