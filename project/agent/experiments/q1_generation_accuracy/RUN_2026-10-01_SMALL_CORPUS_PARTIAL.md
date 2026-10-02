# Small-corpus Q1 execution — partial, 2026-10-01

Run: `data/output/experiment_results/q1_generation_accuracy/q1-hotpotqa-context20-noise50-ready-20261001/`. Fixed 20 HotpotQA questions / 250 documents, `qwen3.8-flash`, Top-5, default BFS.

| Method | Contain | LLM | Status |
|---|---:|---:|---|
| Vanilla RAG | 16/20 | 16/20 | Completed |
| LinearRAG | 15/20 | 16/20 | Completed |
| HippoRAG 2024 | — | — | Incomplete |

Evidence: per-method predictions/evaluation files and `partial_summary.json`. No HippoRAG accuracy or three-method summary exists.

CPU memory failures were resolved by running methods separately and loading the same spaCy model before embeddings. `data/output/run_q1_remaining.py` reuses the real preloaded NER; BFS/model behavior was unchanged.

The live driver was edited during execution and briefly had a duplicate `try` syntax error. The original failed request is `hotpotqa/hipporag/request.failed-start.json`. After an OpenIE API timeout with 37 saved extractions, child system proxy inheritance was checked and one retry used a frozen driver snapshot with HTTP proxy support. OpenIE timed out again; retries stopped per policy.

Retained cache: `data/output/cache/hipporag/3a0642083695000993bc/`, 186/250 passages complete. Successful indexing-response usage across attempts: 199700 prompt + 85547 completion = 285247 tokens, including incomplete passage work; excludes generation/evaluation. No currency estimate.

Recovery snapshot/provenance: `hotpotqa/hipporag/official_runner_recovery.py`, `recovery.json`, `process.log`; launcher `data/output/retry_q1_hipporag.py`. The snapshot differs from the original driver fingerprint. Future recovery must explicitly preserve this cache/protocol and audit source changes; normal adapter invocation may reject the run or select a new cache. Do not delete completed extraction. This is not a completed GraphRAG comparison.


## User-authorized resume

User confirmed API balance and authorized checkpoint resume. The frozen driver and the same corpus/models/prompts were retained; completed Vanilla/LinearRAG results were not rerun. Extraction advanced to 231/250. Passage 227 NER repeatedly received API 400 `InternalError.Algo.DataInspectionFailed: Output data may contain inappropriate content.`; an OpenIE timeout also occurred on the first resumed attempt. One further identical-input retry reproduced the 400. No passage was removed or replaced, and no HippoRAG accuracy was produced. Current count/usage are recorded in `partial_summary.json`; remaining extraction is blocked by provider output inspection, not a proven balance shortage.
