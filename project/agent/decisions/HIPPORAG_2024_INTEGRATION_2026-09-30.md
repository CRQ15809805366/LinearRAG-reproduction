# HippoRAG 2024 external integration — 2026-09-30

## Scope and ownership

Current cache layout (2026-10-01): see the final section, "Shared cache separation". Earlier paths below describe the retained synthetic integration evidence.

User authorized one external official HippoRAG 2024 checkout, an independent environment, file exchange, and a minimal real invocation. Keep engineering bounded and preserve the reproduced LinearRAG algorithms. This is an integration check, not a Q1/Q2/Q4/Q7 comparison or a HippoRAG2 reproduction.

- Official repository: `https://github.com/OSU-NLP-Group/HippoRAG.git`.
- External checkout: `D:\code\HippoRAG-2024`.
- Tag: `v1.0.0` (also advertised as the `legacy` branch at acquisition).
- Pinned HEAD: `b144c46df14cabe5f5822d8caded4bec5f709461`.
- Local tracked compatibility recipe: `experiments/hipporag/compatibility.patch`. The external tree has these documented local modifications; do not describe it as an entirely unmodified checkout.
- Independent interpreter: `D:\code\HippoRAG-2024\.venv\Scripts\python.exe`, Python 3.9.25.
- Installed packages: `experiments/hipporag/requirements-lock.txt`; `uv pip check` passed.

The adapter is `src/baselines/hipporag.py`. It exports original passages, fixed IDs/questions and configuration, invokes `experiments/hipporag/official_runner.py` once per batch, and returns the project's existing passage schema. Only the external process imports official `src`; no official source directory is added to the main process's imports. Runtime files are under `data/output/hipporag/<experiment-id>/`, resolved through `src.paths`. No service, database, submodule, commit or push was introduced.

## Method and compatibility decisions

Use the official HuggingFace encoder route with local `all-mpnet-base-v2`, official mean-pooling functions, explicit OpenIE relations, synonymy candidate search, and PPR (`damping=0.5`, similarity threshold `0.8`, `doc_ensemble=False`, `dpr_only=False`). This validates original HippoRAG mechanics with a local encoder/model configuration, not published GPT/Contriever/ColBERT numerical results. The synthetic fixture produced no synonymy edges, so it does not demonstrate propagation across a nonzero synonymy edge.

Minimal external source changes:

1. Import ColBERT only when that route is selected.
2. Retain exact FAISS inner-product search while enabling the CPU backend. Limit index partitions and per-part neighbor counts to available entities; otherwise the original default `k=2047` creates invalid/padded neighbor indices on tiny corpora.
3. Raise passage-NER failures instead of the original unbounded exception/retry loop.

The external driver uses a bounded thread pool to call the official passage NER/OpenIE functions, saving each completed passage. Graph construction calls official `create_graph`; KNN calls official `RetrievalModule`. Only `kb_to_kb` KNN is required by this route; unused query/relation KNN files are not computed. The official client's factory is replaced inside the external process to use the project's OpenAI-compatible endpoint and `qwen3.8-flash`; prompts remain official. Credentials are read directly from the untracked `.env.local`, passed to clients, and are not written into exchange files or environment variables. API retries are disabled in the bridge client; timeout is 60 seconds.

PyTorch 2.8.0+cu128 files were hardlinked from the already installed project environment, avoiding a redundant 3.2 GiB wheel download. Package resolution remains independent. Do not mutate those shared package files in place. Other dependencies were downloaded with `uv` using the Tsinghua PyPI mirror. A newer greenlet selected by dependency resolution lacked the relevant Windows wheel and attempted a C++ build; pinning greenlet 3.1.1 avoided installing a compiler. LangChain 0.2.17/community 0.2.19 provide the optional `ChatLlamaCpp` import required by upstream; no local llama.cpp model is installed.

Windows PyTorch/FAISS initially aborted with OpenMP Error #15. The bridge child alone now receives `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`. This is a local runtime workaround; formal efficiency conclusions require explicitly reconsidering the DLL/runtime condition. Encoding uses the RTX 4050 Laptop GPU and nearest-neighbor search uses CPU FAISS. Actual candidate scores were checked against NumPy exact normalized inner products.

## Executed evidence

Raw directory: `data/output/hipporag/minimal-20260930/`.

- `request.json`: 12 explicitly synthetic passages, 3 fixed multi-hop questions, Top-5, 4 extraction workers, original input strings and effective configuration.
- `extraction_checkpoint.json`: all 12 real LLM extractions completed. Graph output records 15 triples and 19 entity nodes.
- `retrieval.json`: 3 complete Top-5 rankings, finite scores, original passage strings, official graph logs, per-question query-NER/ranking times. The runner checks official passage order/text against the exported input.
- `predictions.json`, `evaluation_results.json`, `validation.json`: original project generation prompt and `Evaluator`, using `qwen3.8-flash`; all 3 synthetic answers correct under both evaluators. This is only wiring evidence, not dataset accuracy or baseline superiority.
- `llm_usage.jsonl`, `token_usage.json`: actual successful API-response usage, 9,063 prompt + 563 completion = 9,626 extraction/query-NER tokens across invocations. Excludes answer generation/evaluation. Warm/resumed invocations added zero extraction/query-NER tokens.
- `resume_validation.json`: after retaining only two retrieval records, rerunning reused all 12 extractions and the completed graph, preserved the first two records, and regenerated the third using cached question NER. All three passage rankings and scores matched the saved originals exactly. No repeated extraction/NER API calls were added.
- `knn_verification.json`: all 19 entity queries returned 19 distinct valid candidates with descending scores; maximum score difference from NumPy exact normalized inner product was `2.980232238769531e-07`.
- `source_verification.json`: ASTs of official `rank_docs`, `build_graph`, `run_pagerank_igraph_chunk`, and `link_node_by_dpr` match the pinned originals; `src/create_graph.py` has no diff.
- `timing.json`, `timing_history.jsonl`, `adapter_timing.json`: distinguish child startup/imports, model/tokenizer loading, conversion, extraction, graph work, retrieval, and parent file/subprocess overhead. Original startup measurements incorrectly compared Python 3.9 per-process performance-counter origins and returned negative values; those entries in historical/first-completed timing files are invalid. Current startup measurement uses shared wall-clock nanoseconds; stage durations still use local monotonic counters. No cold-index or comparative latency claim is made.
- `process.log`, `recovered_failure.json`: retain installation/runtime recovery evidence. The native OpenMP abort occurred after extraction/vector persistence, and recovery reused those files. No network timeout occurred during the successful installation/run sequence.

Commands executed from the project root:

```powershell
.venv\Scripts\python.exe -m experiments.hipporag.minimal
.venv\Scripts\python.exe -m experiments.hipporag.minimal --retrieval-only
.venv\Scripts\python.exe -m src.smoke_test
uv pip check --python D:\code\HippoRAG-2024\.venv\Scripts\python.exe
git -C D:\code\HippoRAG-2024 apply --reverse --check D:\code\LinearRAG-reproduction\experiments\hipporag\compatibility.patch
```

The LinearRAG smoke exited 0 with `status=passed`, default BFS, `bfs_replay_matches_official=true`, and `human_expectation.all_passed=true`; it reused the existing smoke cache and did not call an LLM. Python compilation of the added adapter/driver/check passed. Q1–Q8 formal entry points, sampling and historical results were not changed by this integration.

## Boundaries for future work

Use a new experiment ID when corpus, questions, encoder, extraction model, generation model or code version changes. Existing cache identity guards the request inputs/configuration; it does not automatically fingerprint all source/package changes. Per-passage and per-question persistence supports practical resume, but this is not transactional infrastructure or an automatic retry service. Existing answer evaluation is rerun when the full minimal command is rerun.

Q1/Q4 can reuse the adapter schema after adding a local method entry. Q2/Q7 require a separate controlled cold-index protocol, matching corpus, real token accounting and declared CPU/GPU/cache/runtime conditions; the integration recovery timings do not establish these comparisons. Q3/Q5/Q6 stay unchanged. Q8 original HippoRAG may provide a supplemental case, never a strict replacement for the paper's HippoRAG2 case.

## Driver readability refactor — 2026-10-01

User requested a behavior-preserving refactor and light verification. The driver now uses top-level functions for passage preparation/extraction, graph construction, question retrieval, credential loading, and summary persistence. `TimedLoader`, `UsageRecorder`, and `ClientFactory` expose their state explicitly; there are no nested function/class definitions. Official algorithms, client options, checkpoint names, request/result schemas, and graph/retrieval settings are retained. Formatting and stage docstrings were clarified.

Light offline verification passed: syntax, absence of nested definitions, extraction output and checkpoint reuse, question-NER cache reuse, passage/score/index/log fields, actual-usage callback persistence, and loader accounting with simulated dependencies. No API calls were made. An attempted full cached comparison stopped before model/client initialization because the temporary directory was on C: while the model is on D:; no end-to-end equivalence claim is made from that attempt. Final syntax was checked again after extracting credential/summary helpers.

## Adapter readability refactor — 2026-10-01

User requested the same behavior-preserving cleanup for `src/baselines/hipporag.py`. Public `index`, `retrieve`, `qa`, constructor defaults, and mutable result semantics are retained. Request preparation/writing, child environment/proxy inheritance, subprocess execution, response/timing persistence, generation, prediction restore, and atomic prediction saving now have separate methods. The nested generation closure was replaced by a bound method; formatting was expanded for readability.

A light offline old/new comparison passed using a mocked subprocess and fake LLM: identical request identity/payload, returned results, generation messages, prediction resume without regeneration, and rejection of changed inputs under the same run ID. Syntax and absence of nested definitions passed. No real subprocess, model run, or API call was performed.

## Shared cache separation — 2026-10-01

User requested cross-Q1/Q4 reuse with minimal added structure. Existing public `index/retrieve/qa`, official algorithms and prompts remain. No new class or framework was added. The adapter now separates shared cache from `run_dir`; Q1 passes its method output directory directly rather than creating an `official/` subdirectory. The minimal example uses new default run ID `minimal-cache-split-20261001`; old integration evidence was not moved or migrated.

- `data/output/cache/hipporag/<index-id>/`: corpus extraction checkpoints, official graph/vector files, corpus-wide question-NER checkpoint and cumulative indexing usage. Identity covers exact ordered passages, extraction model, embedding path, synonymy threshold, external checkout path and driver-file SHA-256. IDs use 20 hex characters to keep Windows paths short. Threads, question batches, Top-k, damping and generator do not affect index identity.
- `queries/<query-id>/` inside the index: question ID/text + index identity + Top-k/damping determine ranked-result identity. Gold/generated answers are excluded. Query-NER usage is attributed to the batch that actually incurred it; overlapping question text reuses the corpus-wide NER checkpoint.
- Experiment `run_dir`: request, cache reference, generated predictions, logs, timing, last-invocation token deltas and append-only token delta history. Successful extraction/query-NER accounting excludes answer generation and evaluation. Exact complete retrieval hits avoid the subprocess and record zero new extraction/query-NER usage.

The driver runs official relative-file code with cwd set to the shared index directory. It writes query results to the explicit query path and experiment summaries to the explicit run path. Incomplete extraction/query results resume. Cache usage totals retain acquisition cost; run deltas record only newly incurred tokens. Failed invocations may have successful response usage in cache logs before experiment-summary persistence; no transactional cost ledger is claimed.

Offline validation: `data/output/hipporag/cache-split-validation-20261001/validation.json`, script `data/output/verify_hipporag_cache_split.py`. Real cache/checkpoint/usage helpers ran with simulated OpenIE, NER, graph, ranking and generation. Verified cross-experiment index/retrieval reuse, new query batches without repeated passage extraction, overlapping query-NER reuse, Top-k changes without index rebuild, exclusion of answers from shared files, experiment-only generation, partial retrieval recovery, usage deltas/totals, corpus/extractor identity changes and rejection of generator changes under one run ID. All changed Python files compile and have function docstrings. No paid API or real model/official ranking run occurred.

Use shared caches sequentially; concurrent writers are not supported. Driver edits automatically create new index identities. Changes to external source/dependencies or model contents at the same path are not fully fingerprinted: explicitly select a fresh `cache_dir` when those change. Old pre-separation request files must not be resumed with this protocol; keep them as historical evidence and use new run IDs.

## Q2 measurement entry — 2026-10-01

`HippoRAG.measure_efficiency()` explicitly invokes the child even when rankings exist. Measurement settings enter request/query identity; normal `index/retrieve/qa` behavior remains. The child measures first query NER+ranking, then real cached-NER rank repetitions, and labels pre-existing index artifacts as reuse/recovery. GPU stage boundaries synchronize CUDA. Q2 uses isolated run-owned caches; no formal measurement or paid API call ran during preparation. See `../experiments/q2_efficiency_analysis/PREPARED_2026-10-01_SHARED_CORPUS.md`.

Because index identity includes the runner SHA-256, this instrumentation also changes subsequent normal-call index identities. Existing pre-change caches/evidence remain intact, but new Q1/Q4/Q8 launches cannot assume those caches will match. Local timing boundaries and model/tokenizer-loading exclusions are documented in the Q2 record; they are not paper-matched timing evidence.

Q2 live recovery on 2026-10-01 added an external OpenIE compatibility adjustment: JSON mode for the bridge `qwen3.8-flash` client and max_tokens 16384 (previously 4096). Original prompts/ranking unchanged; completed lower-ceiling extractions retained. The patch recipe reflects the current external src diff. Q2 remains incomplete after repeated API timeout; see `../experiments/q2_efficiency_analysis/RUN_2026-10-01_SHARED_CORPUS_PARTIAL.md` for raw evidence and billing/timing limits. The actual current bridge client timeout is 180 seconds with trust_env=True; older 60-second/direct-client descriptions above are historical.
