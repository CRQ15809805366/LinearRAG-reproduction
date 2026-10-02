# Q2 shared-corpus run — partial 2026-10-01

Run: `data/output/experiment_results/q2_efficiency_analysis/q2-hotpotqa-context20-noise50-run-20261001/`. User authorized live execution and requested minimal checks, fast completion and brief results. Prepared `-r2` manifest no longer matched current adapter/runner hashes; a new run captured current code without overwriting old preparation.

Input: exactly new Q1 HotpotQA 20 questions / 250 passages. Methods: default BFS LinearRAG and HippoRAG 2024; Top-5, local all-mpnet-base-v2. Five warm retrieval repetitions. No answer generation/evaluation.

LinearRAG completed: cold index 41.7331122 s; first query average 0.09921697 s; warm repeat-average median 0.09986215 s/question; zero LLM calls/tokens. All 20 results passed Top-5 and corpus validation. Raw: `linearrag/measurements.json`, `linearrag/retrieval_results.json`.

HippoRAG did not complete: 192/250 passages checkpointed, 58 remain; no completed graph, query rankings or efficiency comparison. Initial malformed OpenIE responses included a missing `triples` key and responses reaching the 4096-token ceiling. A scoped external compatibility change in `src/openie_with_retrieval_option_parallel.py` enables JSON mode for the bridge `qwen3.8-flash` client (its class differs from upstream's ChatOpenAI import) and raises OpenIE max_tokens to 16384; prompts and ranking were unchanged. Current external diff is retained in `experiments/hipporag/compatibility.patch`. Existing completed extractions were reused; any eventual recovered index duration cannot count as a complete cold-build observation. See run `recovery_adjustment.json`.

Later OpenIE API requests timed out. Verified the child explicitly inherits system HTTP/HTTPS proxy and the actual client uses trust_env=True with 180-second timeout. Retried missing extraction once after that diagnosis; same timeout recurred (passages 102/144). Official OpenIE returns an empty string after timeout, which surfaces as parser SyntaxError. Stopped after retry; no continuous retry or new framework. Previous commentary's 60-second timeout assumption was corrected to the live 180-second value.

Successful API response records across all attempts: 396 responses, 210088 prompt + 96140 completion = 306228 tokens. This includes paid responses later rejected by parsing; it excludes unrecorded failed requests and is not a provider billing total or completed-index cost. Bulk logs/checkpoints/usage remain in the run-owned HippoRAG cache. `run_status.json` marks partial status; no passing formal summary exists.

No dependency installs, commit or push. Resume preserves the completed LinearRAG results and the 192 passage extractions, but the repeated network failure remains unresolved. Do not report HippoRAG timing, accuracy or comparative superiority from this partial run.

User-requested resume: advanced from 192 to 249/250 saved extractions. Missing passage indices: [144]. Verified inherited HTTP/HTTPS proxy and retried missing work once after new timeout; last passage still timed out. No graph/retrieval measurement completed. LinearRAG result unchanged. Current successful-response totals across all attempts: 513 responses, {'prompt_tokens': 268753, 'completion_tokens': 118824, 'total_tokens': 387577}; same billing limitations apply. Updated raw run_status.json; no code changes or dependency checks during this resume.

Final state after last-passage diagnosis: completed with recovery. See `RESULTS_2026-10-01_SHARED_CORPUS.md`. Passage 144 produced a complete 12-triple response using targeted streaming/max_tokens2048 recovery; all 250 extraction checkpoints and both method retrieval measurements now exist. Historical timeout account above remains unchanged. HippoRAG complete cold-index time is still unavailable.
