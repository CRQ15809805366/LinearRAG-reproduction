# Q2 shared-corpus run — completed with recovery 2026-10-01

Run: `data/output/experiment_results/q2_efficiency_analysis/q2-hotpotqa-context20-noise50-run-20261001/`. Historical failure details remain in `RUN_2026-10-01_SHARED_CORPUS_PARTIAL.md`; this record describes final state. Both methods completed on the identical 20-question / 250-passage HotpotQA package, Top-5. No answer generation/evaluation.

| Metric | LinearRAG | HippoRAG |
|---|---:|---:|
| Complete cold indexing | 41.7331122 s | Unavailable after extraction recovery |
| First query average | 0.09921697 s/question | 1.40230738 s/question |
| Median of five warm batch averages | 0.09986215 s/question | 0.017871575 s/question |

HippoRAG warm repeats reuse query NER and execute real rank_docs; first queries include uncached NER. Its final invocation index duration 7.7465308 s restores 250 extraction checkpoints and completes graph/runtime preparation, so it must not be compared with LinearRAG cold indexing. Raw measurements, summary, stage timing, usage and rankings remain under the run directory. Default LinearRAG BFS remained selected. Claims apply only to these local wrapper boundaries.

## Last-passage diagnosis and recovery

Passage 144 has 523 characters / 84 whitespace words, so input length did not explain the persistent timeout. The official OpenIE helper catches API errors and returns an empty string; the later SyntaxError was a secondary parsing symptom. Proxy inheritance and trust_env=True were already verified.

A targeted streaming diagnostic using original official prompts/model/temperature/JSON configuration returned initial content in about 2.4 s and continued generating over 13000 characters for this short passage without finishing. That diagnostic was explicitly stopped to limit cost. This supports abnormal prolonged generation as a cause candidate; it does not prove the cause of every earlier timeout. Its complete output/usage was unavailable.

A second diagnostic used max_tokens=2048 with streaming transport and the same official prompts/model/temp/JSON mode. It completed in 6.4301829 s, finish_reason=stop, 12 entities and 12 well-formed triples, with 319 completion / 685 prompt tokens. Complete validated output was saved into only checkpoint 144. No truncated output or fabricated triples were accepted. The other 249 extractions were retained. This is a documented per-passage output-budget/transport recovery exception, not a uniform extraction-config run.

Evidence: `hipporag/openie_144_diagnostic.json`, `hipporag/openie_144_stream.txt`; script `data/output/diagnose_q2_last_passage.py`. Source repository code was not changed during this final recovery. Original external prompts/ranking remain; previously documented JSON/max-token compatibility changes still apply.

## Logged token scope

Successful response usage across recorded attempts: indexing 270064 prompt + 119283 completion = 389347 tokens; query NER 2354 + 286 = 2640 tokens; combined 391987. Includes parser-rejected paid responses where usage was recorded, but excludes the explicitly stopped diagnostic stream and any unrecorded failed requests. Not a complete provider invoice or fresh successful-build cost. Final-invocation deltas must not substitute for these recorded cumulative values.

`summary.json` is present and `run_status.json` now marks `passed_with_recovery`. Input/runtime comparison complete; HippoRAG full cold-index measurement remains unavailable. No dependency installs, commit, push or additional Q1/Q4 runs.
