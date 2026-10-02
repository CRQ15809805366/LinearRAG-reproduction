# Controlled small-corpus Q1 supplement — prepared 2026-10-01

## Status and authorized boundary

Preparation only. User authorized shared corpus construction and reuse of Q1 through readiness for an experiment. No paid API, real baseline run, answer generation or evaluation was performed. Existing uncommitted source reorganization and HippoRAG work were preserved; no commit/push was made.

## Design and corrected data assumption

HotpotQA was selected because the historical Q1 sample showed a large LinearRAG-versus-Vanilla margin. This is a selected favorable setting, not a generalization claim or evidence of superiority over HippoRAG. Questions are random with seed `20261001`, without filtering on prior predictions.

The local source has 1,000 questions and 1,311 merged/normalized chunks. Its `evidence` field contains candidate contexts (approximately ten documents per question), including distractors; it has no gold `supporting_facts`. Therefore the implementation does **not** claim gold evidence extraction or original-chunk mapping. It preserves every sampled question's candidate document in full, using title + newline + original sentences, deduplicates exact text, adds random documents from the remaining candidate pool, and shuffles the shared corpus. Extra documents may incidentally be useful.

All three methods receive the identical document pool with identical numeric passage prefixes and the same question IDs/text. Runtime question payloads contain only ID, question and answer; candidate-context mappings are not exposed to retrievers. Source/candidate mappings remain in the input manifest for audit, not as gold evidence attribution.

## Prepared inputs and offline scale audit

Builder: `experiments/corpus_construction/build_corpus.py`.

Input package: `data/input/derived_corpora/hotpotqa-context20-noise50-s20261001/` (`questions.json`, `chunks.json`, `manifest.json`). This directory is ignored by Git. Source and generated file hashes, builder hash, document provenance and candidate mappings are retained.

| Questions | Candidate documents | Added random documents | Total documents | Characters | Whitespace words |
|---:|---:|---:|---:|---:|---:|
| 20 | 200 | 50 | 250 | 139509 | 22794 |
| 30 | 300 | 50 | 350 | 193912 | 31518 |
| 50 | 493 | 50 | 543 | 307077 | 49881 |

Only the 20-question package was materialized. This smaller default is a cost-driven design choice, not a power calculation. Different document boundaries/casing and a smaller search space prohibit combining historical full-corpus scores with the new scores. All selected methods must be rerun on this package. Q4 may reuse predictions and mappings, but gold Evidence Recall requires additional verified support labels.

## Q1 implementation

`experiments/q1_generation_accuracy/generation_accuracy.py` now accepts `--corpus-dir` and `--methods vanilla_rag linearrag hipporag`. Original dataset-mode sampling, default two-method selection and two-method aggregation format remain. In fixed-corpus mode, all package questions are used without resampling. Source dataset determines LinearRAG parameters; input hashes and embedding path determine a separate cache root under `data/output/cache/derived_corpora/`. Core algorithms and the default BFS path were unchanged.

Generation/evaluation: explicitly `qwen3.8-flash`, existing prompts and evaluator, Top-5; local embedding `all-mpnet-base-v2`. Original HippoRAG 2024 uses the existing independent-process adapter, extraction model `qwen3.8-flash`, 4 workers, damping=0.5, sim_threshold=0.8. External checkout HEAD was rechecked as `b144c46df14cabe5f5822d8caded4bec5f709461`; the previously documented compatibility patch remains part of the baseline.

Prepared run: `data/output/experiment_results/q1_generation_accuracy/q1-hotpotqa-context20-noise50-ready-20261001/manifest.json`. No predictions or summary exist in this run. Launch commands are in `experiments/q1_generation_accuracy/README.md`.

Resume requires exact manifest agreement and checks complete predictions for ID/text/gold-answer alignment and Top-5 corpus membership. Saved full predictions can be evaluated without regenerating. HippoRAG's internal per-passage extraction and per-question generation persistence remain available. Vanilla/LinearRAG generation is not per-question resumable; interrupted evaluation may repeat some paid judgments. Historical manifests predate the new schema and must not be resumed under the new script; preserve them and use new IDs. Code/dependency changes require new IDs since the complete environment is not automatically fingerprinted.

## Shared HippoRAG cache update

The same-day cache separation moves corpus extraction/index artifacts to `data/output/cache/hipporag/<index-id>/` and ranked results to its `queries/<query-id>/`. Q1's `hotpotqa/hipporag/` retains experiment requests, cache references, generated predictions, logs and per-invocation token deltas/history. There is no nested `official/` result directory in new runs. Shared ranked results exclude gold/generated answers; the adapter attaches the current gold answer on return. Different query batches share corpus extraction and overlapping question NER; identical batches/configurations bypass the child entirely. See `../../decisions/HIPPORAG_2024_INTEGRATION_2026-09-30.md` for current identities and validation limits. No formal run was started.

## Cost boundary

Fresh successful execution is expected to require 250 passage NER + 250 OpenIE + 20 query NER + 60 answer generation + 60 evaluation calls = 640 logical API calls, excluding failures/retries. Local embedding, graph construction and ranking add no online LLM calls. Character/word counts are not billing token estimates; repeated official few-shot prompts, extracted entity lists, triples and generated answers contribute cost. Provider prices and actual output lengths were not verified. No currency estimate or hard spend cap is claimed. Existing usage logging measures successful extraction/query-NER responses; generation/evaluation costs are not included in that HippoRAG token file.

## Verification evidence and limits

Offline evidence: `data/output/experiment_results/q1_generation_accuracy/preparation-20261001/offline_validation.json`; reproducible check script: `data/output/verify_small_corpus_preparation.py`.

Verified deterministic reconstruction, full candidate-context retention, unique passages, provenance/file hashes, exclusion of context metadata from runtime payloads, separate cache, actual three-method wrapper routing with mocked models, shared input/Top-5, default BFS config, complete resume without generation, re-evaluation using saved predictions, changed-configuration rejection, original two-method aggregation and Python docstrings. Local embedding file, external interpreter and credential-file existence were checked without reading or printing credentials.

These are preparation checks with mocked model/evaluator calls. They do not prove current live endpoint validity, real OpenIE quality, full indexing/retrieval execution or comparison accuracy. No Q1/Q4/Q2 formal HippoRAG result was added.
