# Q2 shared-corpus efficiency supplement — prepared 2026-10-01

## Status

Prepared only. No paid API calls, live model runs, indexing or formal retrieval measurement were performed. Existing worktree changes were preserved; no commit or push was made. Historical Q2 results remain unchanged.

## Fixed protocol

- Input: `data/input/derived_corpora/hotpotqa-context20-noise50-s20261001/`, exactly the same 20 questions / 250 passages as the new Q1 supplement; no resampling. Q1/Q2 runtime questions, passage prefixes, ordered inputs and file hashes matched offline.
- Methods: LinearRAG default BFS and original HippoRAG 2024. Same local `all-mpnet-base-v2`, Top-5. LinearRAG uses HotpotQA parameters; HippoRAG uses `qwen3.8-flash`, 4 extraction workers, damping=0.5, sim_threshold=0.8.
- This is a bounded HotpotQA efficiency supplement, not a reproduction of the paper's 2Wiki Table 2.
- Each method gets a run-owned isolated cache. No Q1 index is reused for cold timing.
- Measure index once, first query batch once, one additional untimed warm-up batch, then five real retrieval repetitions. Repetitions share query-NER state for HippoRAG and never substitute saved rankings for execution.
- HippoRAG first-query time includes query NER plus ranking; it is separate from the cached-NER repeat metric. LinearRAG first-query time is also recorded.
- LinearRAG index excludes model setup. HippoRAG index sums conversion, passage extraction, graph/KNN and runtime graph/node-vector initialization, excluding recorded encoder/tokenizer loading. Startup, model loading, subprocess/file overhead and stage timings are retained separately. These are local wrapper boundaries, not proof of equivalence to the paper's timing implementation.
- GPU timing boundaries synchronize CUDA. Official algorithms and prompts are unchanged. CPU FAISS, GPU encoding and Windows OpenMP workaround remain local environment limitations.
- Recovered index artifacts disqualify the resumed duration as a complete cold-index observation. A new full cold observation requires a new run ID and an empty isolated cache. Resume retains extraction/query-NER checkpoints and records reuse.
- No answer generation/evaluation. Future accuracy linkage requires matching Q1 input hashes and method configurations.

## Code changes

- `experiments/q2_efficiency_analysis/efficiency_analysis.py`: fixed corpus input, explicit methods, lazy imports for prepare-only, per-method measurements, run manifest/recovery, and preparation call counts. Legacy 2Wiki sampling and external LinearRAG cache reuse remain available; new output schema is method-separated.
- `src/baselines/hipporag.py`: explicit `measure_efficiency()` always invokes the child; measurement settings enter request/query identity. Normal Q1/Q4/Q8 retrieval caching remains enabled.
- `experiments/hipporag/official_runner.py`: measurement-specific first-query and real rank-repeat path, synchronized stages, complete index boundary and initial-artifact/reuse flags.
- `experiments/corpus_construction/` and reproduced core algorithms were not edited.

The runner hash participates in HippoRAG index identity. This change gives subsequent normal calls a new index identity too; do not assume a pre-change shared extraction cache will be reused. No older cache/evidence was deleted. Prepared Q1/Q4/Q8 IDs should not be treated as executed results or zero-cost future launches.

## Prepared run and cost

Run: `data/output/experiment_results/q2_efficiency_analysis/q2-hotpotqa-context20-noise50-ready-20261001-r2/`. Contains manifest, exact inputs and `preparation.json`; no method outputs or formal summary. Launch commands: `experiments/q2_efficiency_analysis/README.md`.

Fresh successful calls: 500 passage extraction (250 NER + 250 OpenIE), 20 query NER, zero generation/evaluation = 520 logical API calls. Ranking repeats reuse NER without additional LLM calls. Counts exclude failures/retries; tokens/currency cannot be inferred from them. Actual usage logging covers successful responses only. Recovery stores per-invocation deltas and shared cumulative usage; recovery deltas are not fresh-build cost.

## Verification

Evidence: `data/output/experiment_results/q2_efficiency_analysis/preparation-20261001/offline_validation.json`; repeatable script: `data/output/verify_q2_preparation.py`.

Ten offline checks passed: exact Q1/Q2 input equivalence; legacy 2Wiki input; syntax/docstrings; real rank repeats despite saved rankings; first NER and recovery state; complete index-stage sum; separate measurement-query identity; adapter process routing/configuration guard; prepared two-method orchestration/resume; actual LinearRAG measurement orchestration with mocked core/model and BFS/first-query/warm-up/repeat/cache-state checks.

Preparation succeeded when launched from `D:/code`, confirming project-owned relative paths resolve through `src.paths`. External checkout HEAD matched `b144c46df14cabe5f5822d8caded4bec5f709461`; local encoder config, external interpreter and credential file exist. Credentials were not displayed. Source compilation passed. These checks do not establish live endpoint validity, real graph behavior, real timing or benchmark accuracy.

Source walkthrough recovery: initial-index-state detection was incorrectly scoped to save_run_summary; the child main referenced an undefined variable. Moved detection into main before directory creation. Added `data/output/verify_q2_entry.py` with mocked full child entry; it observed fresh/reused states `[false, true]` and made no API/model calls. Evidence: `preparation-20261001/entry_validation.json`. Earlier ten checks were rerun successfully. The original prepared run remains unexecuted and is superseded by the `-r2` run above because the runner hash changed. This is not live-runtime validation.
