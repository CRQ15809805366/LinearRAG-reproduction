# Runtime data

This directory separates inputs supplied to the project from generated runtime artifacts.

## Inputs

- `input/datasets/`: local question and passage files for HotpotQA, 2WikiMultiHopQA, MuSiQue, and Medical.
- `input/models/`: local embedding model assets. These are runtime dependencies, not experiment results.
- `input/examples/smoke/`: small fixed inputs for the minimal indexing and retrieval smoke check.

## Outputs

- `output/cache/`: shared rebuildable caches used by ordinary runs. Keep a cache while it saves meaningful repeated indexing work; it can be rebuilt from inputs when its contents are no longer needed.
- `output/experiment_results/`: per-run raw results. Formal and bounded results include predictions or retrieval records, measurements, summaries, and metadata. Design-check runs are small pipeline or mechanism samples; they are not substitutes for the corresponding bounded results.
- `output/smoke/`: the smoke check's result files. `smoke_result.json` records the default BFS path; `smoke_result_vectorized.json` records the separate optional vectorized path.

Completed experiments may also create isolated caches inside their run folders to measure a genuinely cold index. Those caches are temporary runtime artifacts. Once the run's measurements and metadata have been saved, retain the evidence files and remove large isolated caches unless a later experiment explicitly reuses them. Do not retain empty error placeholders or resume checkpoints after final results are complete.

The current Q1–Q8 state, evidence paths, and interpretation limits are indexed in `project/agent/experiments/README.md`. The entire `data/output/` tree is generated and ignored by Git; preserve raw result files that support the experiment records or the user's learning and presentation goals.
