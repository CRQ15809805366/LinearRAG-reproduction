# Source layout migration — 2026-09-30

## Decision and scope

Group source files by their role without adding a framework or changing the reproduced algorithms:

- `src/linearrag/`: `LinearRAG.py`, `config.py`, `ner.py`.
- `src/baselines/`: `vanilla_rag.py` and an intentionally empty `hipporag.py`.
- `src/common/`: `embedding_store.py`, `evaluate.py`, `utils.py`.
- Keep `src/run.py`, `src/smoke_test.py`, `src/paths.py` and their module entry names unchanged.

Update all Python imports in `src/` and Q1–Q8 experiment adapters, plus current documentation. Historical experiment/source records retain their original paths. No legacy import forwarding modules are maintained.

HippoRAG is not integrated or installed by this migration. Its future adapter will invoke a pinned external official implementation in a separate environment and convert its retrieval results for the existing experiments. The placeholder contains zero bytes.

## Existing entry failures repaired

The pre-migration Q1 script contained `LinearRAGConfig(0`, which failed compilation. Remove the stray `0`.

Q6 imported `compare_retrievals` from the Q3 design-check script, which no longer exports that function. Import the existing comparison, evaluation and JSON helpers directly from `experiments.q3_ablation_study.formal_ablation`. No helper logic was rewritten.

## Verification

- Compile `src/` and `experiments/` successfully after migration.
- Import all nine experiment scripts covering Q1–Q8; parse `--help` for all eight scripts that expose an argument parser. Q8 has no argument parser and was import-checked without executing its case.
- Import and parse both `src.run` and `src.smoke_test` entry points.
- Confirm project root, dataset and cache paths remain identical after changing the current working directory outside the checkout.
- Compare ASTs of all seven relocated core modules against HEAD, after applying only the import-name mapping: identical.
- `python -m src.smoke_test` with the project `.venv` exited 0. `data/output/smoke/smoke_result.json` reports `status=passed`, default BFS, `bfs_replay_matches_official=true`, and `human_expectation.all_passed=true`. Existing `smoke_gpu` cache was reused; no LLM endpoint was called. This is not a full experiment or live API validation.

Runtime inputs, result directories and cache formats are unchanged. No pickle/joblib serialization dependency on the moved module names was found in source. Existing user changes under `project/human/` were left untouched. No commit or push was performed.
