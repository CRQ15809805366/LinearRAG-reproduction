# Codex debugging baseline

This file is operational context for future Codex sessions. It is not a tutorial or a claim of complete paper reproduction. Treat environment details and generated results as historical snapshots until re-verified live.

## Version boundaries

| Commit | Boundary | Debugging meaning |
|---|---|---|
| `75c6c4e` | Minimal source readability refactor | Primarily comments, module structure, and explicit configuration. If behavior appears to have changed, inspect the diff instead of assuming an algorithm rewrite. |
| `f3437f7` | Reproduction environment preparation closure | Established the current dependency/model direction and removed bulky setup evidence from the tracked tree. It does not prove the full paper experiment runs. |
| `4b32e0f` | Repository structure and path refactor | Moved code, runtime data, and project material into separate areas; made project-owned paths independent of the caller's working directory. |

## Repository invariants

- `src/` contains Python implementation and both entry points.
- `data/input/` contains local datasets, embedding models, and tracked smoke inputs.
- `data/output/` contains rebuildable caches and generated results; it is ignored by Git.
- `project/human/` contains personal observations and human-facing project documentation.
- `project/agent/` contains operational context and source material intended for future Codex sessions; the paper and recovered experiment parameters live under `project/agent/paper/`.
- Project-owned paths must be derived from `src.paths.PROJECT_ROOT` and passed into core code. Do not reintroduce paths whose meaning depends on the shell's current directory.
- Run entry points as modules from the repository root:

```powershell
.venv\Scripts\python.exe -m src.run
.venv\Scripts\python.exe -m src.smoke_test
```

## Last verified minimal runtime

Snapshot date: 2026-09-17.

- Platform: Windows.
- Python: `3.9.25`.
- GPU observed by the final smoke run: NVIDIA GeForce RTX 4050 Laptop GPU.
- Embedding model: `data/input/models/all-mpnet-base-v2` on `cuda:0`.
- spaCy model: `en_core_web_trf`.
- Retrieval path tested: default official BFS iteration, not the optional vectorized branch.
- Cache used by the final test: `data/output/cache/smoke_gpu/`.
- Report written to: `data/output/smoke/smoke_result.json`.
- Final result: process exit code `0`, `status=passed`, `bfs_replay_matches_official=true`, and `human_expectation.all_passed=true`.
- The final post-migration run reused an existing cache. When investigating cache construction or invalidation, explicitly perform a cold-cache run instead of treating this result as cold-start evidence.

## Scope that remains unverified

The smoke result does not establish any of the following:

- Full runs over the four paper datasets.
- OpenAI-compatible answer generation or evaluator behavior.
- Paper metrics, scalability, or reproduction of reported tables.
- Correctness or performance of `--use-vectorized-retrieval` after the final calibration.
- Graph restoration from GraphML at startup; the implementation writes GraphML during indexing but no startup restore path has been confirmed.

## First bounded full-entry runtime

Snapshot date: 2026-09-17.

- Entry point: `.venv\Scripts\python.exe -m src.run`.
- Dataset: `2wikimultihop`, limited to the first 10 questions with the existing `--max_questions 10` working-tree change.
- Retrieval: default BFS path with the paper-derived 2Wiki parameters (`max_iterations=3`, `passage_ratio=0.05`, `iteration_threshold=0.4`, `top_k_sentence=1`).
- Generation and evaluation model: the `src.run` default, `qwen3.8-flash`, through the existing OpenAI-compatible client.
- Index state: warm cache reused from `data/output/cache/2wikimultihop/` (658 passages, 41,243 entities, and 21,023 sentences). This was not a cold-index timing run.
- Output: `data/output/runs/2wikimultihop/2026-09-17_14-29-07/` with 10 predictions, five retrieved passages per prediction, and both output JSON files present.
- Final result: process exit code `0`; LLM accuracy `1.0` (10/10); contain accuracy `0.9` (9/10).
- The contain-only miss was orthographic: gold `Ailéan mac Ruaidhrí` versus prediction `Ailean mac Ruaidhrí`; the LLM evaluator marked it correct.
- Scope: this proves a bounded end-to-end `src.run` execution across retrieval, generation, persistence, and evaluation. It does not reproduce the paper's 1,000-question GPT-4o-mini result.

Keep observed execution, static code analysis, and unverified expectations separate in all reports.

## Known debugging traps

1. A passing smoke test only validates the small real indexing/retrieval path described above. Do not report it as a complete reproduction.
2. Embedding Parquet files and NER results are caches tied to the effective model, input text, and configuration. If any of those change, move or clear only the relevant dataset cache and rebuild it before diagnosing ranking behavior.
3. The local igraph build previously reported `GraphML support is disabled` when reading GraphML. That is a reader capability issue, not evidence that the generated XML is corrupt.
4. Sentence records support entity propagation but are not final igraph vertices. The final graph contains passage and entity vertices.
5. Default retrieval is BFS. The vectorized path is opt-in and must not be used as evidence for default behavior.
6. `src.run` additionally depends on datasets, CUDA, an OpenAI-compatible endpoint, generation, and evaluation. Diagnose it separately from `src.smoke_test`.

## Debugging order

Use the smallest evidence-producing sequence:

1. Confirm `git status`, the current commit, Python version, model paths, and effective CLI arguments.
2. Confirm whether the failure is input loading, model loading, cache reuse/build, NER, graph construction, retrieval, generation, or evaluation.
3. Reproduce with `python -m src.smoke_test` when the suspected fault is within indexing or retrieval.
4. Use a cold-cache smoke run only when cache construction or stale artifacts are plausible causes.
5. Run `python -m src.run` only when the failure requires the full dataset/generation/evaluation chain.
6. Record whether each conclusion came from an executed run, static inspection, or an unverified inference.

## Remote safety

`origin` currently names the authors' upstream repository:

```text
https://github.com/DEEP-PolyU/LinearRAG.git
```

Never push to this remote. Before any future upload, create or select a user-owned repository, change or add the remote, display `git remote -v`, and verify the exact destination with the user. A local commit is not authorization to push.
