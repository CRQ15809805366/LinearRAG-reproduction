# Repository instructions

- Keep the three top-level concerns separate: Python code in `src/`, runtime inputs and outputs in `data/`, and human-facing project material in `project/`.
- Treat the reproduced LinearRAG implementation as an experimental object. Do not refactor algorithms or change behavior unless the task explicitly requires it.
- Run the minimal smoke test with `python -m src.smoke_test`; do not present it as the full `src.run` experiment.
- Keep the default BFS retrieval path distinct from the optional `--use-vectorized-retrieval` path.
- Resolve project-owned runtime paths through `src.paths`, not through the caller's current working directory.
- Do not commit local models, datasets, caches, generated run results, secrets, or virtual environments.
- Treat `project/agent/` as agent-owned operational memory, separate from user-facing material in `project/human/`. Use it freely to record each major version's decisions, verified state, risky or consequential operations, recovery information, and other evidence worth preserving; add subdirectories when that improves retrieval. Do not create records for trivial work merely for completeness.
- Write all Agent-maintained files under `project/agent/` in English. Preserve commands, paths, identifiers, and raw evidence in their original form when translation would reduce fidelity.
- Before version work or debugging runtime, environment, cache, or reproduction issues, inspect the relevant records in `project/agent/` (starting with `BASELINE.md` when applicable). Update or add a record when the work changes a recorded fact or leaves important evidence for a later agent. Treat recorded environment and runtime facts as snapshots and re-verify anything that may have drifted.
- `origin` currently points to the upstream authors' repository (`DEEP-PolyU/LinearRAG`). Never push to it. Configure and verify a user-owned remote before any push.
