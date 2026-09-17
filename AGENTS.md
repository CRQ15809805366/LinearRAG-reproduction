# Repository instructions

- Keep the three top-level concerns separate: Python code in `src/`, runtime inputs and outputs in `data/`, and human-facing project material in `project/`.
- Treat the reproduced LinearRAG implementation as an experimental object. Do not refactor algorithms or change behavior unless the task explicitly requires it.
- Run the minimal smoke test with `python -m src.smoke_test`; do not present it as the full `src.run` experiment.
- Keep the default BFS retrieval path distinct from the optional `--use-vectorized-retrieval` path.
- Resolve project-owned runtime paths through `src.paths`, not through the caller's current working directory.
- Do not commit local models, datasets, caches, generated run results, secrets, or virtual environments.
