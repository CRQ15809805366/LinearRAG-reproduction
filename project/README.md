# LinearRAG reproduction project

The repository is organized into three areas:

- `src/`: Python implementation and executable entry points.
- `data/input/`: datasets, local models, and fixed example inputs.
- `data/output/`: rebuildable caches and generated experiment results.
- `project/`: papers, observations, execution records, plans, and project documentation.

Run the formal entry point from the repository root with:

```powershell
.venv\Scripts\python.exe -m src.run
```

Run the minimal smoke test with:

```powershell
.venv\Scripts\python.exe -m src.smoke_test
```

The smoke test exercises the real indexing and retrieval implementation, but it does not replace the full dataset, answer-generation, or evaluation experiment.
