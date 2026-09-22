# LinearRAG reproduction project

The repository separates implementation, runtime evidence, internal research records, and externally readable research material:

- `src/`: Python implementation and executable entry points.
- `data/input/`: datasets, local models, and fixed example inputs.
- `data/output/`: rebuildable caches and generated experiment results.
- `project/human/`: private learning notes, plans, and personal interpretations.
- `project/agent/`: curated operational context, experiment records, decisions, and source material used by future Codex sessions.
- `research_report/`: externally readable reports, curated figures and tables, and presentation material derived from verified experiments.

Bulk logs, predictions, metrics, caches, and other machine-generated run artifacts remain under `data/output/`. Research claims in `research_report/` should be traceable through the experiment records in `project/agent/` to those raw artifacts.

Run the formal entry point from the repository root with:

```powershell
.venv\Scripts\python.exe -m src.run
```

Run the minimal smoke test with:

```powershell
.venv\Scripts\python.exe -m src.smoke_test
```

The smoke test exercises the real indexing and retrieval implementation, but it does not replace the full dataset, answer-generation, or evaluation experiment.

The answer-generation and evaluation path calls an OpenAI-compatible endpoint. Store its credentials in the untracked file `.env.local` at the repository root:

```text
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-endpoint/v1
```

`src.utils` reads these values directly and passes them to the OpenAI client. They are not injected into the process environment or set as Windows user-level variables. Do not commit `.env.local`.
