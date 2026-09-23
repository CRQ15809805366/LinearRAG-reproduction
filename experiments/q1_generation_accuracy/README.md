# Q1: Generation Accuracy

This bounded reproduction compares a frozen Vanilla RAG baseline with the
original LinearRAG implementation on the same sampled questions. One script
performs sampling, both runs, evaluation, and result aggregation.

Prepare and inspect a deterministic sample without loading models:

```powershell
.venv\Scripts\python.exe experiments\q1_generation_accuracy\generation_accuracy.py --prepare-only
```

Run the bounded four-dataset experiment (100 questions per dataset by default):

```powershell
.venv\Scripts\python.exe experiments\q1_generation_accuracy\generation_accuracy.py
```

Generation and evaluation credentials are read directly from the untracked
project-root `.env.local` by `src.utils`. The values do not need to be set in
the process environment; do not commit or display the credential file.

For a small end-to-end trial, select one dataset and a smaller sample:

```powershell
.venv\Scripts\python.exe experiments\q1_generation_accuracy\generation_accuracy.py --datasets 2wikimultihop --max-questions 10
```

Raw predictions, evaluation files, the fixed sample manifest, and the final
comparison are written to
`data/output/experiment_results/q1_generation_accuracy/<experiment-id>/`.
Medical reports GPT accuracy only; the other datasets report both contain-match
and GPT-evaluated accuracy. The default LinearRAG path is the official BFS path.
