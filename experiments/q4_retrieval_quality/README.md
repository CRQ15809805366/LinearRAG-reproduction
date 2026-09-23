# Q4 Retrieval Quality Evaluation

This experiment compares Vanilla RAG and the original LinearRAG retrieval path
on the same balanced sample of Medical questions. It evaluates the saved Top-5
contexts with the two GraphRAG-Benchmark retrieval metrics: context relevance
and evidence recall.

The balanced 200-question bounded run is complete. Its primary results are
recorded in `project/agent/experiments/q4_retrieval_quality/RUN_2026-09-20_200Q.md`.
The smaller design check validates the retrieval and judging path only.

To run a new small design check from the repository root:

```powershell
.venv\Scripts\python.exe experiments\q4_retrieval_quality\retrieval_quality.py --questions-per-type 3 --experiment-id q4-design-check-new
```

Raw retrievals, judge responses, per-question metrics, grouped summaries, and
paired comparisons are written under
`data/output/experiment_results/q4_retrieval_quality/<experiment-id>/`.
