# Q4 Retrieval Quality Evaluation

This experiment compares Vanilla RAG and the original LinearRAG retrieval path
on the same balanced sample of Medical questions. It evaluates the saved Top-5
contexts with the two GraphRAG-Benchmark retrieval metrics: context relevance
and evidence recall.

Run the first-stage check from the repository root:

```powershell
.venv\Scripts\python.exe experiments\q4_retrieval_quality\retrieval_quality.py --questions-per-type 3 --experiment-id q4-design-check-20260920
```

Raw retrievals, judge responses, per-question metrics, grouped summaries, and
paired comparisons are written under
`data/output/experiment_results/q4_retrieval_quality/<experiment-id>/`.
