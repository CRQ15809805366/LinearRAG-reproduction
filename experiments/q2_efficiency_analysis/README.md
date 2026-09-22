# Q2 Efficiency Analysis

This experiment measures the original LinearRAG implementation on
2WikiMultiHopQA without answer generation or LLM evaluation.

The design-check profile uses the complete 658-passage corpus, ten fixed-seed
questions, one cold index build, one untimed retrieval warm-up, and two timed
retrieval repetitions. Each run gets a new isolated cache so that existing
project caches cannot turn the cold-index measurement into a cache-reuse run.

The experiment reports model/setup time separately from indexing time. The
primary measurements are `LinearRAG.index()` wall time and per-question
`LinearRAG.retrieve()` wall time. A fail-fast `NoLLM` object verifies that
neither measured stage calls a generative LLM.

Raw evidence is written to
`data/output/experiment_results/q2_efficiency_analysis/<experiment-id>/`.

Run the first-stage design check from the repository root:

```powershell
.venv\Scripts\python.exe experiments\q2_efficiency_analysis\efficiency_analysis.py --experiment-id q2-design-check
```
