# Q7 Large-scale Efficiency Analysis

This experiment runs a bounded, LinearRAG-only cold-index scalability test.
It uses deterministic prefixes of the local HotpotQA corpus as scale proxies;
it does not claim to reproduce the paper's ATLAS-Wiki dataset or its RAPTOR
and HippoRAG baselines.

The formal bounded targets are 500,000 and 1,000,000 tokens, one tenth of the
paper's 5M and 10M settings. Token counts use the local
`all-mpnet-base-v2` tokenizer. Every invocation creates a new isolated cache,
calls the real `LinearRAG.index()` path, and fails if indexing tries to call a
generative LLM.

Run the first-stage design checks from the repository root:

```powershell
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 25000 --experiment-id q7-design-check-25k
.venv\Scripts\python.exe experiments\q7_large_scale_efficiency\large_scale_efficiency.py --target-tokens 50000 --experiment-id q7-design-check-50k
```

Raw evidence is written under
`data/output/experiment_results/q7_large_scale_efficiency/<experiment-id>/`.
