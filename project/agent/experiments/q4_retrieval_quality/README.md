# Q4 Retrieval Quality Evaluation

## Status

The balanced 200-question formal bounded run completed on 2026-09-20. See
`RUN_2026-09-20_200Q.md`.

The first-stage balanced 12-question design check also completed on
2026-09-20.

Primary raw evidence:
`data/output/experiment_results/q4_retrieval_quality/q4-design-check-20260920/`

## Question and bounded scope

Paper Q4 asks whether LinearRAG can retain high evidence recall while improving
the relevance of retrieved context across Fact Retrieval, Complex Reasoning,
Contextual Summarize, and Creative Generation tasks. This local design check
compares Vanilla RAG and the original BFS LinearRAG retrieval path on three
fixed-seed Medical questions from each task type.

This is a pipeline and metric check, not a reproduction of paper Table 4. The
local run uses `qwen3.8-flash` as the retrieval judge and does not execute the
paper's other GraphRAG baselines.

## Controls

- Dataset: Medical, complete 225-passage corpus.
- Sample: three questions from each of four task types, selected with seed
  `20260920`; 12 questions total.
- Methods: Vanilla RAG Top-5 and original LinearRAG BFS Top-5.
- Embedding model: `data/input/models/all-mpnet-base-v2` on CUDA.
- Medical NER model: `en_core_sci_scibert`.
- LinearRAG parameters: `max_iterations=3`, `passage_ratio=1.5`,
  `iteration_threshold=0.5`, and `top_k_sentence=1`.
- Evaluator: `qwen3.8-flash`, thinking disabled by the existing client.
- Metrics: context relevance and evidence recall, adapted from the official
  MIT-licensed GraphRAG-Benchmark retrieval evaluator.

## Command

```powershell
.venv\Scripts\python.exe experiments\q4_retrieval_quality\retrieval_quality.py --questions-per-type 3 --experiment-id q4-design-check-20260920
```

## Executed results

| Method | Macro context relevance | Macro evidence recall |
|---|---:|---:|
| Vanilla RAG | 0.625 | 0.644 |
| LinearRAG | 0.750 | 0.661 |

All 12 questions produced exactly five passages under both methods. All
per-question metric values were valid, both relevance ratings completed, and
the evaluator classified every reference evidence item.

The two methods changed the retrieval context materially: every question had a
different Top-5 set, and mean Top-5 overlap was `0.317`. At the individual
question level, both metrics improved for two questions, both declined for one,
and at least one metric tied for nine.

Per-task deltas (LinearRAG minus Vanilla RAG) were:

| Task type | Context relevance | Evidence recall |
|---|---:|---:|
| Fact Retrieval | 0.000 | +0.250 |
| Complex Reasoning | +0.333 | -0.089 |
| Contextual Summarize | +0.167 | -0.085 |
| Creative Generation | 0.000 | -0.008 |

## Interpretation and limits

Executed evidence establishes that the Q4 data path, both retrieval methods,
both LLM-judged metrics, task grouping, and paired comparison are runnable. On
this sample, LinearRAG improved macro context relevance and only slightly
improved macro evidence recall. The per-task directions were mixed, so the run
does not support a claim of uniform superiority.

The sample contains only three questions per task type. The judge is local
`qwen3.8-flash`, whereas the paper used GPT-4o-mini. LLM-judged metric values
are discrete and may vary across repeated calls. These values therefore do not
reproduce Table 4, establish statistical significance, or generalize to the
full Medical dataset.
