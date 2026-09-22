# Q6 Embedding Model Robustness

## Status

The four-model, 100-question evaluation completed on 2026-09-21.
See `RESULTS_2026-09-21_100Q.md` for executed evidence and interpretation.

## Question and operational mapping

Paper Q6 asks whether LinearRAG remains effective under different sentence
embedding backbones. The released implementation uses the same
`SentenceTransformer` instance for questions, question entities, passages,
sentences, and graph entities, so this experiment replaces that shared retrieval
embedding backbone while holding all other conditions fixed.

## Design

- Dataset: `2wikimultihop`, complete 658-passage corpus.
- Sample: 100 questions selected with seed `20260921` (one tenth of the paper's
  1,000-question dataset evaluation scale).
- Models: `all-mpnet-base-v2`, `all-MiniLM-L6-v2`,
  `bge-large-en-v1.5`, and `e5-large-v2`.
- Retrieval: official BFS path, Top-5.
- Fixed parameters: `max_iterations=3`, `top_k_sentence=1`, `delta=0.4`,
  and `lambda=0.05`.
- Generator and evaluator: `qwen3.8-flash`.
- Primary metrics: LLM accuracy, containment accuracy, and their mean.
- Mechanism evidence: Top-5 overlap and ranking changes against
  `all-mpnet-base-v2`, activated/propagated entities, and retrieval time.

Each model receives an isolated passage/entity/sentence embedding cache. Only
the model-independent NER result is copied from the verified 2Wiki cache.

## Command

```powershell
$env:PYTHONHASHSEED='0'
.venv\Scripts\python.exe experiments\q6_embedding_model_robustness\embedding_model_robustness.py --max-questions 100 --experiment-id q6-formal-100q-20260921
```

Primary raw evidence:
`data/output/experiment_results/q6_embedding_model_robustness/q6-formal-100q-20260921/`
