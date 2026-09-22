# Q5 Hyper-parameter Sensitivity

## Status

The first-stage ten-question design check and bounded formal 100-question run
completed on 2026-09-20. See `RESULTS_2026-09-20_10Q.md` and
`RESULTS_2026-09-20_100Q.md` for executed evidence and interpretation.

## Question and operational mapping

Paper Q5 studies sensitivity to the dynamic-pruning threshold `delta` and the
initial-passage trade-off coefficient `lambda` on 2WikiMultiHopQA. In the
released implementation these map to `LinearRAGConfig.iteration_threshold`
and `LinearRAGConfig.passage_ratio`, respectively.

The paper body states `delta = 4`, while Figure 5, the recovered upstream
dataset parameters, and the released implementation use `0.4`. This experiment
therefore treats `0.4` as the baseline and records the paper text as a probable
typographical error.

## First-stage design

- Dataset: `2wikimultihop`, complete 658-passage corpus.
- Sample: ten questions selected with seed `20260920`.
- Retrieval: official BFS path, top five passages.
- Fixed parameters: `max_iterations=3`, `top_k_sentence=1`.
- Baseline: `delta=0.4`, `lambda=0.05`.
- Delta sweep: `0.1` through `0.9` in steps of `0.1`, with lambda fixed.
- Lambda sweep: `0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0`, with delta fixed.
- Primary metric: mean of LLM accuracy and containment accuracy.
- Mechanism evidence: active/propagated entity counts, graph-search time,
  Top-5 overlap, ranking changes, and answer changes against the baseline.

The intermediate sweep values are inferred from Figure 5 because the paper
does not publish a complete parameter table for Q5.

## Command

```powershell
$env:PYTHONHASHSEED='0'
.venv\Scripts\python.exe experiments\q5_hyperparameter_sensitivity\hyperparameter_sensitivity.py --max-questions 10 --experiment-id q5-design-check-20260920
```

Primary raw evidence:
`data/output/experiment_results/q5_hyperparameter_sensitivity/q5-design-check-20260920/`

Bounded formal command:

```powershell
$env:PYTHONHASHSEED='0'
.venv\Scripts\python.exe experiments\q5_hyperparameter_sensitivity\hyperparameter_sensitivity.py --max-questions 100 --max-workers 16 --experiment-id q5-formal-100q-20260920
```

Bounded formal raw evidence:
`data/output/experiment_results/q5_hyperparameter_sensitivity/q5-formal-100q-20260920/`
