# Q8 case study

## Scope

This is a minimal reproduction of the single 2WikiMultiHopQA example in paper
Table 7. It runs only question `3a3c2efe0bdc11eba7f7acde48001122`
(`What nationality is Beatrice I, Countess Of Burgundy's husband?`) through
the default BFS LinearRAG path and records the selected entity-propagation
steps, Top-5 passages, and generated answer.

No baseline or batch evaluation is included. The existing
`data/output/cache/2wikimultihop/` artifacts are restored read-only in memory;
the experiment does not call `LinearRAG.index()`.

## Command and evidence

```powershell
.venv\Scripts\python.exe experiments\q8_case_study\case_study.py
```

Raw result:

`data/output/experiment_results/q8_case_study/q8-paper-case-20260921/result.json`

Detailed results:
`RESULTS_2026-09-21_PAPER_CASE.md`.

Configuration: `all-mpnet-base-v2`, `qwen3.8-flash`, default BFS retrieval,
`max_iterations=3`, `iteration_threshold=0.4`, `passage_ratio=0.05`,
`top_k_sentence=1`, and Top-5 retrieval.

## Executed result

- Seed entity: `beatrice i` (score `1.0`).
- Iteration 1 selected the sentence stating that Beatrice I became Holy Roman
  Empress by marriage to Frederick Barbarossa. It activated
  `frederick barbarossa` with score `0.7455`.
- The local trace did not activate `Germany` as an entity in the next
  iteration. The selected Frederick Barbarossa sentence at that step was the
  introductory emperor sentence, whose propagated score fell below the `0.4`
  threshold.
- Ranked passage 1 contained the Beatrice I--Frederick Barbarossa marriage
  evidence.
- Ranked passage 2 contained the Frederick Barbarossa--Germany evidence.
- The local model answered `German`. This is semantically correct for the
  gold answer `Germany`, although the raw exact substring field is false.
- All five pre-existing cache files retained identical SHA-256 hashes before
  and after execution. No cache was rebuilt or rewritten.

## Paper comparison and limit

The local run matches the paper's overall case outcome: it retrieves both
supporting facts in the first two passages and produces the correct answer. It
also reproduces the key semantic bridge from Beatrice I to Frederick
Barbarossa. It does not exactly reproduce the paper's displayed final entity
activation `Germany`; the second fact is recovered at passage ranking time
instead. This is a one-case mechanism demonstration, not a general accuracy
claim or a reproduction of the paper's HippoRAG2 comparison.
