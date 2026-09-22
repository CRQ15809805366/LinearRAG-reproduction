# Q8 Paper Case Results - Single 2WikiMultiHopQA Example

## Execution status

The single paper case completed on 2026-09-21. The run used the existing
2WikiMultiHopQA cache in read-only mode and did not rebuild or rewrite the
index.

Raw evidence:
`data/output/experiment_results/q8_case_study/q8-paper-case-20260921/result.json`

The model produced `German` for gold answer `Germany`. The exact raw containment
field is `false`, but the response is semantically equivalent for this
nationality question.

## Scope and fixed conditions

- Dataset: `2wikimultihop`.
- Question ID: `3a3c2efe0bdc11eba7f7acde48001122`.
- Question: `What nationality is Beatrice I, Countess Of Burgundy's husband?`
- Gold answer: `Germany`.
- Embedding model: `all-mpnet-base-v2`.
- Generator: `qwen3.8-flash`.
- Retrieval: official BFS path, Top-5.
- Parameters: `max_iterations=3`, `iteration_threshold=0.4`,
  `passage_ratio=0.05`, and `top_k_sentence=1`.
- Index state: existing cache restored read-only.

No baseline, batch evaluation, or other GraphRAG comparison was executed.

## Command

```powershell
.venv\Scripts\python.exe experiments\q8_case_study\case_study.py
```

## Entity-propagation trace

| Step | Source entity | Selected sentence | Sentence score | Activated entities |
|---:|---|---|---:|---|
| 0 | `beatrice i` | Seed entity | 1.0000001 | Seed only |
| 1 | `beatrice i` | Beatrice I was Holy Roman Empress by marriage to Frederick Barbarossa | 0.7454857 | `holy roman`, `1148`, `frederick barbarossa`, `beatrice i`, the date span, and `burgundy`, each at `0.7454858` |
| 2 | `holy roman` | Beatrice I was the daughter of Otto I and Adelaide of Italy | 0.4219636 | None above threshold |
| 2 | `1148` | A sentence about Urraca of Portugal | 0.3766946 | None above threshold |
| 2 | `frederick barbarossa` | Frederick Barbarossa was Holy Roman Emperor from 1155 | 0.2433424 | None above threshold |
| 2 | `beatrice i` | Otto I was the fourth son of Frederick I and Beatrice I | 0.5916153 | Five entities at `0.4410408` |
| 2 | `burgundy` | Beatrice of Navarre was Duchess of Burgundy by marriage to Hugh IV | 0.7726580 | Five entities at `0.5760056` |

The local trace activated `frederick barbarossa` in iteration 1. It did not
activate `germany` as a graph entity in iteration 2. The Frederick Barbarossa
sentence selected in iteration 2 received a propagated sentence score of
`0.2433424`, below the `0.4` threshold.

## Top-5 retrieval

| Rank | Source passage index | Passage score | Evidence contribution |
|---:|---:|---:|---|
| 1 | 28 | 0.0714092 | Contains the Beatrice I and Frederick Barbarossa marriage evidence |
| 2 | 27 | 0.0684109 | Contains Frederick Barbarossa and Germany-related evidence |
| 3 | 590 | 0.0181357 | No key evidence for this case |
| 4 | 29 | 0.0130645 | No key evidence for this case |
| 5 | 26 | 0.0125601 | No key evidence for this case |

The two required facts appeared in the first two ranked passages. The first
passage contains the bridge from Beatrice I to Frederick Barbarossa. The second
passage contains the connection between Frederick Barbarossa and Germany.

## Answer and cache integrity

| Field | Value |
|---|---|
| Gold answer | `Germany` |
| Predicted answer | `German` |
| Exact containment | `false` |
| Semantic case outcome | Correct nationality answer |

The existing experiment record states that all five pre-existing cache files
retained identical SHA-256 hashes before and after execution. The case run did
not rebuild or modify the cache.

## Paper comparison and limits

The run matches the paper's overall case outcome: it retrieves both supporting
facts in the first two passages and produces a correct nationality answer. It
also reproduces the semantic bridge from Beatrice I to Frederick Barbarossa.

It does not exactly reproduce the paper's displayed final entity activation
`Germany`. The local run recovered the second fact through passage ranking even
though the propagated score for the relevant Frederick Barbarossa sentence fell
below threshold. This is a one-case mechanism demonstration, not a general
accuracy claim or a reproduction of the paper's HippoRAG2 comparison.
