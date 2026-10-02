# Q2 Efficiency Analysis

## Status

The shared HotpotQA supplement is prepared, not executed: see
`PREPARED_2026-10-01_SHARED_CORPUS.md`. It consumes the same fixed 20-question /
250-passage package as new Q1 and prepares LinearRAG/HippoRAG efficiency
measurement with isolated index caches. The historical 2Wiki evidence below is
unchanged and must not be combined with the new supplement.

The first-stage design check and the bounded formal retrieval experiment
completed on 2026-09-20.

Raw evidence:
`data/output/experiment_results/q2_efficiency_analysis/q2-design-check-20260920/`

Formal retrieval evidence:
`data/output/experiment_results/q2_efficiency_analysis/q2-formal-100q-20260920/`

Additional cold-index evidence:
`data/output/experiment_results/q2_efficiency_analysis/q2-cold-index-repeat-2-20260920/`
and
`data/output/experiment_results/q2_efficiency_analysis/q2-cold-index-repeat-3-20260920/`.

See `RUN_2026-09-20_100Q.md` for the formal protocol and results.

## Question and bounded scope

Paper Q2 asks how cost-efficient and time-efficient LinearRAG is relative to
existing GraphRAG methods. This first-stage check measures only the local
LinearRAG implementation. It does not execute or compare GraphRAG baselines.

The run used the complete 2WikiMultiHopQA corpus of 658 passages and ten
fixed-seed questions. It measured one isolated cold index build, one untimed
retrieval warm-up, and two timed retrieval repetitions on the official BFS
path. Answer generation and LLM evaluation were excluded.

## Protocol

- Embedding model: `data/input/models/all-mpnet-base-v2` on CUDA.
- NER model: `en_core_web_trf`.
- Retrieval: official BFS, top 5.
- Parameters: `max_iterations=3`, `passage_ratio=0.05`,
  `iteration_threshold=0.4`, `top_k_sentence=1`.
- Cold cache isolated below the immutable run directory.
- CUDA synchronization immediately before and after timed calls.
- `NoLLM` fail-fast object used to detect any generative-model call.

## Executed results

| Measurement | Result |
|---|---:|
| Setup time (excluded from index time) | 3.743 s |
| Cold indexing time | 612.697 s |
| Retrieval repeat 1 | 0.148100 s/question |
| Retrieval repeat 2 | 0.156323 s/question |
| Median retrieval time | 0.152212 s/question |
| LLM calls during indexing and retrieval | 0 |
| Prompt/completion tokens | 0 / 0 |

All ten questions returned exactly five passages. During indexing, the isolated
cold cache held 658 passage embeddings, 41,243 entity embeddings, 21,023
sentence embeddings, NER mappings for all 658 passages, and a generated GraphML
file. The temporary per-run caches have since been removed; measured timing
records and run metadata remain.

The paper reports 249.78 seconds for indexing and 0.093 seconds per retrieval
on different hardware and a 1,000-question protocol. The local values are not
directly comparable because this design check uses an NVIDIA GeForce RTX 4050
Laptop GPU, ten sampled questions, and only one cold-index repetition.

## Command

```powershell
.venv\Scripts\python.exe experiments\q2_efficiency_analysis\efficiency_analysis.py --experiment-id q2-design-check-20260920
```

## What this establishes

Executed evidence establishes that the local original BFS implementation can
build a fresh full-2Wiki index and retrieve top-five passages without invoking
a generative LLM. It also establishes the observed local timings for this run.

It does not establish comparative superiority over other GraphRAG systems,
stable timing distributions, the paper's 1,000-question result, or large-scale
linear scalability.

Live Q2 attempt on 2026-10-01 is partial: see `RUN_2026-10-01_SHARED_CORPUS_PARTIAL.md`. LinearRAG completed on the shared HotpotQA input; HippoRAG retained 192/250 extractions but repeated API timeout prevented indexing/retrieval completion. No comparative result is available.

Final shared-corpus state: completed with recovery; see `RESULTS_2026-10-01_SHARED_CORPUS.md`. Both methods completed first-query and warm retrieval timing. HippoRAG full cold-index time is unavailable; the last passage used a documented output-budget/streaming recovery.
