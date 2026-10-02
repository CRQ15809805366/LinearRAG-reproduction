# Q4 three-method small-corpus run — 2026-10-01

Completed, status `passed`, 12 questions (3/task), 30 unchanged Medical source passages, Top-5. Fixed package: `data/input/derived_corpora/medical-type3-noise5-s20261001/`. Extraction and judge: `qwen3.8-flash`; embedding: local `all-mpnet-base-v2`; LinearRAG official BFS.

Evidence: `data/output/experiment_results/q4_retrieval_quality/q4-medical-small-ready-20261001/`. `summary.csv` is the parallel table; each method has complete retrievals and 12 complete per-question evaluations. All reference items were classified; each question has two relevance ratings.

| Method | Macro context relevance | Macro evidence recall |
|---|---:|---:|
| Vanilla RAG | 0.562500 | 0.551085 |
| LinearRAG | 0.645833 | 0.698796 |
| HippoRAG 2024 | 0.729167 | 0.631009 |

Runtime repairs: Q4 attaches IDs after checking ordered question text because official LinearRAG retrieval omits IDs. HippoRAG driver accepts JSON as well as Python literals; length-invalid triple candidates are passed to official create_graph's existing discard/report behavior. Failed passage extraction cancels not-yet-started tasks while collecting/persisting other completed work. Driver-hash changes required carrying forward matching extraction checkpoints and usage logs; inputs/configuration were checked unchanged, and core algorithms were not edited.

Ten per-question evidence judgments were initially incomplete. Only missing evidence classifications were supplemented using a single-reference JSON prompt (`reason`, `attributed`); existing relevance scores and valid classifications were retained. The final run resumed saved retrievals/checkpoints and made no repeated retrieval or completed-question judgments. Raw original and supplement responses remain in evaluation output. Failed runs produced additional extraction calls; initial base-call estimates are not actual billing totals. Generation/judge tokens are outside HippoRAG's extraction/query-NER token ledger.

Scope: constructed candidate-selected corpus, not full Medical or paper Table 4. Partial source review is not complete gold support certification; hypothetical reference details may be unsupported. This run establishes live three-method execution and bounded observed scores, not a stable ranking or a general claim of superiority.
