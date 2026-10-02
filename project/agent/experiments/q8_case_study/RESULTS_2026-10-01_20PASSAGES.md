# Q8 20-passage supplemental case

Completed the paper question `3a3c2efe0bdc11eba7f7acde48001122` on a shared 20-passage corpus: the two supporting passages plus 18 distractors sampled with seed `20261001`. Source passage IDs/text remain intact; source indices are in `prepared.json`. Both methods used `all-mpnet-base-v2` and generated with `qwen3.8-flash`. HippoRAG extraction also used `qwen3.8-flash`.

Raw evidence: `data/output/experiment_results/q8_case_study/q8-20passages-case-20261001/result.json`. Fixed-input entry: `experiments/q8_case_study/case_study.py`. Completion reused saved LinearRAG results and the HippoRAG extraction checkpoints through `data/output/resume_q8_proxy_once.py`; full entry command is `.venv\Scripts\python.exe -m experiments.q8_case_study.case_study`.

- LinearRAG default BFS: `German`; marriage and Germany support passages ranked 1 and 2.
- Original HippoRAG 2024 v1.0.0: `German`; Germany and marriage support passages ranked 1 and 2.
- HippoRAG graph: 1,939 nodes, 6,884 edges; 112 edges touching the case/linked focus nodes saved in `case_trace`.
- HippoRAG linked query entity `beatrice i  countess of burgundy` to `countess joan ii of burgundy` (similarity 0.8324581981), despite `beatrice i`, `frederick barbarossa`, and `germany` nodes existing. Correct retrieval and answer therefore do not establish correct initial entity linking.

Scope: one small-corpus case with support passages deliberately retained. Both answers are semantically correct for gold `Germany`. This is neither the paper's HippoRAG2 comparison nor general accuracy evidence; it does not reproduce the paper's baseline failure. Current client inherits the child environment proxy and uses a 180-second request timeout. No algorithm or extraction prompt was changed.
