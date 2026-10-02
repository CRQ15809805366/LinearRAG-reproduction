# Experiment code

`corpus_construction/` prepares shared, fixed small-corpus input packages under `data/input/derived_corpora/` without API calls. Q1 can read these packages and explicitly select original HippoRAG 2024 alongside Vanilla RAG and LinearRAG. Preparation is not an executed baseline comparison.

This directory contains the bounded Q1–Q8 experiment adapters. Each question's subdirectory holds its executable script and, where present, a short usage guide. The adapters define experiment-specific sampling and measurement around the reproduced implementation in `src/`; they are not a second implementation of LinearRAG.

Generated run evidence belongs under `data/output/experiment_results/<question>/`. Start with `project/agent/experiments/README.md` to identify completed runs, their result records, and the limits on interpreting them. Use a new experiment ID for every new run; do not overwrite recorded evidence.

The supplemental external HippoRAG 2024 driver and minimal integration fixture live in `hipporag/`. Their setup and usage are documented in `hipporag/README.md`; raw integration evidence is under `data/output/hipporag/`. Q1 now offers HippoRAG as an explicit method option; no formal dataset comparison has been executed.

## Function reuse between experiments

The following dependencies were checked against the Python imports on 2026-09-30. The consumer imports code from the provider; it does not require a completed provider experiment run.

| Consumer | Provider | Imported symbols | Purpose |
|---|---|---|---|
| Q3 `q3_ablation_study/ablation_study.py` | Q3 `q3_ablation_study/formal_ablation.py` | `AblationLinearRAG`, `VARIANTS`, `build_summary`, `evaluate_predictions`, `validate_retrieval`, `write_json` | Keep the small design check aligned with the formal ablation implementation, validation, evaluation and reporting. |
| Q5 `q5_hyperparameter_sensitivity/hyperparameter_sensitivity.py` | Q3 `q3_ablation_study/ablation_study.py` | `evaluate_predictions`, `write_json` | Reuse answer evaluation and result persistence. Both names are imported into the provider from `formal_ablation.py`, so this is an indirect dependency on that file. |
| Q5 `q5_hyperparameter_sensitivity/hyperparameter_sensitivity.py` | Q3 `q3_ablation_study/formal_ablation.py` | `qa_with_isolated_failures` | Reuse answer generation with per-question failure isolation. |
| Q6 `q6_embedding_model_robustness/embedding_model_robustness.py` | Q3 `q3_ablation_study/formal_ablation.py` | `compare_retrievals`, `evaluate_predictions`, `write_json` | Reuse retrieval-result comparison, answer evaluation and result persistence. |

Q1, Q2, Q4, Q7 and Q8 currently have no Python imports from another experiment script. Imports from `src/` are shared implementation dependencies and are outside this table.

This arrangement is acceptable for the current bounded project, but Q3 is also serving as a helper provider for Q5 and Q6. Moving or changing its exported helpers can break those consumers or change their measurements. In particular, `evaluate_predictions` defines the reported average accuracy as `(llm_accuracy + contain_accuracy) / 2`; it is not merely a file-writing helper.

When changing a provider helper, inspect its consumers and update this table. Keep sampling, parameters and experiment-specific decisions in their own question directories. Extract a small stable helper into `experiments/common/` only when actual reuse warrants it; no such extraction is performed here.
