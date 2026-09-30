# Experiment index and Agent guide

This directory is the internal working index for the project's Q1–Q8 research questions. It is written for Agents who need to understand what each experiment studies, where its code and evidence live, what has actually been completed, and which limits must be preserved when summarizing results.

## Evidence map

- `experiments/` at the repository root contains the executable experiment adapters. They call the reproduced implementation in `src/` and define sampling, controls, measurements, and result output.
- `data/output/experiment_results/` contains generated predictions, retrieval records, measurements, summaries, and logs. Treat this as raw evidence; do not move bulk output into this Agent directory.
- `data/output/cache/` contains shared rebuildable caches. Some cold-index runs created isolated caches beneath their result directories; those temporary caches have been removed after retaining the measurement records.
- `project/agent/experiments/` contains this index and concise experiment records that explain design, executed evidence, and interpretation boundaries.
- `research_report/` is for material intended for readers outside the Agent workflow. Quantitative claims there must point through an experiment record to raw evidence.

When auditing a run, first read the question's README and any linked `RESULTS_*.md` or `RUN_*.md`; then inspect only the raw files needed to answer the question. Classify claims as executed evidence, static analysis, or inference. A design check validates a workflow or mechanism at a small sample and is not interchangeable with a bounded formal run. A local bounded run is not automatically a full-paper reproduction.

## Current registry

| Question | Current state and scope | Code entry point | Agent record | Main raw evidence |
|---|---|---|---|---|
| Q1 Generation accuracy | Completed bounded comparison: 100 questions for each of four datasets; local `qwen3.8-flash`, not the paper's 1,000-question GPT-4o-mini protocol. | `experiments/q1_generation_accuracy/generation_accuracy.py` | `q1_generation_accuracy/RESULTS_2026-09-17_10PCT.md` | `data/output/experiment_results/q1_generation_accuracy/q1-formal-10pct-20260917-r2/` |
| Q2 Efficiency | Completed 100-question retrieval timing plus three independent isolated cold-index observations. LinearRAG-only; no GraphRAG timing comparison. | `experiments/q2_efficiency_analysis/efficiency_analysis.py` | `q2_efficiency_analysis/RUN_2026-09-20_100Q.md` | `data/output/experiment_results/q2_efficiency_analysis/` |
| Q3 Ablation | Completed bounded run: 100 questions per dataset across four datasets and three variants. Local results vary by dataset and do not reproduce the paper's full-scale result. | `experiments/q3_ablation_study/formal_ablation.py` | `q3_ablation_study/RESULTS_2026-09-20_10PCT.md` | `data/output/experiment_results/q3_ablation_study/q3-formal-10pct-20260920/` |
| Q4 Retrieval quality | Completed balanced 200-question Medical comparison of Vanilla RAG and LinearRAG using local LLM judgments. Does not reproduce Table 4 or compare all paper baselines. | `experiments/q4_retrieval_quality/retrieval_quality.py` | `q4_retrieval_quality/RUN_2026-09-20_200Q.md` | `data/output/experiment_results/q4_retrieval_quality/q4-formal-200q-20260920/` |
| Q5 Parameter sensitivity | Completed 100-question 2Wiki OFAT sweeps for `delta` and `lambda`; one warm index and one run per point. No interaction test or replicated optimum estimate. | `experiments/q5_hyperparameter_sensitivity/hyperparameter_sensitivity.py` | `q5_hyperparameter_sensitivity/RESULTS_2026-09-20_100Q.md` | `data/output/experiment_results/q5_hyperparameter_sensitivity/q5-formal-100q-20260920/` |
| Q6 Embedding robustness | Completed 100-question 2Wiki comparison across four embedding models. Local bounded evidence; does not establish a universal best model. | `experiments/q6_embedding_model_robustness/embedding_model_robustness.py` | `q6_embedding_model_robustness/RESULTS_2026-09-21_100Q.md` | `data/output/experiment_results/q6_embedding_model_robustness/q6-formal-100q-20260921/` |
| Q7 Large-scale efficiency | Completed 25K, 50K, 250K, and 500K token cold-index runs on HotpotQA scale proxies. The 1M attempt did not complete and has no retained output. | `experiments/q7_large_scale_efficiency/large_scale_efficiency.py` | `q7_large_scale_efficiency/RESULTS_2026-09-21_SCALE_SERIES.md` | Four completed run directories under `data/output/experiment_results/q7_large_scale_efficiency/` |
| Q8 Case study | Completed one 2Wiki paper case with the existing cache and a recorded BFS propagation trace. This is a mechanism example, not an accuracy estimate. | `experiments/q8_case_study/case_study.py` | `q8_case_study/RESULTS_2026-09-21_PAPER_CASE.md` | `data/output/experiment_results/q8_case_study/q8-paper-case-20260921/result.json` |

Each question directory contains its design, detailed evidence, commands, and limits. Read the linked result record before reusing numerical claims. The full local runs mostly use `qwen3.8-flash`; distinguish those from the paper's model and sample sizes.

The static Q3/Q4 paper-to-local comparison, including Figure 4/Table 4 source values, protocol mismatches, and old-run alignment checks, is recorded in `Q3_Q4_DISCREPANCY_AUDIT_GATE1.md`. It is a gate-1 audit only; it does not claim residual algorithmic causes or repeatability.

## Maintenance rules

- Keep the reproduced algorithms unchanged unless the user explicitly requests a behavioral change.
- Keep Agent-maintained files in English. Preserve exact paths, run IDs, commands, and model names.
- Keep raw predictions, logs, measurements, and caches under `data/output/`; keep only concise interpretation and operational context here.
- Retain files that support a research claim, learning objective, or reader-facing explanation. Remove empty placeholders, checkpoints after final results are saved, and isolated temporary caches once their recorded measurements are sufficient.
- Keep limitations that affect the interpretation of a result. Omit process-recovery chronology that does not help understand the research question or its evidence.
- Do not describe a bounded local result as a full-paper reproduction or compare against baselines that were not run.
