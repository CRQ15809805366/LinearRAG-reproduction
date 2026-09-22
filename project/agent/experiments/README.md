# Experiment registry

This directory is the internal index for formal Q1-Q8 evaluations. Create one subdirectory per research question when its design work begins.

Each question record should capture:

1. the research question and the paper claim being tested;
2. the role of the experiment in the research process;
3. controls, variables, datasets, models, parameters, and metrics;
4. exact commands and immutable paths to raw evidence under `data/output/`;
5. observed results and comparison with the paper;
6. supported explanations, plausible hypotheses, and unresolved differences;
7. conclusions, limitations, and implications for later work.

Do not copy bulk logs or generated predictions into this directory. Keep an explicit status for each conclusion: executed evidence, static analysis, or unverified inference.

## Registry

- Q1 Generation Accuracy: `q1_generation_accuracy/README.md` (entry point
  implemented; formal result not yet executed).
- Q2 Efficiency Analysis: `q2_efficiency_analysis/README.md` (first-stage
  cold-index design check completed).
- Q3 Ablation Study: `q3_ablation_study/README.md` (first-stage ten-question
  three-variant design check completed).
- Q5 Hyper-parameter Sensitivity: `q5_hyperparameter_sensitivity/README.md`
  (first-stage ten-question design check completed).
- Q4 Retrieval Quality Evaluation: `q4_retrieval_quality/README.md`
  (first-stage balanced 12-question design check completed).
- Q7 Large-scale Efficiency Analysis: `q7_large_scale_efficiency/README.md`
  (25K/50K-token cold-index design check completed on a HotpotQA scale proxy).
- Q6 Embedding Model Robustness: `q6_embedding_model_robustness/README.md`
  (four-model 100-question evaluation completed).
- Q8 Case Study: `q8_case_study/README.md` (single paper case completed with
  existing cache and a recorded BFS propagation trace).
