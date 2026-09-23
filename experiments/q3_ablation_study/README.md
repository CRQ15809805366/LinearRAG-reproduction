# Q3 Ablation Study

This directory contains the Q3 experiment implementations. `ablation_study.py` runs the ten-question design check; `formal_ablation.py` runs the bounded 100-question-per-dataset comparison.

The bounded formal run completed for HotpotQA, 2WikiMultiHopQA, MuSiQue, and Medical. Its primary evidence is under `data/output/experiment_results/q3_ablation_study/q3-formal-10pct-20260920/`; design-check evidence is under `data/output/experiment_results/q3_ablation_study/q3-design-check-clean-20260920/`. The Agent-facing results and interpretation are in `project/agent/experiments/q3_ablation_study/RESULTS_2026-09-20_10PCT.md`.

The local run compares the full method with two operational ablations. Results are bounded local evidence using `qwen3.8-flash`; they do not reproduce the paper's 1,000-question GPT-4o-mini evaluation. See the Agent record for definitions and limits.
