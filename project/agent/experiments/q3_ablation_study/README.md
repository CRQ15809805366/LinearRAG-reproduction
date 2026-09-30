# Q3 Ablation Study

## Status and evidence

The ten-question design check and the bounded formal run completed on 2026-09-20. The formal comparison used 100 fixed-seed questions for each of HotpotQA, 2WikiMultiHopQA, MuSiQue, and Medical, comparing the full system with two operational ablations.

Primary evidence: `data/output/experiment_results/q3_ablation_study/q3-formal-10pct-20260920/`.

Design-check evidence: `data/output/experiment_results/q3_ablation_study/q3-design-check-clean-20260920/`.

See `RESULTS_2026-09-20_10PCT.md` for the measured outcomes and limits.

## Code and design

- Canonical variant implementation, validation, evaluation, and summary: `experiments/q3_ablation_study/formal_ablation.py`.
- Bounded formal entry point: `experiments/q3_ablation_study/formal_ablation.py`.
- The design-check runner at `experiments/q3_ablation_study/ablation_study.py` imports and reuses the canonical formal implementation.
- Full variant: original BFS entity propagation followed by personalized PageRank.
- `without_entity_activation`: query seed entities only, retaining PPR.
- `without_global_importance`: keeps initial passage scoring and removes PPR.

The formal run used `all-mpnet-base-v2`, `en_core_web_trf`, `qwen3.8-flash`, official BFS retrieval, Top-5 passages, and 16 workers. The paper does not publish an ablation implementation, so the no-global-importance variant is an operational interpretation of the description.

## Main findings and boundaries

The two ablations reduced the mean accuracy score on HotpotQA and 2Wiki. Differences were small or reversed on MuSiQue and Medical. Both ablations changed retrieved contexts; this bounded evidence does not establish the paper's 1,000-question GPT-4o-mini results or a universal effect across datasets.

The small ten-question design check validates the experiment path and retrieval mechanism only. Do not use it as the Q3 performance result.
