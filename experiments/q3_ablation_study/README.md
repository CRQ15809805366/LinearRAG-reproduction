# Q3 Ablation Study

`formal_ablation.py` is the canonical implementation for Q3. It contains the three retrieval variants, design validation, prediction evaluation, and summary generation. The formal runner and the first-stage design check share this implementation, so variant behavior stays aligned.

Run the bounded formal experiment from the repository root:

```powershell
python -m experiments.q3_ablation_study.formal_ablation --experiment-id q3-formal-<run-id>
```

By default, this runs 100 fixed-seed questions for each of HotpotQA, 2WikiMultiHopQA, MuSiQue, and Medical. To run selected datasets or change the sample size, pass `--datasets` and `--max-questions`. Use a new `--experiment-id` for a fresh output directory; add `--resume` only to continue an existing run directory.

The first-stage design check remains available as a smaller workflow:

```powershell
python -m experiments.q3_ablation_study.ablation_study --retrieval-only
```

Its default is 10 fixed-seed 2Wiki questions. Formal evidence is stored under `data/output/experiment_results/q3_ablation_study/<experiment-id>/`; the completed bounded run is `q3-formal-10pct-20260920/`. Design-check evidence is under `q3-design-check-clean-20260920/`. See `project/agent/experiments/q3_ablation_study/RESULTS_2026-09-20_10PCT.md` for the existing run's methods, findings, and limits.

The local run compares the full method with two operational ablations. Results are bounded local evidence using `qwen3.8-flash`; they do not reproduce the paper's 1,000-question GPT-4o-mini evaluation.
