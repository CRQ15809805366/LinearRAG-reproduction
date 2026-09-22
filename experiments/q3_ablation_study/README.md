# Q3 Ablation Study

Run the first-stage three-variant check on ten fixed 2WikiMultiHopQA questions:

```powershell
.venv\Scripts\python.exe experiments\q3_ablation_study\ablation_study.py --experiment-id q3-design-check-20260920
```

If a long run is interrupted after a variant completes, rerun the same command
with `--resume`; completed variants are loaded from their saved evidence.

Use `--retrieval-only` to validate the three retrieval paths before making any
generation or evaluator calls. Raw results are written below
`data/output/runs/q3_ablation_study/<experiment-id>/`.
