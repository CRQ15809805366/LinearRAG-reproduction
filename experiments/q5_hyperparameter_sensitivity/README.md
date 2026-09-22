# Q5 Hyper-parameter Sensitivity

`hyperparameter_sensitivity.py` runs a fixed-sample, one-factor-at-a-time
sweep over the dynamic-pruning threshold (`delta`) and dense-passage trade-off
coefficient (`lambda`) on 2WikiMultiHopQA. It reuses one real LinearRAG index,
keeps the official BFS path, generates and evaluates answers for every point,
and writes raw evidence below `data/output/experiment_results/`.

First-stage command:

```powershell
$env:PYTHONHASHSEED='0'
.venv\Scripts\python.exe experiments\q5_hyperparameter_sensitivity\hyperparameter_sensitivity.py --max-questions 10 --experiment-id q5-design-check-20260920
```
