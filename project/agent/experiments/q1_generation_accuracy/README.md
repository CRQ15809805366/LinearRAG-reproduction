# Q1 Generation Accuracy

## Status

The first bounded formal Q1 run completed on 2026-09-17. See
`RESULTS_2026-09-17_10PCT.md` for the protocol, execution history, and results.

On 2026-09-17, a four-dataset 10-question preparation check completed and
wrote `data/output/experiment_results/q1_generation_accuracy/design-check-20260917/manifest.json`.
A one-question 2Wiki end-to-end check then stopped before retrieval and
generation because `OPENAI_API_KEY` was not present in the process environment.
The local `all-mpnet-base-v2` model loaded successfully. This is environment
evidence only, not a Q1 result.

The credentials were subsequently found in Windows user-level environment
variables and injected into the child process without exposing their values.
That was the mechanism in effect on 2026-09-17; since 2026-09-21 both values
live in the untracked project-root `.env.local` instead (see
`../../history/ENVIRONMENT_MIGRATION_2026-09-21.md`).
The one-question check was rerun with `qwen3.8-flash` and completed for both
Vanilla RAG and the official BFS LinearRAG path. Both methods scored 1/1 on
contain accuracy and LLM accuracy. Raw evidence is under
`data/output/experiment_results/q1_generation_accuracy/end-to-end-qwen-check-20260917/`.
This single case verifies the execution path only and is not a Q1 finding.

## Question and bounded scope

Paper Q1 asks how LinearRAG compares with state-of-the-art GraphRAG methods in
generation performance. This bounded reproduction tests a narrower claim:
whether the original LinearRAG implementation outperforms a frozen Vanilla RAG
baseline under matched questions, corpus, embedding model, generator, prompt,
and top-5 context budget.

It does not support a claim that LinearRAG outperforms all GraphRAG baselines.

## Controls

- Deterministic, saved sample IDs shared by both methods.
- `all-mpnet-base-v2` embeddings by default.
- `gpt-4o-mini` generation and LLM evaluation by default.
- Top-5 retrieval for both methods.
- The original dataset-specific LinearRAG parameters recovered from upstream.
- Official BFS retrieval; the optional vectorized path is disabled.
- Contain accuracy and GPT accuracy for HotpotQA, 2Wiki, and MuSiQue; GPT
  accuracy only is reported for Medical.

## Entry point and evidence

Entry point: `experiments/q1_generation_accuracy/generation_accuracy.py`

Reusable Vanilla RAG implementation: `src/vanilla_rag.py`

Raw evidence:
`data/output/experiment_results/q1_generation_accuracy/<experiment-id>/`

The script intentionally keeps preparation, both method runs, evaluation, and
aggregation together. Vanilla RAG is an experiment control, not a new algorithm
or a separate learning target.
