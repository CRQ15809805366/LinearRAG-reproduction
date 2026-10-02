# Small-corpus Q1 — completed replacement run, 2026-10-01

Run: `data/output/experiment_results/q1_generation_accuracy/q1-hotpotqa-context20-noise50-20261001-r2/`. Raw summary: `summary.json`; per-method predictions, evaluations and `hotpotqa/comparison.json` retain the evidence. All 20 question IDs/text/gold answers and Top-5 corpus membership were checked.

User explicitly authorized modifying the input directly. Random distractor at chunk index 227, "Taichung International Airport", was replaced with "Boltzmann constant". Replacement was drawn with fixed seed `20261001:replacement:227` from the remaining source candidate documents, excluding existing corpus texts. No sampled question or its candidate contexts changed. Input manifest records replacement provenance, updated hashes and statistics. API rejection did not establish that airports were the triggering category.

Same 20 HotpotQA questions / 250 documents; 139402 characters, 22786 whitespace words. All three methods ran on the changed shared input. `qwen3.8-flash` generation/evaluation, `all-mpnet-base-v2`, Top-5, default LinearRAG BFS; original HippoRAG 2024 configuration remains local, not paper model settings.

| Method | Contain accuracy | LLM accuracy |
|---|---:|---:|
| Vanilla RAG | 16/20 (80%) | 16/20 (80%) |
| LinearRAG | 15/20 (75%) | 16/20 (80%) |
| HippoRAG 2024 | 18/20 (90%) | 20/20 (100%) |

Runtime launcher: `data/output/run_q1_r2.py`. Separate method processes and loading the real spaCy model before embeddings avoided memory peaks. HippoRAG reused 231 exact-text matching extraction records from the incomplete original run; no graph was transferred. It extracted the remaining 19, built the new graph and retrieved/generated/evaluated all 20 questions. `hotpotqa/hipporag/cache_reuse.json` records source/target cache and driver snapshot identity. Historical usage copied with the extraction checkpoint represents acquisition cost, not new calls. The frozen driver uses inherited system proxy and a 120-second HTTP timeout; prompts, text and extraction/generation models were unchanged.

New successful indexing/query-NER usage in this run: 25567 + 2651 = 28218 tokens, excluding answer generation/evaluation. No currency estimate. Shared cache references and frozen runner are retained in the method directory; normal adapter routing uses the current driver fingerprint, so later cross-experiment reuse must account for this snapshot rather than accidentally selecting a fresh cache.

These are one 20-question favorable-setting sample results. HippoRAG had higher observed accuracy here; no significance, universal superiority or full-paper reproduction claim is made. Original failed-run results remain separate. Current input overrides the original builder output and is reproducible using its manifest's explicit replacement seed/record.
