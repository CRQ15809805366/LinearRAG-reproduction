# Q3 Ablation Study

## Status

The first-stage 10-question design check completed on 2026-09-20.

The bounded formal 100-question-per-dataset run also completed on 2026-09-20.
See `RESULTS_2026-09-20_10PCT.md` for the protocol, execution incident, results,
and interpretation boundaries.

Primary raw evidence:
`data/output/experiment_results/q3_ablation_study/q3-design-check-clean-20260920/`

The run used `PYTHONHASHSEED=0` and executed all three variants in one process.
This kept Python set iteration order shared across variants. The propagated
entity counts matched question by question between the full variant and the
variant without global importance aggregation.

## Question and bounded scope

Paper Q3 asks how much each core LinearRAG component contributes to overall
performance. This first-stage check compares three variants on ten fixed-seed
2WikiMultiHopQA questions:

1. `full`: original BFS entity propagation followed by personalized PageRank;
2. `without_entity_activation`: query seed entities only, with PPR retained;
3. `without_global_importance`: entity propagation retained, with passages
   ranked directly by the existing initial passage weights instead of PPR.

The paper does not publish the ablation implementation. The third variant is
therefore an operational interpretation based on the released source: it keeps
`calculate_passage_scores()` intact and removes only `run_ppr()`.

## Controls

- Dataset: `2wikimultihop`, complete 658-passage corpus.
- Sample: ten questions selected by seed `20260920`.
- Embedding model: `data/input/models/all-mpnet-base-v2` on CUDA.
- NER model: `en_core_web_trf`.
- Generator and evaluator: `qwen3.8-flash`, thinking disabled by the existing
  client configuration.
- Retrieval: official BFS path and top five passages.
- Parameters: `max_iterations=3`, `passage_ratio=0.05`,
  `iteration_threshold=0.4`, `top_k_sentence=1`.
- Shared warm cache: `data/output/cache/2wikimultihop/`.

## Command

```powershell
$env:PYTHONHASHSEED='0'
.venv\Scripts\python.exe experiments\q3_ablation_study\ablation_study.py --experiment-id q3-design-check-clean-20260920
```

## Executed results

| Variant | LLM accuracy | Contain accuracy | Mean accuracy |
|---|---:|---:|---:|
| Full | 1.00 | 0.80 | 0.90 |
| Without entity activation | 1.00 | 0.80 | 0.90 |
| Without global importance | 1.00 | 0.80 | 0.90 |

All ten questions used graph retrieval and returned exactly five passages.
The full variant called PPR ten times. The entity-activation ablation produced
zero propagated entities for every question while retaining ten PPR calls. The
global-importance ablation preserved the full variant's propagated-entity
counts and made zero PPR calls.

The unchanged answer metrics do not mean that the modules had no effect on
retrieval:

- without entity activation, nine of ten Top-5 sets changed and mean Top-5
  overlap with the full variant was `0.74`;
- without global importance, all ten Top-5 sets changed and mean Top-5 overlap
  was `0.48`;
- generated answer strings were unchanged for all ten questions in both
  ablations.

## Runtime incident and recovery

The first end-to-end attempt exhausted CPU memory while loading the third spaCy
Transformer instance in the same process. The entry point was updated to
release each completed model and to support `--resume`. A resumed run proved
that the third variant itself was executable. Because that recovery crossed a
process boundary, it was not used as the primary comparison. The clean run
above then completed all variants in one process with explicit cleanup.

### Retired recovery output

The earlier output directory
`data/output/experiment_results/q3_ablation_study/q3-design-check-20260920/`
was retired on 2026-09-22 after its failure and recovery facts were
consolidated here. It was not the primary comparison.

- The run started at 2026-09-20 10:11:03. Its initial end-to-end process
  exhausted CPU memory while loading the third spaCy Transformer instance.
- Recovery resumed at 2026-09-20 10:15:36 and the resumed output reached
  `status=passed` at 10:16:01. The clean run was then executed separately at
  10:17:08-10:19:20.
- The retired directory used the shared
  `data/output/cache/2wikimultihop/` cache. It contained no isolated cache,
  model artifact, or other large generated dependency.
- Before removal it contained 16 files totaling 722,282 bytes (0.688822 MiB)
  and no reparse points.
- Key file SHA-256 values were: `manifest.json`
  `9FC34EA0BBDEF6C6E22E2C21DEDE60D9F9FA8E5A46487A873A34BC0B07E399DB`;
  `summary.json`
  `4D740B9B78A5A68EB0B264AB005AA329E7A0DCEBF188026D8ACD11AF1E960D31`;
  `experiment.log`
  `9D48EB822E6A307319B8C197B8F7FF685170BF2E49C225855B61AED60D68A35E`.
- The retired and clean runs had identical answer-level metrics. Their
  propagated-entity counts differed for question 7 of 10 in the `full` variant
  (`35` versus `31`) and the `without_global_importance` variant (`33` versus
  `31`); all other per-question counts matched. This supports retaining the
  clean one-process run as the primary evidence instead of treating the two
  runs as interchangeable.

## What this establishes

Executed evidence establishes that the two ablations are runnable and that the
intended modules were actually removed while the complementary module remained
active. It also shows that both modules materially changed retrieved context on
this sample, while `qwen3.8-flash` produced the same answers from all three
contexts.

This does not establish the paper's reported Q3 performance drops, statistical
significance, behavior on other datasets, or a general conclusion that the
modules do or do not improve answer accuracy. Ten questions are a pipeline and
mechanism check, not a formal ablation result.
