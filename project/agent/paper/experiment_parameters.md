# LinearRAG Paper Experiment Parameters

The parameters below were recovered from the upstream repository's original `scripts/run.sh`. They are intended for running the four dataset experiments through `run.py`.

All experiments use:

- Embedding model: `data/input/models/all-mpnet-base-v2`
- LLM: `gpt-4o-mini`
- `max_workers`: `16`

| Dataset | spaCy model | `max_iterations` | `passage_ratio` | `iteration_threshold` | `top_k_sentence` |
|---|---|---:|---:|---:|---:|
| `medical` | `en_core_sci_scibert` | 3 | 1.5 | 0.5 | 1 |
| `musique` | `en_core_web_trf` | 5 | 2.0 | 0.1 | 4 |
| `2wikimultihop` | `en_core_web_trf` | 3 | 0.05 | 0.4 | 1 |
| `hotpotqa` | `en_core_web_trf` | 3 | 0.05 | 0.4 | 1 |

These are not all of `run.py`'s defaults. They are dataset-specific experiment configurations left by the authors. The configuration blocks in the original script were commented out, so the corresponding values must be passed explicitly when running a particular experiment.
