# LinearRAG minimal local smoke verification

## 1. Scope and conclusion

Working directory: `D:\code\LinearRAG-reproduction`.

The official LinearRAG core indexing and retrieval flow has passed a tiny, manually checkable, CPU-only smoke verification on this Windows machine. This is not a reproduction of the paper's benchmark results, answer generation, GPT evaluation, or scalability claims.

## 2. Source and directory structure

Official source:

- URL: https://github.com/DEEP-PolyU/LinearRAG
- Branch: `main`
- Commit: `bcc94e66c221f798801255efba09311d6fbcd8d6`
- Retrieved: 2026-09-13 (Asia/Shanghai)
- Import: exact-commit GitHub codeload archive after Git transport reset; all 15 official files were SHA-256 checked before the temporary checkout was removed.
- Audit record: `SOURCE.md`.

Final logical structure (generated caches are shown but ignored by Git):

```text
LinearRAG-reproduction/
|-- 2510.10114v4.pdf
|-- 提示词.md
|-- readme.md                       # official
|-- requirements.txt               # official
|-- run.py                         # official combined entry
|-- src/                           # official core, unchanged
|-- scripts/                       # official
|-- figure/                        # official
|-- requirements-windows.txt       # local compatibility/observation additions
|-- smoke_test.py                  # OpenAI-free smoke entry and trace observer
|-- examples/smoke/                # original and adjusted tiny inputs
|-- tools/                         # NER probe and model recovery script
|-- artifacts/                     # saved output, environment and setup evidence
|-- model/                         # ignored local embedding model
|-- import/smoke/                  # ignored official index/cache outputs
`-- .venv/                         # ignored independent Python environment
```

The paper path stated in the task was `D:\code\2510.10114v4.pdf`; the file actually supplied and used is `D:\code\LinearRAG-reproduction\2510.10114v4.pdf`.

## 3. Machine and environment

Measured, not assumed:

- Windows 11 Pro 10.0.26200 (build 26200).
- Intel Core i7-13620H, 16 logical processors.
- 15.74 GiB physical RAM; 7.86 GiB available at the initial capture.
- NVIDIA GeForce RTX 4050 Laptop GPU; 6141 MiB VRAM, 5877 MiB initially free; driver 572.83.
- System Python 3.10.11; project Python 3.9.25.
- Git 2.47.0.windows.2; uv 0.11.21.
- Free space at capture: C: 20.54 GiB, D: 267.09 GiB.

Exact snapshots are in `artifacts/system-environment.json` and `artifacts/environment-freeze.txt`. Key installed versions are NumPy 1.21.0, Pandas 1.3.0, spaCy 3.6.1, en-core-web-sm 3.6.0, SentenceTransformers 2.2.2, Transformers 4.30.2, PyTorch 2.8.0+cpu, igraph 0.11.8, PyArrow 12.0.1, and sentencepiece 0.1.99.

Environment recovery:

```powershell
uv python install 3.9
uv venv --python 3.9 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-windows.txt
uv pip install --python .venv\Scripts\python.exe "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl"
powershell -ExecutionPolicy Bypass -File tools\download_embedding_model.ps1
.venv\Scripts\python.exe smoke_test.py
```

The official pins fail on Python 3.10 because NumPy 1.21.0 has no suitable Windows/Python 3.10 wheel and its source build requires MSVC. Python 3.9 resolves the official direct pins. One transitive pin was still necessary: sentencepiece 0.2.2 caused a native access violation with the official Transformers 4.30.2 stack, so `requirements-windows.txt` pins sentencepiece 0.1.99. `psutil` is added only for memory observation. Original constraints remain untouched in `requirements.txt`; failure logs are under `artifacts/setup/`.

The official preferred spaCy model is `en_core_web_trf` (and its README specifies a SciSpaCy model for the medical dataset). This low-resource smoke run uses `en_core_web_sm`. The official embedding path is `model/all-mpnet-base-v2`; its 438 MB download repeatedly broke on the CDN connection, so the smoke run uses the API-compatible `all-MiniLM-L6-v2` on CPU, pinned to model revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and checked with model SHA-256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

## 4. What the official program actually provides

- Real combined entry: `run.py:main()`.
- Index entry: `LinearRAG.index(passages)`.
- Retrieval entry: `LinearRAG.retrieve(questions)`; BFS is the default, vectorized retrieval is optional.
- Answer generation: `LinearRAG.qa()` calls `LLM_Model.infer()` through OpenAI chat completions.
- GPT evaluation: `Evaluator.evaluate()` calls an LLM correctness judge and also computes containment accuracy.
- Dataset: `dataset/<name>/chunks.json` is a JSON list of strings; `questions.json` is a list of objects containing at least `question` and `answer`. `run.py` prefixes chunks with `index:`.
- Model loading: SentenceTransformer from a local/configured path, forced to `device="cuda"` by official `run.py`; spaCy loads by package name.
- Python: official README prefers Python 3.9; `requirements.txt` contains old exact pins.
- CUDA: official `run.py` hard-codes `CUDA_VISIBLE_DEVICES=4`; vectorized retrieval selects CUDA if available. The smoke entry does neither and explicitly uses CPU/BFS.
- Paths: official caches default to `./import/<dataset>/`; logs and predictions default to `results/<dataset>/<timestamp>/`. Relative paths work on Windows when launched from the repository root.
- OpenAI boundary: `run.py`, `qa()`, and LLM evaluation need API configuration. Passage reading, NER, embeddings, indexing, BFS/vectorized retrieval, passage initialization, PPR, and Top-k retrieval do not require OpenAI when invoked directly.

The paper presents a Tri-Graph with passage, sentence, and entity node types. In the checked commit, sentence nodes live in the sentence embedding store and sentence-entity maps used by BFS, while the actual igraph built for PPR contains only passage and entity vertices. This is recorded as an implementation detail; it was not changed.

## 5. Local changes and algorithm boundary

1. Environment adaptation: `requirements-windows.txt`, local Python 3.9 environment, local CPU embedding model, small spaCy model.
2. Tiny sample integration: `examples/smoke/*.json` and `smoke_test.py`.
3. Observability: `ObservableLinearRAG` captures query matching, official passage reset weights, PPR scores, and performs an observation-only BFS replay. It asserts the replayed entity weights/scores equal the official method's output; replayed data never feeds ranking. Environment and raw structured output are saved under `artifacts/`.
4. Algorithm changes: none. All official files, including `src/LinearRAG.py`, are byte-for-byte unchanged.

Runtime settings are deliberately small: CPU embedding, batch size 2, one worker, BFS, Top-1 sentence per entity, three configured iterations, and threshold 0.05. The threshold meaning and comparison remain official; 0.05 is needed because the second-hop MiniLM score is 0.06655, which the initially tried 0.1 threshold correctly pruned.

## 6. Tiny input and NER

Original requested input is preserved in `examples/smoke/original_input.json`. With numbered passages, en-core-web-sm 3.6.0 did not recognize `Beatrice I` in passage 1, although the question yielded `Beatrice`; using it would force a wrong query-to-graph match. The full failed observation is retained in `artifacts/smoke_result.json` under `original_ner_failure_evidence`.

Adjusted input:

```text
P1: Beatrice, Duchess of Burgundy, was married to Frederick Barbarossa.
P2: Frederick Barbarossa was King of Germany.
P3: Paris is the capital of France.
Q:  What nationality was the husband of Beatrice, Duchess of Burgundy?
```

Actual adjusted NER after official `index:` prefixes:

- P1 sentence: `0:Beatrice, Duchess of Burgundy, was married to Frederick Barbarossa.` -> Beatrice (ORG), Frederick Barbarossa (PERSON).
- P2 sentence: `1:Frederick Barbarossa was King of Germany.` -> Frederick Barbarossa (PERSON), Germany (GPE).
- P3 sentence: `2:Paris is the capital of France.` -> Paris (GPE), France (GPE).
- Query -> Beatrice (ORG).

The ORG label is imperfect, but official code uses entity text and only filters CARDINAL/ORDINAL; no label correction was added.

## 7. Graph, matching, propagation and ranking

Graph/index statistics:

- Passage nodes: 3.
- Sentence embedding-store nodes: 3.
- Entity nodes: 5.
- Actual igraph vertices: 8 (3 passage + 5 entity).
- Actual igraph edges: 8 (6 passage-entity occurrence edges + 2 adjacent-passage edges).
- Entity-sentence links used by BFS: 6.

Query matching:

- Query entity `beatrice` -> graph entity `Beatrice`, cosine similarity 1.00000.

Official BFS propagation captured by the asserted observer:

- Iteration 1 processes Beatrice at 1.00000; selects P1's sentence at query similarity 0.75503; activates Frederick Barbarossa and Beatrice at 0.75503.
- Iteration 2 processes Frederick Barbarossa; selects P2's sentence at 0.08814; activates Germany and Frederick Barbarossa at 0.06655. Beatrice has no unused sentence.
- Observer equality assertion: passed.

Official initial passage node weights after the configured `passage_node_weight=0.05`:

- P1: 0.1122276154.
- P2: 0.0015143826.
- P3: 0.0066872624.

PPR / final Top-k:

1. P1, 0.2661878878.
2. P2, 0.1333502280.
3. P3, 0.0267827737.

Human expectation: passed. P1 and P2 precede P3; Beatrice is the seed; Frederick Barbarossa and Germany appear in the real computed propagation trace.

The saved final run rebuilt the smoke index from an empty `import/smoke` cache and took about 1.21 seconds. Process RSS grew from about 331.3 MiB to 520.3 MiB, with a peak working set around 545.2 MiB. `nvidia-smi` showed 45 MiB used both before and after; no retrieval/model workload was placed on the GPU. No MemoryError, OS error 1455, CUDA OOM, or process termination occurred.

## 8. Acceptance matrix

| Area | Result | Evidence |
|---|---|---|
| A. Isolation | Pass | All formal files/caches/results are under the working directory; the sealed old project was not accessed. |
| B. Source | Pass | URL, branch, commit, import method and hashes in `SOURCE.md`; no nested Git repository. |
| C. Environment | Pass with documented adaptation | Hardware/software measured; independent `.venv`; original Python 3.10 failure retained; Python 3.9 recovery works; no OpenAI needed. |
| D. Program | Pass | Official index and retrieve methods run; tiny index/retrieval complete without unacceptable memory errors or hard-coded answers. |
| E. Behavior | Pass on adjusted wording | Matching, two-hop propagation and P1/P2/P3 ranking are in the structured trace. Original wording failure is retained. |
| F. Reproducibility | Pass | Inputs, output, environment freeze, setup failures, source hashes, model download script and one-command smoke entry are saved. |

## 9. Remaining limitations

- The smoke uses en-core-web-sm and all-MiniLM-L6-v2, not the paper/official default en-core-web-trf and all-mpnet-base-v2, because this stage prioritizes a small local run and the larger model download failed on the current connection.
- The original `Beatrice I` wording fails passage NER with the small spaCy model; the adjusted wording is required and documented.
- Official `run.py` still hard-codes GPU selection and combines retrieval with OpenAI generation/evaluation. It is preserved unchanged; use `smoke_test.py` for this stage.
- This test has three passages and one question. It cannot establish benchmark accuracy, scaling behavior, GPU vectorized behavior, answer quality, or equality with paper tables.

## 10. Exactly what can and cannot be concluded

Can conclude: the official LinearRAG core indexing and retrieval flow has completed a local smoke verification on this Windows machine using a tiny artificial example, with observable NER, embeddings, graph construction, query matching, entity propagation, passage initialization, PPR and Top-k output.

Cannot conclude: that the paper has been fully reproduced, that official benchmark scores are matched, that the default large models work on this machine, or that generation/GPT evaluation/scalability claims have been validated.

One bounded next-stage suggestion: rerun this same single sample with the official `en_core_web_trf` and `all-mpnet-base-v2` once network availability permits, changing no corpus, algorithm, or evaluation scope, and compare only the NER/matching/propagation trace.
