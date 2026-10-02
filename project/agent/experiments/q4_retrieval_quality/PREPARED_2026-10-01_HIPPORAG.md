# Q4 small Medical corpus preparation — 2026-10-01

Supersedes the earlier full-corpus preparation. No paid API calls or live Medical HippoRAG run occurred.

- Code changes are limited to `experiments/corpus_construction/build_corpus.py` and `experiments/q4_retrieval_quality/retrieval_quality.py`.
- The builder preserves the Q1/Q2 HotpotQA path and adds a Medical-specific path. Q7/Q8 are outside its scope.
- Fixed Medical sample: seed `20261001`, three questions per task, 12 total. TF-IDF question/reference-relation Top-2 candidates are combined with offline-reviewed additions (original source indices 173, 89, 133, 14), then five random extra passages are added and the corpus is shuffled. Full original passage text and reference statements are retained.
- Input package: `data/input/derived_corpora/medical-type3-noise5-s20261001/`. It has 30 passages, 157351 characters, 25901 whitespace words. Review input: `data/input/derived_corpora/medical-type3-review-s20261001.json`. Source mappings and partial review notes live in the package manifest; they are not complete gold support annotations. Some references, including hypothetical patient details, remain unverified.
- Q4 accepts `--corpus-dir`, uses its questions without resampling, checks file hashes, and separates corpus/model-specific caches. Retrievers receive only ID/question/answer; reference evidence is added afterward for judging. Default BFS, Top-5 and judge prompts remain unchanged.
- Deleted cross-run result copying (`--reuse-from`, `reuse_saved_results`) and previously deleted pairwise differences remain absent. Same-run resume and parallel `summary.csv` are retained.
- Prepared run: `data/output/experiment_results/q4_retrieval_quality/q4-medical-small-ready-20261001/`. Base fresh calls: 60 passage extraction, 12 query NER, 36 judge calls per method (108 total). Retries and evidence fallback are excluded; billing tokens and currency cost remain unmeasured.
- Minimal offline verification: deterministic Medical reconstruction, original passage retention, unchanged Q1 HotpotQA input package, mocked three-method routing with identical inputs, BFS and parallel CSV. Evidence: `data/output/experiment_results/q4_retrieval_quality/q4-medical-small-offline-check-20261001/OFFLINE_VALIDATION.json`. That run contains fixtures, not benchmark evidence.
- Earlier directories `q4-hipporag-precheck-20261001` and `q4-hipporag-200q-20261001` describe the superseded full-corpus/reuse design and must not be resumed with the new script.

Prepared-run command (paid if `--prepare-only` is omitted; not executed live):

```powershell
.venv\Scripts\python.exe experiments/q4_retrieval_quality/retrieval_quality.py --corpus-dir data/input/derived_corpora/medical-type3-noise5-s20261001 --methods vanilla_rag linearrag hipporag --experiment-id q4-medical-small-ready-20261001 --resume --prepare-only
```
