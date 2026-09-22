# Upstream Import and Removal Archive

This file is operational context for future agents investigating source provenance, missing files, or behavior differences across repository history. It is not user-facing documentation, and it does not claim that the current working tree remains byte-for-byte identical to upstream. Treat environment details and runtime results as historical snapshots until re-verified.

## 1. Upstream source

| Item | Recorded value |
|---|---|
| Official repository | `https://github.com/DEEP-PolyU/LinearRAG` |
| Official branch | `main` |
| Imported upstream commit | `bcc94e66c221f798801255efba09311d6fbcd8d6` |
| Upstream commit date | `2026-07-05T00:55:44Z` |
| Local acquisition date | `2026-09-13`, Asia/Shanghai |
| First local import commit | `a5a73cb0df38cc9d82d003f5159403152cded21c` |
| Chinese documentation/comment archive commit | `19493cda19e9ac43c6031b1ac1bc074eb893d5cd` |

The first two Git HTTPS attempts failed with:

```text
Recv failure: Connection was reset
```

The exact commit was then downloaded from GitHub's official codeload endpoint:

```text
https://codeload.github.com/DEEP-PolyU/LinearRAG/zip/bcc94e66c221f798801255efba09311d6fbcd8d6
```

The archive was extracted into a temporary directory inside this project. Fifteen upstream files were compared byte-for-byte with SHA-256 before being copied into the project root. The temporary ZIP and extraction directory were then removed. No nested `.git` directory was retained.

Important limitation: `bcc94e66...` is not currently available as a local Git object. Do not assume `git show bcc94e66...` will work. The complete initial provenance record and original hash manifest remain available in local history:

```powershell
git show a5a73cb:SOURCE.md
git show 19493cd:SOURCE.md
```

## 2. Later version boundaries

| Commit | Meaning |
|---|---|
| `a5a73cb` | First local import, environment evidence, and minimal CPU/BFS smoke closure |
| `19493cd` | Chinese README, project documentation, and code comments; best recovery point for material removed later |
| `75c6c4e` | Source readability refactor; inspect the diff before assuming behavior was unchanged |
| `f3437f7` | Reproduction-environment preparation closure; removed bulky one-off evidence, generated results, and upstream presentation assets |
| `4b32e0f` | Separated code, runtime data, and project material; project-owned paths were routed through `src.paths` |

Use `project/agent/BASELINE.md` for the current runtime and debugging baseline. This archive only answers where the source came from, which materials were later removed, and how to trace them safely.

## 3. Material removed by `f3437f7`

The following tracked material was removed from the active tree by `f3437f7`, but remains recoverable from `19493cd`:

- Provenance and acceptance documentation: `SOURCE.md`, `REPRODUCTION_REPORT.md`, and the old `readme.md`.
- One-off environment evidence: `artifacts/environment-freeze.txt`, `artifacts/system-environment.json`, and `artifacts/setup/*`.
- Historical runtime evidence: `artifacts/smoke_console.txt`, `artifacts/smoke_result.json`, and `artifacts/smoke_result_cold_start.json`.
- Upstream presentation images: `figure/efficiency_result.png`, `figure/generation_results.png`, and `figure/main_figure.png`.
- Old environment helper file: `requirements-windows.txt`.
- Old scripts: `scripts/run.sh`, `tools/download_embedding_model.ps1`, and `tools/probe_ner.py`.

The paper PDF was not permanently deleted. It moved from the repository root to `paper/`, then into the reorganized `project/` area, and now resides at `project/agent/paper/2510.10114v4.pdf` as Agent-facing source material.

These removals do not erase source provenance. Git history still contains the old files. For present-day execution, follow the later commits and `BASELINE.md`, not obsolete setup artifacts.

## 4. Safe inspection and recovery

Read historical content without modifying the working tree:

```powershell
git show 19493cd:SOURCE.md
git show 19493cd:REPRODUCTION_REPORT.md
git show 19493cd:artifacts/smoke_result_cold_start.json
git show --stat f3437f7
git diff 19493cd..f3437f7 -- src
```

To recover one old file, first confirm that the destination has no user-owned change, then name both the source commit and exact path. For example:

```powershell
git restore --source=19493cd -- SOURCE.md
```

Do not restore an entire old commit wholesale. Do not use `git reset --hard` or `git clean -fd`. The directory layout changed after `4b32e0f`; broad restoration can mix obsolete and current paths.

## 5. Remote safety

`origin` currently points to the upstream authors' repository:

```text
https://github.com/DEEP-PolyU/LinearRAG.git
```

Treat it as a read-only upstream source. Never push to it. Before any future upload, configure a user-owned remote and verify the exact destination with the user.

## 6. How to use this record during debugging

1. When a file appears to have been deleted accidentally, inspect `f3437f7` and `4b32e0f` first. Distinguish deletion from relocation and ignored generated output.
2. When behavior appears to diverge from upstream, compare the exact upstream commit, `a5a73cb`, and the current code. Do not mistake translated comments or path migration for an algorithm change.
3. For environment or model failures, read `BASELINE.md` first and verify live state. Old `artifacts/` are historical evidence only.
4. Use `python -m src.smoke_test` for minimal indexing/retrieval verification. It is not evidence that the full `src.run` experiment or the paper has been reproduced.
