# OpenAI-compatible credential migration (2026-09-21)

## Context

Before 2026-09-21, `OPENAI_API_KEY` and `OPENAI_BASE_URL` were stored as
Windows user-level environment variables. The CC Switch utility reported them as
two system environment variables that could override its own provider
configuration, which created a risk of unintended global overrides for other
tools such as Codex.

The values were also displayed in a CC Switch screenshot, so the API key must be
treated as exposed and should be rotated in the provider console.

## Decision

Move both values into the untracked project-root file `.env.local` and load them
from project code when they are absent from the process environment. Remove the
Windows user-level copies once project loading is verified.

## Implementation

- Added `src/local_env.py` with `load_project_env()`, which reads only
  `OPENAI_API_KEY` and `OPENAI_BASE_URL` from `.env.local` and uses
  `os.environ.setdefault`, so an already-set process value still wins.
- `src/utils.py` calls `load_project_env()` before constructing `LLM_Model`.
- `.env.local` was added to `.gitignore` and holds both values. Its contents were
  never printed or written into any tracked record.
- `src.local_env --migrate-windows-user-env` performed the initial copy from the
  Windows user environment (`HKEY_CURRENT_USER\Environment`).

## Verification

- `git check-ignore -v .env.local` confirmed the file is ignored.
- With `OPENAI_API_KEY` and `OPENAI_BASE_URL` cleared from the child process
  environment, `src.utils` still loaded both values from `.env.local`.
- `user_vars_before=2` and `user_vars_after=0` confirmed the user-level removal.
- `python -m src.smoke_test` passed after the migration (`status=passed`,
  `bfs_replay_matches_official=true`). The smoke path does not call the LLM, so
  this verifies loading and the retrieval path rather than live endpoint access.

## Consequences and limits

- CC Switch no longer sees the two conflicting user-level variables.
- Anyone who deletes `.env.local`, moves the checkout, or runs the code from a
  different clone must recreate the file; the loader fails quietly, so the
  missing-variable error appears later at client construction.
- The repository does not include a live endpoint call in its automated checks.
Endpoint validity remains unverified until a bounded `src.run` or experiment
call is executed.

## Superseding implementation (2026-09-22)

`src.local_env` was removed. `src.utils` now reads `.env.local` directly and
passes both values explicitly to `LLM_Model`; no values are written to
`os.environ`.
