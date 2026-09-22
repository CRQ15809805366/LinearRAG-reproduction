# Decision records

Use this directory for consequential research, evaluation, and repository decisions whose rationale must survive across sessions.

Each record should state the context, decision, alternatives considered, consequences, and the evidence available at the time. Do not create records for routine edits or self-evident implementation details.

## Current structural decision

On 2026-09-17, the repository separated three forms of research material:

- bulk machine-generated evidence under `data/output/`;
- curated internal experiment and operational records under `project/agent/`;
- externally readable research material under `research_report/`.

## Environment decision

On 2026-09-21, the OpenAI-compatible `OPENAI_API_KEY` and `OPENAI_BASE_URL`
values moved out of Windows user-level environment variables. Since
2026-09-22, `src.utils` reads the untracked project-root `.env.local` directly
and passes the values to the OpenAI client without injecting them into the
process environment.
