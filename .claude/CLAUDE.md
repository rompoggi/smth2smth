# smth2smth — agent instructions

Read and follow **all** rules under `.claude/rules/`, especially:

- **`distributed-run-log.md`** — per-host run log; link **`experiments/<doc>.md`** (read-only); edit `track_a.tex` / `track_b.tex` **only if explicitly asked**.
- **`start-resume-runs.md`** — nohup + `uv`, log paths (`logs/track_a|b/…`), health check, W&B resume, pid cleanup.

Project layout, tracks, Hydra, and `nohup` jobs: see `.cursor/rules/project-context.mdc` (Cursor) — same repo conventions apply here.
