# Experiment run logs (distributed)

To avoid merge conflicts on shared LaTeX logbooks (`track_a.tex`, `track_b.tex`), **each machine and user** keeps its own markdown log:

```
report/{username}/{hostname}.md
```

| Component | How to obtain | Example |
|-----------|---------------|---------|
| `username` | `whoami` | `romain.poggi`, `thomas.turkieh` |
| `hostname` | `hostname`, strip at first `.` | `rouget.polytechnique.fr` → `rouget` |

Examples:

- `report/romain.poggi/rouget.md`
- `report/thomas.turkieh/sole.md`

Two teammates on the same host still get **separate files** (different `whoami`).

## Rules for humans and agents

1. **Only edit your own file** for run status / metrics / submissions.
2. **Do not** edit `track_a.tex` or `track_b.tex` unless someone **explicitly asks** (LaTeX is for manual consolidation, not live multi-host logging).
3. Each log entry is a new section (separator `---`) with title, track (`A` / `B` / `A+B`), timestamp, and a **short results summary**.
4. Reference the run spec with a **markdown link into `experiments/`**, e.g. [`experiment_16052026`](../experiments/experiment_16052026.md) — **no line numbers**; treat **`experiments/**` as read-only** when logging. Hydra preset name (`experiment=...`) is plain text, not a link to `configs/experiment/`.

Agent enforcement:

- **Outcomes / status:** `.cursor/rules/distributed-run-log.mdc`, `.claude/rules/distributed-run-log.md`
- **Launch / resume / logs:** `.cursor/rules/start-resume-runs.mdc`, `.claude/rules/start-resume-runs.md`

Training logs live under `logs/track_a/`, `logs/track_b/`, or study dirs (e.g. `logs/hc/`). Names look like `mae150-ft-f4_20260525.log`, not `e1_host.log`.

## Consolidation

Periodically, one person merges highlights from `report/*/*.md` into the LaTeX report on a quiet branch — not while jobs are running on many hosts.
