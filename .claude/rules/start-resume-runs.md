# Starting and resuming runs (mandatory)

Same rules as `.cursor/rules/start-resume-runs.mdc`. Pair with **`distributed-run-log.md`**.

## Launch

- **`nohup`** + **`uv run`** + `PYTHONPATH=src` + `PYTHONUNBUFFERED=1`.
- Semantic **`RUN_NAME`** (e.g. `mae150-ft-f4`), not `e1_hostname`.
- **`scripts/launch_*.sh`:** read before use — defaults may still point to flat `logs/`. **Verify** the resolved log path matches `logs/track_a|b|…/`; override `LOG=` on the CLI if not. **Do not edit scripts** unless the user asks.
- After start, confirm output goes to the intended file (`tail` that path).

## Log path

`logs/{dir}/{RUN_NAME}_{YYYYMMDD}.log`

| `dir` | When |
|-------|------|
| `track_a` | `track=a` / Track A experiments |
| `track_b` | `track=b` / Track B experiments |
| `hc`, `ensemble`, … | Dedicated study (e.g. HC ablation → `logs/hc/`) |

Ask the user if track/dir is unclear.

**Restart / VM resume:** append to the **same** log file and reuse the same `RUN_NAME`.

## Log header (first lines, ASCII)

```text
# run: mae150-ft-f4
# started: 2026-05-25T01:50:00+02:00
# track: a
# experiment_doc: experiments/new_ideas_tracka.md
# hydra: experiment=track_a_videomae_official_ssv2_ft
```

`experiment_doc` → **`experiments/*.md`** (read-only). No line-number anchors.

## PID files

- Use **only** when ongoing monitoring needs `kill -0 $(cat pid)`.
- Otherwise log `trainer_pid=…` inside the `.log` and skip `.pid`.
- **`rm` the `.pid` file when the run ends.**

## W&B

- **Train/pretrain:** online; copy URL from `[wandb] run started: …` in the log.
- **Submit / short jobs:** disabled.
- **Resume:** same config + checkpoint; `WANDB_RESUME=allow` + `WANDB_RUN_ID=<previous id>`; append same log.

## Health check (~30–90s after start)

Confirm step lines appear at `log_interval_steps`, one step per line, ASCII-only on training lines, no traceback.

If broken: stop, report to user, offer restart into the **same** log path after fix.

## Report to user + `report/{user}/{host}.md`

After a healthy start, state: log path (link), healthy yes/no, W&B link or n/a, `tail -f` hint.

Update distributed run log with links to **log**, **W&B** (if any), **pid** (only if RUNNING + pid file exists).

## Failed boot

If the log only has config + Python error, you may delete/truncate and relaunch to the **same path** after user confirmation.
