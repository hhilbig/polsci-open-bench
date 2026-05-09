# Project Agent Instructions

## Startup Checklist

- At the start of any new session in this repository, read `docs/run_status.md`
  before reconstructing state from memory or shell history.
- If `docs/run_status.md` lists active or queued remote jobs, verify live state
  before reporting status:
  - first check connectivity to `mac2`; this host has sometimes dropped
    connections even when sleep is disabled, SSH keys are configured, and
    Tailscale is set up
  - use a short non-interactive SSH check such as
    `ssh -o BatchMode=yes -o ConnectTimeout=8 mac2 hostname`
  - if SSH fails, check Tailscale reachability before assuming the job failed
  - `ssh mac2 'pgrep -fl "benchmark.py|batch_benchmark.py|queue|ollama pull"'`
  - inspect the relevant log path listed in `docs/run_status.md`
  - inspect the relevant output CSV/ledger when needed
- Treat `output/run_registry.jsonl` as the append-only source of run history.
  Use `python3 code/run_registry.py render-status` after adding or updating run
  events.
- Register long-running local, `mac2`, droplet, sidecar, or API jobs before
  launch. Include host, runner, model scope, task scope, batch sizes if relevant,
  output path, log path, and dependency/queue notes.

## Current Remote Workflow

- `mac2` is the default host for long-running local Ollama work.
- Do not assume a queued mac2 job is running. Check the queue wrapper process and
  the target benchmark process separately.
- Do not assume a missing or stale process report means a job failed until mac2
  connectivity itself has been checked.
- The current mac2 sidecar root used for recent runs is:
  `/Users/hannohilbig/sidecar-runs/phi4-mini-2026-05-09`

## Output Discipline

- Keep canonical benchmark outputs separate from sidecar outputs unless an
  explicit merge/import step is requested.
- For sidecar runs, prefer `output/sidecar/...` on the remote host and document
  the path in `docs/run_status.md`.
- Before promoting any sidecar model into `models/`, compare both quality and
  speed against the current local baselines.
