# Run status

Last rendered: 2026-05-09T21:28:52Z

Open this file at the start of a session before reconstructing runs from logs.

## Active Runs

- **mac2-granite41-8b-serial-20260509**: status: `running`; runner: `benchmark.py`; host: `mac2`; task scope: `tasks`; model scope: `ibm/granite4.1:8b`; output: `/Users/hannohilbig/sidecar-runs/phi4-mini-2026-05-09/output/sidecar/granite41_8b_predictions.csv`; log: `/Users/hannohilbig/sidecar-runs/phi4-mini-2026-05-09/logs/granite41_8b_serial_20260509.log`; note: Phi dependency finished; queue wrapper is pulling ibm/granite4.1:8b (5.3 GB) before starting the full 18-task benchmark.

## Failed Or Needs Attention

- **mac2-next-wave-serial-20260506**: status: `needs_attention`; runner: `benchmark.py`; host: `mac2`; task scope: `tasks_next`; model scope: `7 observed models`; output: `/Users/hannohilbig/sidecar-runs/next-wave-2026-05-06/output/sidecar/next_wave_openweight_predictions.csv; /Users/hannohilbig/sidecar-runs/next-wave-2026-05-06/output/sidecar/next_wave_cheap_api_predictions.csv`; note: Reachability checked 2026-05-09: SSH OK; no benchmark or ollama processes found by pgrep; tmux command unavailable; sidecar prediction and summary files are present on mac2; still needs local archive/import and missing gpt-5.5 plus claude-sonnet-4-6 cells.

## Recent Completed Runs

- **mac2-phi4-mini-serial-20260509-r2**: status: `completed`; runner: `benchmark.py`; host: `mac2`; task scope: `tasks`; model scope: `phi4-mini`; output: `/Users/hannohilbig/sidecar-runs/phi4-mini-2026-05-09/output/sidecar/phi4_mini_predictions.csv`; log: `/Users/hannohilbig/sidecar-runs/phi4-mini-2026-05-09/logs/phi4_mini_serial_20260509_r2.log`; task: `brandt_political_relevance`; rows: `8820`; note: Completed on mac2: 18/18 tasks, 8820 rows, 29 parse errors. Decision: archive as speed-baseline artifact, do not promote to main models lineup; only ~1.35x faster than Qwen3 30B-A3B with materially weaker accuracy.
- **mac2-phi4-mini-serial-20260509**: status: `cancelled`; runner: `benchmark.py`; host: `mac2`; task scope: `tasks`; model scope: `phi4-mini`; output: `/Users/hannohilbig/sidecar-runs/phi4-mini-2026-05-09/output/sidecar/phi4_mini_predictions.csv`; log: `/Users/hannohilbig/sidecar-runs/phi4-mini-2026-05-09/logs/phi4_mini_serial_20260509.log`; note: Cancelled after early parse audit; Phi emitted bare scalar binary labels and parser was widened before restart.
- **droplet-batched-deepseek-20260508**: status: `completed`; runner: `batch_benchmark.py`; host: `droplet`; task scope: `tasks`; model scope: `deepseek-v4-pro`; batch sizes: `10,20`; tmux: `pob-deepseek`; output: `/root/polsci-open-bench/output/predictions_batched_deepseek_run.csv`; log: `/root/polsci-open-bench/logs/batched_deepseek_20260508.log`; cost cap: `15`; note: Completed on droplet; needs merge into canonical batched file.
- **droplet-batched-gpt54nano-20260508**: status: `completed`; runner: `batch_benchmark.py`; host: `droplet`; task scope: `tasks`; model scope: `gpt-5.4-nano`; batch sizes: `10,20`; tmux: `pob-gpt54nano`; output: `/root/polsci-open-bench/output/predictions_batched_gpt54nano_run.csv`; log: `/root/polsci-open-bench/logs/batched_gpt54nano_20260508.log`; cost cap: `15`; note: Completed on droplet; needs merge into canonical batched file.

## Ledger

- Source: `output/run_registry.jsonl`
- Events: 20
- Runs: 6

## Operating Rule

- Register every sidecar, API, local, droplet, or mac2 run before launch.
- Record the host, tmux session, log path, output path, model scope, task scope, and cost cap when applicable.
- Render this file before ending a session.
