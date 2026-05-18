# Release Workflow

Last updated: 2026-05-18

The current public release is the 34-task benchmark described in the PDF report.
The report is the authoritative narrative artifact; the checked-in CSVs,
figures, tables, prompts, and manifests support that report.

## Public Release Files

The public benchmark files live directly under `output/`:

- [`output/predictions.csv`](../output/predictions.csv)
- [`output/summary.csv`](../output/summary.csv)
- [`output/predictions_batched.csv`](../output/predictions_batched.csv)
- [`output/summary_batched.csv`](../output/summary_batched.csv)
- [`output/summary_batched_local_b10.csv`](../output/summary_batched_local_b10.csv)
- [`output/report_pdf.pdf`](../output/report_pdf.pdf)

The task and model definitions live in:

- [`tasks/`](../tasks)
- [`models/`](../models)

The generated task inventory is [`docs/task_inventory.md`](task_inventory.md).
Regenerate it whenever task manifests change:

```bash
python3 code/task_inventory.py --write
python3 code/task_inventory.py --check
```

## Development Runs

Intermediate or exploratory outputs should not overwrite the public release
files until the comparison set is complete enough to interpret. Keep such files
in ignored working directories such as `output/sidecar/`, `output/openai_batch/`,
or `output/anthropic_batch/`, then promote only after the intended task-model
grid is complete and checked.

Before launching a long local, remote, or API run, register it in the run ledger.
The append-only ledger lives at `output/run_registry.jsonl`, and the readable
snapshot is `docs/run_status.md`. Both files are ignored because they can
contain local paths, process ids, API batch ids, and cost notes.

Typical registration:

```bash
python3 code/run_registry.py start \
  --run-id local-example-20260518 \
  --runner benchmark.py \
  --host local \
  --task-scope tasks \
  --model-scope example-model \
  --output output/sidecar/example_predictions.csv \
  --log logs/example_20260518.log

python3 code/run_registry.py render-status
```

Use `finish` for terminal states and `update --run-status tabled` for completed
runs that should not enter the public benchmark, for example because invalid
output rates are too high.

## Promotion Rule

Promote a run into the public release files only when all of the following hold:

- the intended task-model grid is complete or the missing cells are explicitly
  out of scope;
- row counts match the task inventory;
- summaries rebuild cleanly;
- report figures and tables rebuild cleanly;
- the PDF renders; and
- public docs describe the new task and model counts.

## Release Checklist

Before posting or pushing a public release:

```bash
python3 code/task_inventory.py --check
python3 code/build_coverage_matrix.py
python3 -m unittest discover -s tests
Rscript code/build_report_assets.R
quarto render output/report_pdf.qmd --to pdf
```

Then check:

- [`README.md`](../README.md)
- [`docs/status.md`](status.md)
- [`docs/schema.md`](schema.md)
- [`docs/reproduce.md`](reproduce.md)
- [`docs/prompts_provenance.md`](prompts_provenance.md)
- [`docs/twitter_thread.md`](twitter_thread.md)
- [`docs/post_release_changes.md`](post_release_changes.md)

The public docs should report the same task count, model count, main F1
terminology, and batching terminology as the PDF report.
