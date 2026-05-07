# Release workflow

This repo uses three distinct states for benchmark work:

1. `canonical`
   - The official public benchmark artifacts.
   - In practice, this means the checked-in outputs on `main`, especially:
     - [`output/predictions.csv`](../output/predictions.csv)
     - [`output/summary.csv`](../output/summary.csv)
     - [`output/predictions_batched.csv`](../output/predictions_batched.csv)
     - [`output/summary_batched.csv`](../output/summary_batched.csv)
     - [`output/report_pdf.pdf`](../output/report_pdf.pdf)

2. `live-dev`
   - New code, tasks, models, and docs under active development.
   - Current convention: do this work on a long-lived development branch such as
     `v2-benchmark-expansion`.

3. `sidecar`
   - Real benchmark outputs that are informative and preserved, but not yet
     canonical.
   - These live in dated archive folders under [`output/archive/`](../output/archive).

## Why this split exists

- Task and model expansion is incremental.
- Full benchmark reruns are expensive and time-consuming.
- Partial reruns are often useful before the whole comparison grid is ready.
- The public repo can still show incremental development without pretending
  that every intermediate run is the new benchmark.

## Branch convention

- `main`
  - last stable public benchmark release
- `v2-benchmark-expansion`
  - current development branch for the next benchmark iteration

Use the development branch for:

- new task ingestion
- new model manifests
- infrastructure refactors
- sidecar runs
- release-candidate rebuilds

## Sidecar convention

Use a sidecar when a run is any of the following:

- partial across tasks
- partial across models
- exploratory or provisional
- a new condition not yet intended for the core benchmark
- a completed subset whose comparison set is still incomplete

Store sidecars in a dated folder, for example:

- [`output/archive/deepseek_sidecar_2026-05-02/`](../output/archive/deepseek_sidecar_2026-05-02)
- [`output/archive/openweight_new_tasks_2026-05-02/`](../output/archive/openweight_new_tasks_2026-05-02)

Each sidecar folder should contain:

- raw predictions
- summary outputs
- a short `README.md` describing scope, model/task coverage, and merge status

## Coverage matrix

The model x task coverage matrix lives at [`docs/coverage_matrix.md`](coverage_matrix.md)
and [`output/coverage_matrix.csv`](../output/coverage_matrix.csv). It is rebuilt by
`python3 code/build_coverage_matrix.py`, which scans `output/predictions.csv`, every
`*predictions*.csv` under `output/sidecar/`, and every `*predictions*.csv` under
`output/archive/`. `code/benchmark.py` calls the same refresh at the end of every
run, so the matrix stays in sync with active sidecar files as well as archived
sidecars. After moving a sidecar into `output/archive/<sidecar>/` by hand, rerun
the script so the matrix records the archived source label.

## Promotion rule

Do not merge sidecar outputs into canonical outputs until the comparison set is
complete enough to interpret cleanly.

Examples:

- four new tasks run only for local open-weight models: keep as sidecar
- four new tasks run for the intended local and API comparison set: candidate
  for canonical merge
- DeepSeek thinking-mode experiment: keep as sidecar unless it becomes an
  explicitly benchmarked condition

## Documentation rule

Whenever a change affects benchmark logic, live task/model scope, scored
artifacts, or public-facing docs:

1. update the relevant code or docs
2. verify the change
3. update [`docs/post_release_changes.md`](post_release_changes.md)

Use [`docs/status.md`](status.md) as the short operational snapshot and
[`docs/post_release_changes.md`](post_release_changes.md) as the running
post-release changelog.

## Release cycle

Typical pattern:

1. add tasks, models, or infrastructure on `v2-benchmark-expansion`
2. run smoke tests
3. run selective reruns as sidecars
4. keep the changelog current
5. once the intended matrix is complete, rerun the canonical outputs
6. rebuild summaries and report
7. merge or publish as the next benchmark version

## Naming guidance

- `canonical outputs`
  - official benchmark files on `main`
- `live-dev`
  - current branch state under development
- `sidecar`
  - preserved but non-canonical result bundle
- `prepared`
  - code or manifest exists, but the model/task has not been benchmarked yet
