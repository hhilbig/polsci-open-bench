# Project status

Last updated: 2026-05-02

## Current snapshot

- Main public artifact: [`output/report_pdf.pdf`](../output/report_pdf.pdf)
- Serial benchmark artifact: [`output/predictions.csv`](../output/predictions.csv)
- Batched benchmark artifact: [`output/predictions_batched.csv`](../output/predictions_batched.csv)
- Serial summary: [`output/summary.csv`](../output/summary.csv)
- Batched summary: [`output/summary_batched.csv`](../output/summary_batched.csv)
- Task-length audit: [`output/task_length_audit.csv`](../output/task_length_audit.csv)
- Task-length note: [`output/task_length_analysis.md`](../output/task_length_analysis.md)
- DeepSeek sidecar archive: [`output/archive/deepseek_sidecar_2026-05-02/`](../output/archive/deepseek_sidecar_2026-05-02)
- Open-weight new-task sidecar archive: [`output/archive/openweight_new_tasks_2026-05-02/`](../output/archive/openweight_new_tasks_2026-05-02)
- Post-release change log: [`docs/post_release_changes.md`](post_release_changes.md)
- Release workflow note: [`docs/release_workflow.md`](release_workflow.md)

## Benchmark scope

- 10 classification tasks
- 7 serial models
- 3 batched sizes in the checked-in batched benchmark output
- 500 sampled items per task

## Repository maintenance

- Python dependencies are pinned in [`requirements.txt`](../requirements.txt).
- Report dependencies and rerun instructions are documented in [`docs/reproduce.md`](reproduce.md).
- Output schema and prompt provenance are tracked in [`docs/schema.md`](schema.md) and [`docs/prompts_provenance.md`](prompts_provenance.md).
- Loader/item-id integrity checks live in [`tests/test_benchmark_integrity.py`](../tests/test_benchmark_integrity.py).
- Built-in tasks are defined in YAML manifests under [`tasks/`](../tasks), and custom-task instructions are in [`docs/custom_tasks.md`](custom_tasks.md).
- Built-in models are defined in YAML manifests under [`models/`](../models), and custom-model instructions are in [`docs/custom_models.md`](custom_models.md). The live built-in model set now includes `deepseek-v4-pro`, plus a prepared `gemma4:26b` manifest on the `v2` branch for the next local-model expansion pass.
- DeepSeek non-thinking and thinking serial runs are archived as sidecar outputs, not merged into the canonical release artifacts.
- The four post-release tasks have now also been run for the four local open-weight models, but those outputs are likewise archived as sidecar results rather than merged into the canonical release artifacts.

## Latest repair

- Sampled `item_id` values are now unique within every task.
- The checked-in serial and batched raw prediction files were repaired to use the
  same deterministic duplicate-id suffixing as the live loaders.
- Stale `state_adaptation` columns were removed from `output/predictions_batched.csv`.
- Rebuilt summaries left headline F1 values unchanged. The only metric updates
  were small paired-bootstrap CI changes, concentrated in `gilardi_relevance`.

## New artifacts

- Manifest-driven task loading now replaces the hard-coded task registry for the benchmark and summary scripts.
- Manifest-driven model loading now replaces the hard-coded model registry for the benchmark and serial summary scripts.
- The task-length audit now writes [`output/task_length_audit.csv`](../output/task_length_audit.csv) and [`output/task_length_analysis.md`](../output/task_length_analysis.md).

## Deferred modularization

- The repo now supports custom tasks and custom models within the built-in `ollama`, `openai`, and `anthropic` backends.
- A later refactor should move inference into backend adapters plus a backend registry, then add a `subprocess` backend so unsupported providers can be wrapped without editing core benchmark code.
- The staged plan for that work is tracked in [`TODO.md`](../TODO.md).

## Next checks

- Keep task docs synchronized with the live manifests in [`tasks/`](../tasks).
- Keep `item_id` values unique within each task's sampled 500-item set so selective reruns and merges remain one-to-one.
- Keep [`docs/post_release_changes.md`](post_release_changes.md) updated whenever a change affects benchmark logic, live task/model scope, scored artifacts, or public-facing documentation.
- Keep the branch / sidecar / canonical distinction consistent with [`docs/release_workflow.md`](release_workflow.md).
- Track substantive benchmark extensions in [`TODO.md`](../TODO.md).
