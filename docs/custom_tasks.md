# Custom tasks

The benchmark can now load tasks from YAML manifests instead of requiring edits
to `code/benchmark.py`.

## Minimal interface

Use either:

- `--task-manifest path/to/task.yaml`
- `--task-dir path/to/task_directory`
- `--tasks-dir path/to/directory_of_manifests`

If you use `--task-dir`, the directory should contain at least:

- `task.yaml`
- `data.csv` unless `task.yaml` points elsewhere with `data_file`
- `prompt.txt` unless `task.yaml` points elsewhere with `prompt_file`

## Manifest schema

Required fields:

- `label_kind`: `binary`, `categorical`, or `multi_binary`
- `text.template`: Python-style format string using CSV column names such as `{text}` or `{case}`
- `ground_truth`: label column definition

Common optional fields:

- `name`: task name; defaults to the manifest stem
- `family`: task family label
- `source`: human-readable provenance label
- `data_file`: relative or absolute path to the task CSV
- `prompt_file`: relative or absolute path to the prompt text file
- `labels`: required for `categorical` and `multi_binary`
- `label_key`: required for `binary` and `categorical`
- `id.kind`: `column` or `generated`
- `id.column`: source id column when `id.kind: column`
- `id.prefix`: prefix such as `item` or `semeval` when `id.kind: generated`
- `sampling.n_v1`, `sampling.n_v2_new`, `sampling.seed`: benchmark sampling controls
- `text.truncations`: optional field-specific truncation rules

## Example

The bundled example in [`examples/minimal_custom_task`](../examples/minimal_custom_task)
uses:

```yaml
name: minimal_custom_task
label_kind: binary
label_key: relevant
labels:
  - relevant
sampling:
  n_v1: 4
  n_v2_new: 0
  seed: 20260422
id:
  kind: column
  column: item_id
text:
  template: |
    Text: {text}
ground_truth:
  column: gt_relevant
```

## Commands

Run the example task without paid API calls:

Shell 1:

```bash
python3 examples/local_openai_stub_server.py --max-requests 4
```

Shell 2:

```bash
python3 code/benchmark.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/local_openai_stub.yaml \
  --output output/custom_predictions.csv

python3 code/build_summary.py \
  --task-dir examples/minimal_custom_task \
  --model-manifest examples/minimal_custom_models/local_openai_stub.yaml \
  --predictions output/custom_predictions.csv \
  --output output/custom_summary.csv
```

The first command starts a deterministic local OpenAI-compatible test server and
exits after four requests. In a separate shell, run the benchmark and summary
commands. This path does not use OpenAI, Anthropic, or DeepSeek.

The smoke test succeeds when `output/custom_predictions.csv` has four rows and
`output/custom_summary.csv` has one row with `parse_err_rate = 0` and
`headline_f1 = 1`.

If you omit `--model-manifest`, the runner uses the built-in model set. That can
call paid API models when API keys are present in the environment.

## Ground Truth Details

For binary and categorical tasks, set:

```yaml
ground_truth:
  column: gt_relevant
```

For multi-binary tasks, either provide columns named `gt_{label}` for every
entry in `labels`, or provide an explicit mapping:

```yaml
ground_truth:
  columns:
    Activities: gt_activities
    Budget: gt_budget
```

Binary and multi-binary ground-truth values must be coercible to `0` or `1`.
Categorical ground-truth values must exactly match one of the labels in the
manifest.

## Task inventory contract

The repo tracks task counts in [`docs/task_inventory.md`](task_inventory.md).
That file is generated from `tasks/`, not hand-maintained.

After adding, removing, or changing a built-in task manifest under `tasks/`,
run:

```bash
python3 code/task_inventory.py --write
python3 code/task_inventory.py --check
```

The test suite includes a stale-doc check for this file, so adding a new task
without refreshing the inventory should fail before the change is merged.

External custom tasks passed with `--task-dir` or `--task-manifest` do not
require updating the repo task inventory.
