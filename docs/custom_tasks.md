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

Run the example task:

```bash
python3 code/benchmark.py \
  --task-dir examples/minimal_custom_task \
  --output output/custom_predictions.csv

python3 code/build_summary.py \
  --task-dir examples/minimal_custom_task \
  --predictions output/custom_predictions.csv \
  --output output/custom_summary.csv
```
