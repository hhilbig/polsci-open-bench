# Public Release Notes

Last updated: 2026-09-18

This note summarizes the current public release state. The report is the
authoritative narrative artifact; the repository stores the data, prompts,
summaries, figures, and reproduction instructions that support it.

## Current Release

- The public benchmark covers 34 political science classification tasks.
- The serial benchmark covers 10 models: 6 local open-weight models and 4
  commercial API models.
- The serial grid is complete: 34 tasks x 10 models = 340 task-model pairs.
- The release includes 164,250 serial model-item classifications.
- The local prompt-batching grid is complete for the six local models with 10
  items per prompt: 34 tasks x 6 models = 204 task-model pairs.
- The main public report is [`output/report_pdf.pdf`](../output/report_pdf.pdf).

## Main Artifacts

- Serial predictions: [`output/predictions.csv`](../output/predictions.csv)
- Serial summary: [`output/summary.csv`](../output/summary.csv)
- Prompt-batched predictions:
  [`output/predictions_batched.csv`](../output/predictions_batched.csv)
- Prompt-batched summary:
  [`output/summary_batched.csv`](../output/summary_batched.csv)
- Local 10-item comparison summary:
  [`output/summary_batched_local_b10.csv`](../output/summary_batched_local_b10.csv)
- Task inventory: [`docs/task_inventory.md`](task_inventory.md)
- Output schema: [`docs/schema.md`](schema.md)
- Reproduction guide: [`docs/reproduce.md`](reproduce.md)

## Maintenance Since The 34-Task Release

The 34-task task set itself is unchanged. Maintenance changes landed after it:

- 2026-05-20: Report assets were rebuilt (`code/build_report_assets.R`, figures,
  appendix tables, references, and the report PDF). The two CAP task manifests
  had their `source:` citation strings corrected. Data, labels, and prompts were
  unchanged, so no rerun was required and results are unaffected.
- 2026-06-06: Email addresses embedded in dataset text were replaced with
  `[EMAIL]` across the nine affected corpora, removing personal contact
  information (notably citizen emails in the Erlich ATI data). No raw email
  addresses remain in `data/`; the text is otherwise unchanged.
- 2026-06-13: Added Qwen3.5 35B-A3B as a sixth local model, scored serially and
  at 10 items per prompt on all 34 tasks. The benchmark now covers ten models
  (six local, four API). Qwen3 30B-A3B is retained as a prior-generation
  reference. Qwen3.5 improves on Qwen3 30B-A3B by 0.044 mean F1 (0.637 vs 0.593),
  but runs about four times slower. The batch parser was extended to recover
  newline-delimited JSON (JSONL) output, which Qwen3.5 emits for batched prompts;
  array-emitting models are unaffected.

The published predictions predate the 2026-06-06 redaction and were generated on
the pre-redaction text, so the released corpora now differ from the exact inputs
scored for a small number of items. The redacted tokens are incidental contact
strings rather than classification signal, so the effect on F1, accuracy, and
MCC is negligible. Predictions were not regenerated.

## What Changed Since The Earlier Release

The earlier public repository described a smaller benchmark. The current release
promotes the expanded task and model set into one public 34-task benchmark.

- The task set now contains 34 task manifests under [`tasks/`](../tasks).
- The model set now contains six local Ollama models and four API models:
  OpenAI, Anthropic, and DeepSeek.
- The serial prediction and summary files were rebuilt for the full 34-task
  grid.
- Local prompt batching with 10 items per prompt was completed for all 34 tasks
  and all six local models.
- The report was rebuilt around the current results, including local/API gaps,
  coding complexity, runtime, batching reliability, cost, and appendix tables.
- Public docs were updated to use the current terminology: "main F1" in prose,
  "items" rather than "texts" where the unit is a benchmark row, and "10 items
  per prompt" rather than shorthand batching notation.

## Result Notes

- Local models match or exceed API performance on 10 of 34 tasks when comparing
  the best local and best API model within each task.
- On average, the best API model exceeds the best local model by 0.011 main F1.
- The four strongest models, Claude Sonnet 4.6, gpt-5.5, Gemma 4 31B, and
  DeepSeek V4 Pro, are separated by 0.021 mean F1.
- API models have their clearest edge on high-complexity tasks with many active
  labels, long codebooks, or multiple outputs per item.
- Local prompt batching usually reduces runtime per item, but some model-task
  pairs return invalid response formats or invalid labels.

## Reliability And Scope

- Performance metrics are computed over usable outputs. Response-format and
  label failure rates therefore need to be read alongside F1, accuracy, and MCC.
- Provider Batch API mode is different from prompt batching. Provider Batch API
  changes request processing and billing; prompt batching places several items
  into one model call.
- Granite 4.1 8B is not part of the current nine-model benchmark. It was tabled
  after a completed run showed an elevated parse-error rate.
- The benchmark studies prompt-based classification. It is not a replacement for
  supervised baselines when large labeled datasets exist.

## Release Checklist

- [x] Report PDF rebuilt.
- [x] README updated for the 34-task release.
- [x] Task inventory refreshed.
- [x] Output schema updated for all 34 tasks.
- [x] Twitter/X thread draft saved in [`docs/twitter_thread.md`](twitter_thread.md).
- [x] Appendix tables kept in the PDF.

## 2026-09-18: Audit fixes

A second source-fidelity audit (see
[`docs/task_source_fidelity_audit.md`](task_source_fidelity_audit.md)) asked
whether each gold label is recoverable from the text the model is shown, and
whether the harness scores every model on equal terms. The first audit asked only
whether labels come from their source, so none of this was visible there.

### Metric correction (changes published numbers)

Categorical macro F1 averaged over every label declared in a manifest, including
labels with no gold support in the sample. Such a label can only score 0, so it
entered the mean as a zero. `code/scoring.py` is now the single implementation
and averages only over labels with gold support.

Three tasks change; the other 31 are identical:

| Task | Published | Corrected | Cause |
|---|---|---|---|
| `mellon_bes_mii_2024` | 0.505 | 0.721 | 15 of 50 declared labels have no gold support |
| `cap_crs_policy_topic` | 0.669 | 0.702 | 1 of 21 |
| `haunss_papea_claims` | 0.396 | 0.426 | 2 of 28 |

(Values are means across the ten models; per-model figures are in
`output/summary.csv`.)

Every qualitative conclusion is unchanged. Model ordering is identical, the mean
best-API-minus-best-local gap moves from +0.0108 to +0.0112, and local models
still match or exceed API models on 10 of 34 tasks. All ten model means rise by
roughly 0.009, because the correction removes a penalty that applied to every
model equally.

`output/summary.csv` and `output/summary_batched.csv` now hold the corrected
metric. The published versions are preserved as
`output/summary_legacy_all_labels.csv` and
`output/summary_batched_legacy_all_labels.csv`, and
`python3 code/build_summary.py --legacy-all-labels` reproduces them exactly.
`code/build_frontier_2026.py` and `code/summarize_hive_bakeoff.py` are pinned to
the legacy metric on purpose so the frozen frontier and refresh panels keep
reproducing; migrating them is a separate step that requires rebuilding those
panels. For the same reason, the 18-task frontier panel keeps its frozen
calibration: the constants in `experiments/frontier_panel_18.yaml` were fitted on
legacy-metric model means, and the correction moves the fitted intercept from
0.05447 to 0.05167. `tests/test_frontier_builder.py` therefore reads
`output/summary_legacy_all_labels.csv`. Recalibrating that panel against the
corrected metric is a deliberate analysis decision, not a side effect of a
scoring fix, and has not been made.

**The report PDF has not been rebuilt.** Its prose quotes model means that the
correction moves, and paper text is not edited without explicit approval. Figures
and appendix tables in `output/figures/` and `output/tables/` have been rebuilt.

### Task versioning

Corrected task definitions live in `tasks_v2/` with prompts in `prompts_v2/`.
The v1 manifests, prompts and predictions are untouched, so published results stay
reproducible. Two loader mechanisms were added to `code/task_registry.py`:
`ground_truth.exclude_labels` drops gold classes that record coder uncertainty
rather than a property of the text, and `ground_truth.label_map` renames gold
values at load time without regenerating a cleaned CSV from a source archive.

| v2 task | Fix | Item-paired with v1 |
|---|---|---|
| `brandt_gtd_attack_type` | Drops the `Unknown` class: GTD assigns it when the source report did not specify the method, which the model cannot see. 47 of 500 items, 3.2% accuracy. | No, rows dropped |
| `cap_crs_policy_topic` | Reads the prebuilt `text` column, so the 190 of 500 items with no summary stop rendering an empty `Summary:` heading. Topic 5 renamed to `Labor`. Codebook definitions added. | Yes |
| `cap_party_platform_policy_topic` | Topic 5 renamed to `Labor`. Codebook definitions added. | Yes |
| `halterman_keith_bfrs` | Removes the source header telling the model to write the bare label "with no other text", which contradicted the prompt's own JSON-output block. | Yes |
| `halterman_keith_cmp` | Same header removal. | Yes |

No v2 predictions have been generated yet.

### Open

- `osnabruegge_cross_domain_topic` and `halterman_ccc_protest` have gold-label
  problems that cannot be fixed without re-obtaining their source archives.
  Neither has a build script and no source URL is recorded.
- The constrained-decoding experiment is configured
  (`experiments/schema_parity_20260918_{guided,plain}.yaml`) but not run.
- Deduplication is documented and deliberately deferred.
