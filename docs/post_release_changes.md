# Public Release Notes

Last updated: 2026-06-13

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
