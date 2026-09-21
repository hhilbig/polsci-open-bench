# Benchmark page: proposed text edits and cuts

- **Page:** `output/sidecar/refresh_20260910_release_33/preview/llm-benchmark/index.html`
- **Generated from:** `code/build_refresh_preview.py` (page text) and
  `code/render_refresh_paper_figures.R` (figure titles and captions)
- **Date:** 2026-09-21

**Scope.** Every paragraph and figure caption on the page. I checked each one
with hanno-voice (first person "I", as in the paper), clarity-reviewer and
avoid-ai-writing.

**What this document contains:**
- rewrites and cuts where text is redundant;
- one new section on where the models ran;
- one substantive choice, item 5, which is flagged.

Items with no proposed change are not listed.

**Current text.** "Before" quotes the text as it appears on the page. Values in
{braces} are filled in from the release when the page is built.

**Nothing is applied until you comment.** An empty **HH:** line means the item
is not approved.

**Estimated effect** if every item is accepted:
- the visible page loses about 90 words and the collapsed section loses five of
  its seven figures;
- the new section adds about 130 words.

---

## 1. Intro: drop the sentence that Methods repeats

**Problem:** the last sentence of the first paragraph is repeated almost word
for word in the Methods section. The model count is also more useful split into
API and open-weight models.

Before:
> Can researchers code political science texts with open-weight language models instead of commercial APIs? This page compares {31} models on the same {3,300} texts, 100 from each of {33} coding tasks drawn from political science papers and public datasets. Every model receives the same texts, prompts and gold labels.

After:
> Can researchers code political science texts with open-weight language models instead of commercial APIs? This page compares {31} models, {11} commercial APIs and {20} open-weight models, on the same {3,300} texts: 100 from each of {33} coding tasks drawn from political science papers and public datasets.

**HH:** Sure 

## 2. Results intro: cut the second sentence

**Problem:** the second sentence repeats the title of the collapsed section
below it.

Before:
> The figures below focus on {18} models: all API models, the most recent open-weight models and two Llama 70B models as reference points. The section further down repeats them for all models.

After:
> The figures below show {18} models: all API models, the most recent open-weight models and two Llama 70B models as reference points.

**HH:** sure 

## 3. Caption, overall performance

**Problem:** "printed on the right" describes what the reader can already see.
The colour sentence can be shorter.

Before:
> Each row shows one model. Points show mean F1 across the 33 tasks, printed on the right, and grey bars show 95% intervals from resampling tasks. Blue marks open-weight models and grey marks API models.

After:
> Each row shows one model. Points mark mean F1 across the 33 tasks, and grey bars mark 95% intervals from resampling tasks. Open-weight models are blue and API models grey.

**HH:** sure 

## 4. Caption, cost and performance: state the finding first

**Problem:** the current caption describes the axes but not what the figure
shows. The papers open a results paragraph with the finding.

Before:
> Each point shows one API model: its mean F1 across the 33 tasks against its cost per 1,000 texts at standard prices in September 2026. Hollow points mark costs estimated from token counts rather than taken from provider bills. The dashed line marks the best open-weight model that runs on one GPU, which has no per-text charge.

After:
> Higher prices buy small gains in accuracy. Claude Opus 5 costs about 220 times as much per text as Jev 1.13 and scores 0.054 higher. Seven of the 11 API models score above the best open-weight model that runs on one GPU (dashed line), which has no per-text charge. Costs are per 1,000 texts at standard prices in September 2026; hollow points are estimated from token counts rather than taken from provider bills.

**HH:** Sure 

## 5. Task-gap figure: compare against all one-GPU open models (substantive)

**Problem:** the featured task-gap figure compares the 11 API models with only
the 6 featured open models. The older open models it leaves out, such as Qwen
2.5 72B, win some tasks. The featured figure therefore makes open models look
worse than the full panel does:

| Open models compared | Tasks where open matches or beats API | Mean gap |
|---|---:|---:|
| 6 featured | 8 of 33 | 0.034 |
| All 19 on one GPU | 14 of 33 | 0.013 |

The 14-of-33 and 0.013 figures are the ones in the README.

**Proposal:** the featured figure uses all 19 one-GPU open models, and the
duplicate all-model version in the collapsed section is deleted (item 8). The
complexity figure follows the same rule (item 7).

Before (caption):
> Each row shows one task. Points show the best score among the 11 API models minus the best score among the 6 open-weight models that run on one GPU. Blue points left of zero mark tasks where an open model performs better. Because the best model is chosen after observing the results, these gaps describe the best case for each group.

After (caption):
> Each row shows one task. Points show the best API score minus the best score among the 19 open-weight models that run on one GPU. An open model matches or beats the best API model on 14 of the 33 tasks (blue points). 

NOTE: this changes which models the figure uses, not only its wording.

**HH:** made edits, good

## 6. Caption, annotation types

**Problem:** "below" is wrong once the figure sits above the task selector
anyway, and "all tasks" repeats "overall".

Before:
> Mean F1 within the five annotation types used in the paper, with each task weighted equally. Models appear in the same order in every panel, ranked by their mean across all tasks. These types differ from the categories in the task selector below.

After:
> Mean F1 within the five annotation types used in the paper, with each task weighted equally. Models appear in the same order in every panel, ranked by their overall mean. These types differ from the categories in the task selector.

**HH:** Sure 

## 7. Caption, coding complexity: state the finding first

**Problem:** the caption defines the groups but does not say what the figure
shows. The numbers below assume item 5, with all 19 one-GPU open models; I
recompute them if item 5 is rejected.

Before:
> Tasks grouped by coding complexity, following the paper. High-complexity tasks allow several labels per text or use at least eight labels in practice; medium-complexity tasks use at least three labels or have a prompt of 300 words or more; the remaining tasks are low complexity. The number of labels in practice is the exponential of the entropy of the gold labels, which counts rare labels less than common ones. Open models are those that run on one GPU.

After:
> The gap between API and open models grows with coding complexity. On the 18 low-complexity tasks, the best open model on each task scores as high as the best API model; on medium- and high-complexity tasks it trails by about 0.04 F1. High-complexity tasks allow several labels per text or use at least eight labels in practice, and medium-complexity tasks use at least three labels or have a prompt of 300 words or more. The number of labels in practice is the exponential of the entropy of the gold labels, which counts rare labels less than common ones.

**HH:** Sure 

## 8. Cut three duplicate figures from the collapsed section

**Problem:** these three figures repeat the featured versions with more models.
Once item 5 is accepted, the task-gap and complexity duplicates would be
identical to the featured figures. The 31-row annotation-type figure is very
tall, and its values are in the figure JSON download.

Cut:
- "API and open models by task, all models" (`fig-best-local-api-gap`);
- "Coding complexity, all models" (`fig-complexity`);
- "Performance by annotation type, all models" (`fig-family`).

Keep "Overall performance, all models", which is now the only place the full
ranking of 31 models appears.

**HH:** sure 

## 9. Cut the label-structure figure

**Problem:** the figure shows that the API advantage grows with the number of
labels, which the complexity figure already shows more readably. The per-task
values stay in the figure JSON download.

**HH:** Sure 

## 10. Cut the runtime-per-1,000 figure and move its range into the speed caption

**Problem:** the runtime figure plots the same seven measurements as the speed
figure, on a different scale.

Before (speed caption):
> Each point shows one open-weight model: mean F1 against generation time per text. All models coded the same 2,142 texts on one RTX PRO 6000 GPU with identical settings. Load and queue times are excluded.

After (speed caption):
> Each point shows one open-weight model: mean F1 against generation time per text. The seven models coded the same 2,142 texts on one RTX PRO 6000 GPU with identical settings, and they need between 0.6 and 1.4 minutes per 1,000 texts. Load and queue times are excluded.

**HH:** Sure 

## 11. Intro to the collapsed section

**Problem:** after items 8–10, only two figures remain in the collapsed section.
The current intro describes a rule about the removed figures.

Before (summary line and paragraph):
> More analyses and all-model figures
>
> These figures include all models. Comparisons between API and open models use only open models that run on a single GPU.

After:
> All 31 models and open-model speed
>
> The first figure repeats the overview for all 31 models. The second compares the speed of the seven open-weight models that ran under identical settings.

**HH:** sure 

## 12. Excluded models: cut the "Pending" sentence

**Problem:** no model is pending any longer, so the definition refers to nothing
in the list.

Before:
> The models below are not ranked. Not evaluated means that I could not run the model, for the reason listed. Excluded means that a run finished but violated the benchmark's rules. Pending means that the run is not finished. None of these categories implies a score of zero.

After:
> The models below are not ranked. Not evaluated means that I could not run the model, for the reason listed. Excluded means that a run finished but violated the benchmark's rules.

**HH:** Sure , made edit 

## 13. New section: where the models ran (before Methods, visible)

**Problem:** the page does not say what hardware the open models used, or that
the strongest configurations were not tested. Both facts come from the release
records:
- API run dates of 10–20 September 2026;
- one RTX PRO 6000 Blackwell GPU per open model, two for Qwen3.8 Flash-Next;
- vLLM 0.26 (a development build for Flash-Next), temperature 0 and a 256-token output limit (Jev has none);
- the documented exclusions of Kimi K3, GLM-5.3 and MiniMax M3.

New text, under the heading "Where the models ran":
> I called the API models through each provider's API between 10 and 20 September 2026. Requests to OpenAI and Anthropic went through their batch endpoints; the other providers received one request per text. Each open-weight model ran on a single NVIDIA RTX PRO 6000 Blackwell GPU with 96 GB of memory on UC Davis's Hive computing cluster, using vLLM 0.26 at temperature 0. Qwen3.8 Flash-Next needed two of these GPUs and a development version of vLLM. Reasoning is disabled where a model allows it and otherwise set to its lowest level. Every model may return at most 256 tokens, except Jev 1.13, which returns a choice among the labels rather than free text.
>
> Two groups of models were not tested. I did not run the API models with extended reasoning, which raises cost and response time, and I could not run the largest open-weight models (Kimi K3, GLM-5.3 and MiniMax M3) on this hardware. The top scores on this page may therefore understate what the strongest configurations of these models reach.

**HH:** Sure 

## 14. Methods: cut "Settings"

**Problem:** item 13 covers every sentence of this paragraph except the pointer
to the downloads. That pointer moves to the Downloads paragraph (item 15).

Before:
> Settings. Reasoning is disabled where a model allows it and otherwise set to its lowest level. API models may return at most 256 tokens. The downloads report model versions, quantization and runtime settings, which can affect performance.

After: *(paragraph removed)*

**HH:** I dont want this section  

## 15. Downloads paragraph

**Problem:** this paragraph takes over the pointer from item 14.

Before:
> The downloads contain aggregate scores and run records. I do not include the texts or item-level predictions while redistribution rights are checked.

After:
> The downloads contain aggregate scores and the run records for each model, including model versions, quantization and runtime settings. I do not include the texts or item-level predictions while redistribution rights are checked.

**HH:** I dont want downloads here

---

## Implementation record (2026-09-21)

**Applied:** items 1–13 as proposed, with your edits to items 5 and 12 used
verbatim:
- the last sentence of the item 5 caption is removed;
- the "Neither category…" sentence in item 12 is removed.

**Item 14:** only the "Settings" paragraph is removed (your answer to the
follow-up question). Methods keeps Sample, Scoring and Limits.

**Item 15:** the whole Downloads section is removed (your answer to the
follow-up question):
- The link to the code on GitHub moved to the footer.
- The "Downloads" navigation link is replaced by "Hardware", which points to
  the new section.
- The task selector's fallback text no longer mentions the downloads.
- The download files are still written and checked against the release, but the
  page no longer links to them.

**Deviations from the proposed text:**
- **Computed from the data:** the counts and ratios in the cost, task-gap and
  speed captions (220 times, 0.054, seven of 11, 19 models, 14 of 33, 0.6–1.4
  minutes) and the model counts in the intro. They are not typed in, so they
  update if the release changes. The rendered values equal the proposed ones.
- **Typed in:** "about 0.04 F1" in the complexity caption stays fixed text.
- **Item 11:** the model counts in the new summary line and intro come from the
  release.

**Verification:**
- The page rebuilds, and `--verify-only` passes (31 models, 33 tasks).
- The overview figures' means match the release.
- The refresh preview, release and audit tests pass.
- In the browser: 7 figures, all images load, no horizontal scroll, and the
  task selector and paired comparison work.
