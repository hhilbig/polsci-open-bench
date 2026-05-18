# Twitter / X Thread: 34-Task Benchmark

Four-post thread to accompany the current PDF report.

Repo link: <https://github.com/hhilbig/polsci-open-bench>

## Tweet 1 — Decision Hook

Attach:

- `output/figures/fig-best-local-api-gap.png`

Text:

> For routine text classification, commercial APIs are no longer the obvious default.
>
> I benchmark 5 local open-weight LLMs vs. 4 commercial API models on 34 political science coding tasks.
>
> Best local matches or beats best API on 9/34 tasks. The average API advantage is 0.015 F1.

## Tweet 2 — Concrete Speed / Quality Example

Attach:

- `output/figures/fig-local-runtime-per-1000.png`
- Optional: `output/figures/fig-speed.png`

Text:

> The local option is practical, not just theoretically interesting.
>
> Gemma 4 26B runs on a 32 GB MacBook, trails gpt-5.5 by 0.021 F1 on average, and classifies about 4,000 documents in ~52 minutes, or ~30 minutes with 10-item prompt batching.

## Tweet 3 — Complexity

Attach:

- `output/figures/fig-complexity.png`
- Optional: `output/figures/fig-label-structure-gap.png`

Text:

> APIs still have their clearest edge on complex coding tasks.
>
> Simpler = short binary/3-class tasks: relevance, stance, incivility.
>
> Complex = long codebooks, many labels, or multiple outputs: policy topics, events/protests.
>
> High complexity: best API 0.605 F1 vs local 0.555.

## Tweet 4 — Validation Workflow

Attach:

- Optional: `output/figures/fig-mean-f1.png`

Text:

> Practical takeaway: validate before production coding.
>
> Test a cheap API, a flagship API, and two local models. Report F1, accuracy, MCC, and invalid-output rates.
>
> Batching helps, but check format failures before scale.
>
> PDF/code: https://github.com/hhilbig/polsci-open-bench

## Notes

- The first tweet now uses the decision hook. The 147,825 model-item
  classifications can be added if there is space; it refers to the 34-task
  one-item-at-a-time benchmark: 16,425 labeled items across tasks times 9 models.
- The 4,000-document runtime example is grounded in the median cleaned corpus size
  across the 34 task files, rounded from 3,816.
- Gemma 4 26B timing uses median runtime per item across the 34-task benchmark:
  about 0.782 seconds one-at-a-time and 0.444 seconds with 10-item prompt batching.
- Gemma 4 26B's average F1 is 0.639, compared with 0.660 for gpt-5.5.
- Keep "GPT/Claude" in the hook for audience clarity, even though DeepSeek V4
  Pro is also included in the API model set.
