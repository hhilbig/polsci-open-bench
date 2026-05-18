# Twitter / X Thread: 34-Task Benchmark

Four-post thread to accompany the current PDF report.

Repo link: <https://github.com/hhilbig/polsci-open-bench>

## Tweet 1 — Main Claim And Results

Attach:

- `output/figures/fig-mean-f1.png`
- `output/figures/fig-best-local-api-gap.png`
- Optional: `output/figures/fig-family.png`

Text:

> Update on my open-weight LLM benchmark: I now compare 5 local models and 4 API models on 34 political-science text-classification tasks, covering 147,825 model-item classifications.
>
> Across these tasks, local models remain competitive with GPT/Claude, especially when cost, privacy, or reproducibility matter.
>
> The best local model matches or beats the best API model on 9 of 34 tasks, and task-level winners are split across 8 of 9 models.

## Tweet 2 — Concrete Speed / Quality Example

Attach:

- `output/figures/fig-speed.png`

Text:

> One concrete example: Gemma 4 26B runs locally on a 32 GB MacBook, trails gpt-5.5 by only 0.021 F1 on average, and classifies about 4,000 documents in ~52 minutes, or ~30 minutes with 10-item prompt batching.

## Tweet 3 — Complexity

Attach:

- `output/figures/fig-complexity.png`
- Optional: `output/figures/fig-label-structure-gap.png`

Text:

> APIs still have their clearest edge on harder coding tasks.
>
> In this benchmark, "easy" means short binary or 3-class tasks like relevance, stance, or incivility. "Hard" means long codebooks, many active labels, or multiple outputs per item, e.g. policy topics or event/protest coding.
>
> Among high-complexity tasks, the best API model averages 0.605 F1 vs 0.555 for the best local model.

## Tweet 4 — Link / Optional Bottom Line

Attach:

- Optional: `output/figures/fig-local-runtime-per-1000.png`

Text:

> Bottom line: local open-weight models are now a serious option for applied text coding, but batching still needs reliability checks because some task-model pairs return invalid labels or response formats.
>
> PDF/code: https://github.com/hhilbig/polsci-open-bench

## Notes

- The first tweet's 147,825 model-item classifications refers to the 34-task
  one-item-at-a-time benchmark: 16,425 labeled items across tasks times 9
  models.
- The 4,000-document runtime example is grounded in the median cleaned corpus size
  across the 34 task files, rounded from 3,816.
- Gemma 4 26B timing uses median runtime per item across the 34-task benchmark:
  about 0.782 seconds one-at-a-time and 0.444 seconds with 10-item prompt batching.
- Gemma 4 26B's average F1 is 0.639, compared with 0.660 for gpt-5.5.
- Keep "GPT/Claude" in the hook for audience clarity, even though DeepSeek V4
  Pro is also included in the API model set.
