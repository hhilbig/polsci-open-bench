# Twitter / X thread: polsci-open-bench

Six-post thread to accompany the PDF report. Each post under 280 characters
(the X free-tier limit; X Premium allows 25,000 chars but most readers see
truncated thumbnails on phones, so keep it tight).

Image attachments noted as "> Attach Figure N" — these refer to the four
PDF figures.

Replace `<LINK>` with the GitHub repo URL or a short link to the PDF before
posting.

---

### Post 1 — Hook

<!-- 271 chars -->

Can local LLMs do political-science text classification well enough to replace API models for some applied research tasks?

I ran a small benchmark: 10 classification tasks, 6 models, 500 items per task, 30,000 predictions. The goal is practical guidance, not a definitive leaderboard.

---

### Post 2 — Headline performance result

<!-- 268 chars -->

Main result: the best local model is essentially tied with the strongest API model on average.

gpt-5.5 has the highest mean F1, but Gemma 4 31B is only 0.002 points behind. Most task-level differences are within bootstrap-CI overlap, so I would not read this as a sharp ranking.

> Attach Figure 1 (mean F1 with per-task dots)

---

### Post 3 — Task heterogeneity

<!-- 271 chars -->

The bigger lesson is task heterogeneity. Different models lead on different families: stance/tone, relevance/incivility, event coding, and policy-topic coding.

For applied researchers, the safest workflow is still a small validation sample on the actual task before committing.

> Attach Figure 3 (mean F1 by task family)

---

### Post 4 — Speed

<!-- 261 chars -->

Speed is not simply "API fast, local slow."

Gemma 4 31B is the slowest in the grid, but Qwen3-30B-A3B (a local mixture-of-experts model) is the fastest. Local inference can be practical on consumer hardware; model architecture and quantization matter more than provenance.

> Attach Figure 2 (mean F1 vs median latency, log-x)

---

### Post 5 — Batching

<!-- 277 chars -->

Batching helps, but not everywhere.

On short-prompt tasks, b=10 gives 1.5–4x speedups with little F1 loss. On long-codebook tasks, batching becomes fragile: gpt-5.5 batched on Halterman CCC at b=10 emits malformed JSON for 68% of items. gpt-5.4-nano is more robust there.

> Attach Figure 4 (speedup vs delta-F1, batched cells)

---

### Post 6 — Bottom line + link

<!-- 277 chars -->

Bottom line: local open-weight models are credible for many political-science classification workflows, but not as a blanket replacement for API models.

Task-level reversals are common. The aggregate leaderboard is useful for screening candidates, not for choosing a production model without validation.

PDF + code: <LINK>

---

## Notes for posting

- Order: 1 → 2 → 3 → 4 → 5 → 6, replying to each previous post.
- Alt text for images: keep short ("Figure 1: bar chart of mean F1 across 6 models with per-task dots overlaid").
- Bluesky: 300-char limit; thread reads identically.
- Mastodon: 500-char limit; could merge posts 5 and 6 if desired.
