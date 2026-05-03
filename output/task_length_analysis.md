# Task length and complexity audit

Generated: 2026-05-02

This is exploratory. The current benchmark still has a small number of tasks, so the length/performance relationships should be read as descriptive rather than definitive.

## Cross-task correlations

- `Median sampled item chars` vs `local_minus_api_best_f1`: `0.358`
- `Prompt chars` vs `local_minus_api_best_f1`: `-0.273`
- `Label count` vs `local_minus_api_best_f1`: `-0.104`
- `Complexity proxy` vs `local_minus_api_best_f1`: `-0.089`

## Task table

| task | family | label_count | prompt_chars | item_chars_median | item_chars_p90 | local_best_model | local_best_f1 | api_best_model | api_best_f1 | local_minus_api_best_f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| halterman_ccc_protest | Event coding | 4 | 2319 | 2678.0 | 5243.7 | qwen3:14b-q4_K_M | 0.499 | gpt-5.5 | 0.453 | 0.046 |
| halterman_keith_bfrs | Event coding | 12 | 13554 | 181.0 | 394.4 | gemma4:31b-it-q4_K_M | 0.659 | gpt-5.5 | 0.693 | -0.035 |
| haunss_papea_fgz_forms | Event coding | 7 | 1195 | 41.0 | 78.1 |  |  |  |  |  |
| halterman_keith_cmp | Policy-topic coding | 7 | 2538 | 132.0 | 237.1 | gemma4:31b-it-q4_K_M | 0.577 | gpt-5.5 | 0.597 | -0.02 |
| mellon_bes_mii_2024 | Policy-topic coding | 50 | 3999 | 33.0 | 46.0 | mistral-small:24b-instruct-2501-q4_K_M | 0.526 | gpt-5.5 | 0.547 | -0.021 |
| osnabruegge_cross_domain_topic | Policy-topic coding | 8 | 2603 | 367.5 | 3052.1 |  |  |  |  |  |
| ballard_incivility | Relevance / Incivility | 1 | 1747 | 147.0 | 252.2 | gemma4:31b-it-q4_K_M | 0.522 | claude-sonnet-4-6 | 0.563 | -0.041 |
| brandt_political_relevance | Relevance / Incivility | 1 | 624 | 1917.0 | 3174.0 |  |  |  |  |  |
| gilardi_relevance | Relevance / Incivility | 1 | 2244 | 242.0 | 304.1 | gemma4:31b-it-q4_K_M | 0.915 | gpt-5.5 | 0.95 | -0.035 |
| rheault_line_of_fire_incivility | Relevance / Incivility | 1 | 745 | 139.0 | 172.0 |  |  |  |  |  |
| chae_semeval_stance | Sentiment / Stance / Tone | 3 | 1268 | 133.0 | 150.0 | gemma4:31b-it-q4_K_M | 0.772 | gpt-5.5 | 0.773 | -0.0 |
| gilardi_stance | Sentiment / Stance / Tone | 3 | 2627 | 220.5 | 302.0 | gemma4:31b-it-q4_K_M | 0.529 | claude-sonnet-4-6 | 0.568 | -0.039 |
| ornstein_scotus_sentiment | Sentiment / Stance / Tone | 3 | 1191 | 176.5 | 298.0 | qwen3:14b-q4_K_M | 0.665 | gpt-5.4-nano | 0.592 | 0.073 |
| wesleyan_creative_ads_2022 | Sentiment / Stance / Tone | 3 | 1604 | 1381.0 | 3049.1 | qwen3:14b-q4_K_M | 0.673 | gpt-5.4-nano | 0.711 | -0.038 |
