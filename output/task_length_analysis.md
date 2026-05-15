# Task length and complexity audit

Generated: 2026-05-15

This is exploratory. The current benchmark still has a small number of tasks, so the length/performance relationships should be read as descriptive rather than definitive.

## Cross-task correlations

- `Median sampled item chars` vs `local_minus_api_best_f1`: `0.288`
- `Prompt chars` vs `local_minus_api_best_f1`: `-0.007`
- `Label count` vs `local_minus_api_best_f1`: `-0.334`
- `Complexity proxy` vs `local_minus_api_best_f1`: `-0.126`

## Task table

| task | family | coding_complexity | effective_label_count | label_count | prompt_chars | item_chars_median | item_chars_p90 | local_best_model | local_best_f1 | api_best_model | api_best_f1 | local_minus_api_best_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| politicause_causal_relation | Causal relation detection | Low | 1.863 | 1 | 700 | 167.0 | 278.0 | gemma4:31b-it-q4_K_M | 0.664 | claude-sonnet-4-6 | 0.716 | -0.053 |
| brandt_gtd_attack_type | Event coding | Medium | 5.088 | 9 | 1391 | 296.5 | 496.2 | mistral-small:24b-instruct-2501-q4_K_M | 0.575 | gpt-5.4-nano | 0.617 | -0.042 |
| burnham_polnli_event_entailment | Event coding | Low | 1.997 | 1 | 467 | 223.5 | 422.1 | gemma4:31b-it-q4_K_M | 0.974 | gpt-5.5 | 0.973 | 0.001 |
| douglass_icbe_sentence_event_type | Event coding | Medium | 4.629 | 5 | 942 | 160.0 | 255.4 | gemma4:26b | 0.53 | deepseek-v4-pro | 0.506 | 0.024 |
| halterman_ccc_protest | Event coding | Medium | 2.537 | 4 | 2319 | 2678.0 | 5243.7 | qwen3:14b-q4_K_M | 0.499 | gpt-5.5 | 0.453 | 0.046 |
| halterman_keith_bfrs | Event coding | Medium | 7.836 | 12 | 13554 | 181.0 | 394.4 | gemma4:26b | 0.681 | gpt-5.5 | 0.693 | -0.012 |
| haunss_papea_claims | Event coding | High | 12.957 | 28 | 665 | 57.0 | 140.2 | gemma4:31b-it-q4_K_M | 0.413 | gpt-5.5 | 0.462 | -0.049 |
| haunss_papea_fgz_forms | Event coding | Medium | 4.341 | 7 | 1195 | 41.0 | 78.1 | gemma4:31b-it-q4_K_M | 0.958 | gpt-5.5 | 0.965 | -0.007 |
| plover_cameo_event | Event coding | High | 15.344 | 18 | 464 | 165.0 | 216.8 | gemma4:31b-it-q4_K_M | 0.577 | gpt-5.5 | 0.654 | -0.076 |
| burnham_polnli_entailment | Hypothesis-conditioned classification | Low | 1.97 | 1 | 409 | 276.0 | 586.1 | gemma4:31b-it-q4_K_M | 0.89 | gpt-5.5 | 0.908 | -0.018 |
| cap_crs_policy_topic | Policy-topic coding | High | 15.846 | 21 | 765 | 901.0 | 3841.1 | gemma4:31b-it-q4_K_M | 0.69 | gpt-5.5 | 0.732 | -0.043 |
| cap_party_platform_policy_topic | Policy-topic coding | High | 17.076 | 21 | 751 | 184.5 | 290.0 | gemma4:31b-it-q4_K_M | 0.647 | claude-sonnet-4-6 | 0.731 | -0.085 |
| erlich_ati_topics | Policy-topic coding | High | 13.829 | 7 | 1282 | 301.0 | 904.4 | gemma4:31b-it-q4_K_M | 0.477 | claude-sonnet-4-6 | 0.502 | -0.025 |
| halterman_keith_cmp | Policy-topic coding | Medium | 5.728 | 7 | 2538 | 132.0 | 237.1 | gemma4:31b-it-q4_K_M | 0.577 | gpt-5.5 | 0.597 | -0.02 |
| mellon_bes_mii_2024 | Policy-topic coding | High | 10.811 | 50 | 3999 | 33.0 | 46.0 | gemma4:26b | 0.528 | gpt-5.5 | 0.547 | -0.019 |
| muller_fujimura_campaign_policy_area | Policy-topic coding | Medium | 7.379 | 12 | 3283 | 48.0 | 72.0 | gemma4:31b-it-q4_K_M | 0.628 | gpt-5.5 | 0.636 | -0.008 |
| osnabruegge_cross_domain_topic | Policy-topic coding | Medium | 6.455 | 8 | 2603 | 367.5 | 3052.1 | gemma4:31b-it-q4_K_M | 0.435 | claude-sonnet-4-6 | 0.476 | -0.04 |
| ballard_incivility | Relevance / Incivility | Low | 1.449 | 1 | 1747 | 147.0 | 252.2 | gemma4:31b-it-q4_K_M | 0.522 | claude-sonnet-4-6 | 0.563 | -0.041 |
| brandt_political_relevance | Relevance / Incivility | Low | 1.567 | 1 | 624 | 1917.0 | 3174.0 | qwen3:14b-q4_K_M | 0.739 | claude-sonnet-4-6 | 0.726 | 0.013 |
| gilardi_relevance | Relevance / Incivility | Medium | 1.991 | 1 | 2244 | 242.0 | 304.1 | gemma4:31b-it-q4_K_M | 0.915 | gpt-5.5 | 0.95 | -0.035 |
| rheault_line_of_fire_incivility | Relevance / Incivility | Low | 1.426 | 1 | 745 | 139.0 | 172.0 | gemma4:26b | 0.588 | deepseek-v4-pro | 0.607 | -0.019 |
| theocharis_dynamics_incivility | Relevance / Incivility | Low | 1.813 | 1 | 803 | 133.0 | 148.0 | gemma4:31b-it-q4_K_M | 0.657 | gpt-5.5 | 0.682 | -0.025 |
| toxicity_protests_es | Relevance / Incivility | Low | 2.0 | 1 | 662 | 145.0 | 269.1 | gemma4:26b | 0.894 | claude-sonnet-4-6 | 0.888 | 0.006 |
| twitcivility_impoliteness | Relevance / Incivility | Low | 1.731 | 1 | 637 | 185.5 | 290.0 | gemma4:31b-it-q4_K_M | 0.631 | gpt-5.4-nano | 0.622 | 0.009 |
| agoraspeech_criticism_agenda | Rhetoric / Discourse Function | Low | 1.968 | 2 | 622 | 771.0 | 1715.1 | mistral-small:24b-instruct-2501-q4_K_M | 0.922 | deepseek-v4-pro | 0.914 | 0.007 |
| dicocco_manifesto_populism | Rhetoric / Populism | Low | 1.328 | 1 | 693 | 185.0 | 702.0 | mistral-small:24b-instruct-2501-q4_K_M | 0.455 | claude-sonnet-4-6 | 0.361 | 0.094 |
| bestvater_kavanaugh_stance | Sentiment / Stance / Tone | Low | 1.993 | 1 | 690 | 243.0 | 307.0 | gemma4:31b-it-q4_K_M | 0.932 | gpt-5.5 | 0.954 | -0.023 |
| bestvater_wm_stance | Sentiment / Stance / Tone | Low | 1.531 | 1 | 703 | 128.5 | 152.0 | gemma4:31b-it-q4_K_M | 0.859 | claude-sonnet-4-6 | 0.905 | -0.046 |
| burnham_covid_threat_minimization | Sentiment / Stance / Tone | Low | 1.779 | 1 | 663 | 201.0 | 290.0 | mistral-small:24b-instruct-2501-q4_K_M | 0.602 | gpt-5.4-nano | 0.538 | 0.064 |
| burnham_trump_stance | Sentiment / Stance / Tone | Low | 2.884 | 3 | 672 | 142.0 | 251.1 | gemma4:31b-it-q4_K_M | 0.768 | claude-sonnet-4-6 | 0.804 | -0.035 |
| chae_semeval_stance | Sentiment / Stance / Tone | Low | 2.785 | 3 | 1268 | 133.0 | 150.0 | gemma4:31b-it-q4_K_M | 0.772 | gpt-5.5 | 0.773 | -0.0 |
| gilardi_stance | Sentiment / Stance / Tone | Medium | 2.351 | 3 | 2627 | 220.5 | 302.0 | gemma4:31b-it-q4_K_M | 0.529 | claude-sonnet-4-6 | 0.568 | -0.039 |
| ornstein_scotus_sentiment | Sentiment / Stance / Tone | Low | 2.462 | 3 | 1191 | 176.5 | 298.0 | qwen3:14b-q4_K_M | 0.665 | deepseek-v4-pro | 0.668 | -0.003 |
| wesleyan_creative_ads_2022 | Sentiment / Stance / Tone | Low | 1.999 | 3 | 1604 | 1381.0 | 3049.1 | gemma4:26b | 0.698 | gpt-5.4-nano | 0.711 | -0.013 |
