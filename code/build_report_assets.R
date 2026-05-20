#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(knitr)
  library(ggrepel)
  library(scales)
  library(haschaR)
  library(kableExtra)
})

file_arg <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", file_arg[grepl("^--file=", file_arg)][1])
if (is.na(script_file)) {
  script_file <- "code/build_report_assets.R"
}
repo <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
setwd(repo)

dir.create("output/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("output/tables", showWarnings = FALSE, recursive = TRUE)

write_latex <- function(table, path) {
  writeLines(as.character(table), path, useBytes = TRUE)
}

summary_path <- "output/summary.csv"
summary_raw <- read_csv(summary_path, show_col_types = FALSE)

summary_all_raw <- summary_raw
total_task_count <- n_distinct(summary_all_raw$task)
active_task_count <- total_task_count

sb_raw <- read_csv("output/summary_batched.csv", show_col_types = FALSE)
sb_local_b10_raw <- read_csv("output/summary_batched_local_b10.csv", show_col_types = FALSE)

task_length_raw <- if (file.exists("output/task_length_audit.csv")) {
  read_csv("output/task_length_audit.csv", show_col_types = FALSE)
} else {
  tibble()
}

if (nrow(task_length_raw) > 0 && !"complexity_proxy_z" %in% names(task_length_raw)) {
  task_length_raw <- task_length_raw |>
    mutate(
      complexity_proxy_z = rowMeans(
        cbind(
          as.numeric(scale(log1p(prompt_chars))),
          as.numeric(scale(log1p(item_chars_median))),
          as.numeric(scale(log1p(label_count)))
        ),
        na.rm = TRUE
      )
    )
}

family_map <- tibble::tribble(
  ~task,                          ~family,
  "gilardi_relevance",            "Relevance & Harm",
  "brandt_political_relevance",   "Relevance & Harm",
  "burnham_polnli_entailment",    "Claims & Relations",
  "ballard_incivility",           "Relevance & Harm",
  "rheault_line_of_fire_incivility","Relevance & Harm",
  "theocharis_dynamics_incivility","Relevance & Harm",
  "gilardi_stance",               "Position & Tone",
  "chae_semeval_stance",          "Position & Tone",
  "ornstein_scotus_sentiment",    "Position & Tone",
  "wesleyan_creative_ads_2022",   "Position & Tone",
  "halterman_ccc_protest",        "Events & Actions",
  "halterman_keith_bfrs",         "Events & Actions",
  "haunss_papea_fgz_forms",       "Events & Actions",
  "douglass_icbe_sentence_event_type","Events & Actions",
  "halterman_keith_cmp",          "Issues & Topics",
  "mellon_bes_mii_2024",          "Issues & Topics",
  "osnabruegge_cross_domain_topic","Issues & Topics",
  "muller_fujimura_campaign_policy_area","Issues & Topics",
  "toxicity_protests_es",         "Relevance & Harm",
  "twitcivility_impoliteness",    "Relevance & Harm",
  "brandt_gtd_attack_type",       "Events & Actions",
  "haunss_papea_claims",          "Issues & Topics",
  "plover_cameo_event",           "Events & Actions",
  "burnham_polnli_event_entailment","Claims & Relations",
  "bestvater_wm_stance",          "Position & Tone",
  "burnham_trump_stance",         "Position & Tone",
  "burnham_covid_threat_minimization","Claims & Relations",
  "bestvater_kavanaugh_stance",   "Position & Tone",
  "erlich_ati_topics",            "Issues & Topics",
  "cap_party_platform_policy_topic","Issues & Topics",
  "cap_crs_policy_topic",         "Issues & Topics",
  "politicause_causal_relation",  "Claims & Relations",
  "dicocco_manifesto_populism",   "Claims & Relations",
  "agoraspeech_criticism_agenda", "Claims & Relations"
)

long_codebook_tasks <- c(
  "halterman_ccc_protest",
  "halterman_keith_bfrs",
  "haunss_papea_fgz_forms",
  "haunss_papea_claims",
  "brandt_gtd_attack_type",
  "plover_cameo_event",
  "halterman_keith_cmp",
  "mellon_bes_mii_2024",
  "osnabruegge_cross_domain_topic",
  "muller_fujimura_campaign_policy_area",
  "erlich_ati_topics",
  "cap_party_platform_policy_topic",
  "cap_crs_policy_topic"
)

model_levels <- c("Gemma 4 26B","Gemma 4 31B","Qwen3 14B","Qwen3 30B-A3B","Mistral Sm 24B",
                  "gpt-5.5","gpt-5.4-nano","DeepSeek V4 Pro","Claude Sonnet 4.6")

model_palette <- c(
  "Gemma 4 26B"       = "#009E73",
  "Gemma 4 31B"       = "#0072B2",
  "Qwen3 14B"         = "#d95f02",
  "Qwen3 30B-A3B"     = "#7570b3",
  "Mistral Sm 24B"    = "#e7298a",
  "gpt-5.5"           = "#1f1f1f",
  "gpt-5.4-nano"      = "#8a8a8a",
  "DeepSeek V4 Pro"   = "#555555",
  "Claude Sonnet 4.6" = "#a8a8a8"
)

label_models <- function(data) {
  data |>
    mutate(
      model_short = recode(model,
        "gemma4:31b-it-q4_K_M"                    = "Gemma 4 31B",
        "gemma4:26b"                               = "Gemma 4 26B",
        "qwen3:14b-q4_K_M"                         = "Qwen3 14B",
        "qwen3:30b-a3b-q4_K_M"                     = "Qwen3 30B-A3B",
        "mistral-small:24b-instruct-2501-q4_K_M"   = "Mistral Sm 24B",
        "gpt-5.5"                                  = "gpt-5.5",
        "gpt-5.4-nano"                             = "gpt-5.4-nano",
        "deepseek-v4-pro"                          = "DeepSeek V4 Pro",
        "claude-sonnet-4-6"                        = "Claude Sonnet 4.6"
      ),
      model_short = factor(model_short, levels = model_levels),
      is_api = model %in% c("gpt-5.5", "gpt-5.4-nano", "claude-sonnet-4-6", "deepseek-v4-pro")
    )
}

label_tasks <- function(data, expanded = TRUE) {
  data |>
    mutate(
      task_short = recode(task,
        "gilardi_relevance"          = "Gilardi relevance",
        "brandt_political_relevance" = "Brandt relevance",
        "burnham_polnli_entailment"  = "Burnham PolNLI",
        "ballard_incivility"         = "Ballard incivility",
        "rheault_line_of_fire_incivility" = "Line of Fire incivility",
        "theocharis_dynamics_incivility" = "Theocharis incivility",
        "gilardi_stance"             = "Gilardi stance",
        "ornstein_scotus_sentiment"  = "SCOTUS sentiment",
        "halterman_ccc_protest"      = if (expanded) "CCC protest type" else "CCC protest",
        "haunss_papea_fgz_forms"     = if (expanded) "PAPEA protest forms (7 classes)" else "PAPEA protest forms",
        "douglass_icbe_sentence_event_type" = if (expanded) "ICBe event type (5 classes)" else "ICBe event type",
        "chae_semeval_stance"        = if (expanded) "SemEval 2016 stance" else "SemEval stance",
        "halterman_keith_bfrs"       = if (expanded) "BFRS violence (12 classes)" else "BFRS Pakistan",
        "halterman_keith_cmp"        = if (expanded) "CMP manifestos (7 classes)" else "CMP manifestos",
        "mellon_bes_mii_2024"        = if (expanded) "BES MII (50 classes)" else "BES MII",
        "muller_fujimura_campaign_policy_area" = if (expanded) "Campaign policy area (12 classes)" else "Campaign policy area",
        "osnabruegge_cross_domain_topic" = if (expanded) "Cross-domain topics (8 classes)" else "Cross-domain topics",
        "wesleyan_creative_ads_2022" = if (expanded) "Wesleyan ad tone" else "Wesleyan ads",
        "toxicity_protests_es"       = "Spanish protest toxicity",
        "brandt_gtd_attack_type"     = "GTD attack type (9 classes)",
        "haunss_papea_claims"        = "PAPEA protest claims (28 classes)",
        "twitcivility_impoliteness"  = "TwitCivility impoliteness",
        "bestvater_wm_stance"        = "Women's March stance",
        "erlich_ati_topics"          = "ATI topics (7 labels)",
        "plover_cameo_event"         = "PLOVER CAMEO event (18 classes)",
        "burnham_polnli_event_entailment" = "PolNLI event entailment",
        "burnham_trump_stance"       = "Trump stance",
        "burnham_covid_threat_minimization" = "COVID threat minimization",
        "dicocco_manifesto_populism" = "Manifesto populism",
        "bestvater_kavanaugh_stance" = "Kavanaugh stance",
        "politicause_causal_relation" = "PolitiCAUSE causal relation",
        "cap_party_platform_policy_topic" = "CAP party platforms (21 classes)",
        "cap_crs_policy_topic"       = "CAP CRS reports (21 classes)",
        "agoraspeech_criticism_agenda" = "AgoraSpeech criticism/agenda"
      )
    )
}

df <- summary_all_raw |>
  left_join(family_map, by = "task") |>
  label_models() |>
  label_tasks(expanded = TRUE)

if (any(is.na(df$family))) {
  stop("Unmapped report task type: ", paste(sort(unique(df$task[is.na(df$family)])), collapse = ", "))
}

sb <- sb_raw |>
  label_models() |>
  label_tasks(expanded = FALSE)

sb_local_b10 <- sb_local_b10_raw |>
  label_models()

b1_latency <- sb |> filter(batch_size == 1) |>
  select(task, model, b1_latency = median_latency_s)
b1_f1 <- sb |> filter(batch_size == 1) |>
  select(task, model, b1_headline_f1 = headline_f1)
sb <- sb |>
  left_join(b1_latency, by = c("task", "model")) |>
  left_join(b1_f1, by = c("task", "model")) |>
  mutate(
    speedup = b1_latency / median_latency_s,
    delta_f1 = headline_f1 - b1_headline_f1
  )

batch_local_df <- sb_local_b10 |>
  filter(!is_api, batch_size == 10) |>
  transmute(
    task, model,
    b10_latency_s = median_latency_s,
    b10_headline_f1 = headline_f1,
    b10_parse_err_rate = parse_err_rate
  ) |>
  inner_join(
    df |>
      filter(!is_api) |>
      transmute(
        task, model, model_short, task_short,
        serial_latency_s = median_latency_s,
        serial_headline_f1 = headline_f1,
        serial_parse_err_rate = parse_err_rate
      ),
    by = c("task", "model")
  ) |>
  mutate(
    speedup = serial_latency_s / b10_latency_s,
    delta_f1 = b10_headline_f1 - serial_headline_f1,
    prompt_type = factor(
      if_else(task %in% long_codebook_tasks, "Long codebook", "Short prompt"),
      levels = c("Short prompt", "Long codebook")
    )
  )

# Figure 1: mean F1
agg_f1 <- df |>
  group_by(model_short) |>
  summarise(n_tasks = n_distinct(task),
            mean_f1 = if_else(n_tasks == total_task_count,
                              mean(headline_f1, na.rm = TRUE),
                              NA_real_),
            .groups = "drop") |>
  arrange(is.na(mean_f1), desc(mean_f1), desc(n_tasks)) |>
  mutate(model_short = factor(model_short, levels = model_short))

per_task <- df |>
  select(model_short, headline_f1) |>
  mutate(model_short = factor(model_short, levels = levels(agg_f1$model_short)))

p_mean_f1 <- ggplot() +
  geom_jitter(data = per_task,
              aes(x = model_short, y = headline_f1),
              width = 0.22, height = 0, size = 1.45, alpha = 0.18,
              color = "grey78") +
  geom_point(data = agg_f1 |> filter(!is.na(mean_f1)),
             aes(x = model_short, y = mean_f1, fill = model_short),
             shape = 21, size = 5, stroke = 0.6, color = "grey20") +
  geom_text(data = agg_f1 |> filter(!is.na(mean_f1)),
            aes(x = model_short, y = mean_f1,
                label = sprintf("%.3f", mean_f1)),
            vjust = -1.5, size = 2.8, fontface = "bold") +
  scale_y_continuous(limits = c(0.20, 1.02), breaks = seq(0.2, 1, 0.2)) +
  scale_fill_manual(values = model_palette, guide = "none") +
  labs(x = NULL, y = "Mean F1 across tasks") +
  theme_hanno(fontsize = 10.5) +
  theme(panel.grid.major.x = element_blank(),
        axis.text.x = element_text(angle = 25, hjust = 1),
        legend.position = "none",
        plot.margin = margin(5.5, 12, 12, 24))

ggsave("output/figures/fig-mean-f1.png", plot = p_mean_f1,
       width = 6.0, height = 3.45, dpi = 300, bg = "white")

# Figure 2: family/type means
fam_levels <- c(
  "Relevance & Harm\n\"Is it relevant or harmful?\"",
  "Position & Tone\n\"What position or tone does it express?\"",
  "Events & Actions\n\"What action or event is described?\"",
  "Claims & Relations\n\"What claim or relationship is asserted?\"",
  "Issues & Topics\n\"What issue is this about?\""
)
family_label <- function(f) {
  recode(f,
    "Relevance & Harm" = fam_levels[1],
    "Position & Tone" = fam_levels[2],
    "Events & Actions" = fam_levels[3],
    "Claims & Relations" = fam_levels[4],
    "Issues & Topics" = fam_levels[5]
  )
}
fam_df <- df |>
  group_by(family, model_short) |>
  summarise(family_mean_f1 = mean(headline_f1, na.rm = TRUE), .groups = "drop") |>
  mutate(family = factor(family_label(family), levels = fam_levels))

p_family <- ggplot(fam_df, aes(x = fct_rev(model_short), y = family_mean_f1, fill = model_short)) +
  geom_col(width = 0.75) +
  geom_text(aes(label = sprintf("%.2f", family_mean_f1)),
            hjust = -0.15, size = 2.2, color = "grey20") +
  coord_flip() +
  facet_wrap(~ family, ncol = 2) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.25),
                     expand = expansion(mult = c(0, 0.05))) +
  scale_fill_manual(values = model_palette) +
  labs(x = NULL, y = "Mean F1 within type") +
  theme_hanno(fontsize = 10.5) +
  theme(legend.position = "none",
        panel.grid.major.y = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        strip.background = element_rect(fill = "grey94", color = "grey45", linewidth = 0.35),
        strip.text = element_text(face = "bold", size = 7.4))

ggsave("output/figures/fig-family.png", plot = p_family,
       width = 6.35, height = 5.65, dpi = 300, bg = "white")

# Figure 3: local speed and performance
serial_speed_df <- df |>
  filter(!is_api) |>
  group_by(model_short) |>
  summarise(mean_f1 = mean(headline_f1, na.rm = TRUE),
            median_latency = median(median_latency_s, na.rm = TRUE),
            n_tasks = n_distinct(task),
            .groups = "drop") |>
  mutate(call_mode = "One-at-a-time")

batched_speed_df <- sb_local_b10 |>
  filter(!is_api, batch_size == 10) |>
  group_by(model_short) |>
  summarise(mean_f1 = mean(headline_f1, na.rm = TRUE),
            median_latency = median(median_latency_s, na.rm = TRUE),
            n_tasks = n_distinct(task),
            .groups = "drop") |>
  filter(n_tasks == active_task_count) |>
  mutate(call_mode = "10 items per prompt")

speed_df <- bind_rows(serial_speed_df, batched_speed_df) |>
  mutate(call_mode = factor(call_mode, levels = c("One-at-a-time", "10 items per prompt")))

speed_label_df <- speed_df |>
  group_by(model_short) |>
  summarise(
    label_x = exp(mean(log(median_latency), na.rm = TRUE)),
    label_y = mean(mean_f1, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    label_x = case_when(
      model_short == "Gemma 4 26B" ~ 0.58,
      model_short == "Gemma 4 31B" ~ 3.10,
      model_short == "Qwen3 14B" ~ 0.94,
      model_short == "Qwen3 30B-A3B" ~ 0.41,
      model_short == "Mistral Sm 24B" ~ 1.62,
      TRUE ~ label_x
    ),
    label_y = case_when(
      model_short == "Gemma 4 26B" ~ 0.633,
      model_short == "Gemma 4 31B" ~ 0.642,
      model_short == "Qwen3 14B" ~ 0.612,
      model_short == "Qwen3 30B-A3B" ~ 0.579,
      model_short == "Mistral Sm 24B" ~ 0.622,
      TRUE ~ label_y
    )
  )

p_speed <- ggplot(speed_df, aes(x = median_latency, y = mean_f1, color = model_short)) +
  geom_line(aes(group = model_short), linewidth = 0.45, alpha = 0.55, show.legend = FALSE) +
  geom_point(aes(shape = call_mode), size = 4, stroke = 1.2) +
  geom_text(data = speed_label_df,
                  aes(x = label_x, y = label_y, label = model_short, color = model_short),
                  inherit.aes = FALSE, size = 2.8, show.legend = FALSE) +
  scale_x_log10(breaks = c(0.2, 0.5, 1, 2, 5, 10),
                labels = function(x) paste0(x, "s"),
                expand = expansion(mult = c(0.16, 0.08))) +
  scale_color_manual(values = model_palette, guide = "none") +
  scale_shape_manual(values = c("One-at-a-time" = 16, "10 items per prompt" = 17),
                     name = NULL) +
  labs(x = "Median seconds per item (log scale)",
       y = "Mean F1") +
  theme_hanno(fontsize = 11.5) +
  theme(legend.position = "bottom",
        panel.grid.minor = element_blank(),
        plot.margin = margin(5.5, 12, 5.5, 12))

ggsave("output/figures/fig-speed.png", plot = p_speed,
       width = 5.55, height = 3.15, dpi = 300, bg = "white")

# Appendix figure: local runtime per 1,000 items
runtime_single <- df |>
  filter(!is_api) |>
  group_by(model_short) |>
  summarise(
    minutes_per_1000 = median(median_latency_s * 1000 / 60, na.rm = TRUE),
    n_tasks = n_distinct(task),
    .groups = "drop"
  ) |>
  mutate(call_mode = "One-at-a-time")

runtime_b10 <- sb_local_b10 |>
  filter(!is_api, batch_size == 10) |>
  group_by(model_short) |>
  summarise(
    minutes_per_1000 = median(median_latency_s * 1000 / 60, na.rm = TRUE),
    n_tasks = n_distinct(task),
    .groups = "drop"
  ) |>
  filter(n_tasks == active_task_count) |>
  mutate(call_mode = "10 items per prompt")

runtime_order <- runtime_single |>
  arrange(minutes_per_1000) |>
  pull(model_short)

runtime_per_1000 <- bind_rows(runtime_single, runtime_b10) |>
  mutate(
    model_short = factor(model_short, levels = rev(runtime_order)),
    call_mode = factor(call_mode, levels = c("One-at-a-time", "10 items per prompt"))
  )

write_csv(runtime_per_1000, "output/local_runtime_minutes_per_1000.csv")

p_runtime <- ggplot(
  runtime_per_1000,
  aes(x = minutes_per_1000, y = model_short, fill = call_mode)
) +
  geom_col(width = 0.68, position = position_dodge2(width = 0.78, preserve = "single")) +
  geom_text(aes(label = sprintf("%.1f", minutes_per_1000)),
            hjust = -0.12, size = 2.8,
            position = position_dodge2(width = 0.78, preserve = "single")) +
  scale_x_continuous(limits = c(0, max(runtime_per_1000$minutes_per_1000, na.rm = TRUE) * 1.18),
                     breaks = seq(0, 60, 15),
                     expand = expansion(mult = c(0, 0.01))) +
  scale_fill_manual(values = c("One-at-a-time" = "grey50",
                               "10 items per prompt" = "#009E73"),
                    name = NULL) +
  labs(x = "Median minutes per 1,000 items",
       y = NULL) +
  theme_hanno(fontsize = 10.5) +
  theme(legend.position = "bottom",
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank())

ggsave("output/figures/fig-local-runtime-per-1000.png", plot = p_runtime,
       width = 6.4, height = 3.7, dpi = 300, bg = "white")

# Appendix figure: complexity
task_complexity <- task_length_raw |>
  select(task, coding_complexity, effective_label_count) |>
  mutate(coding_complexity = factor(coding_complexity, levels = c("Low", "Medium", "High")))

complexity_cells <- df |>
  left_join(task_complexity, by = "task") |>
  mutate(compute_class = if_else(is_api, "API", "Local")) |>
  filter(!is.na(coding_complexity), !is.na(headline_f1))

complexity_means <- complexity_cells |>
  group_by(coding_complexity, compute_class) |>
  summarise(mean_f1 = mean(headline_f1, na.rm = TRUE), n = n(), .groups = "drop") |>
  mutate(panel = "All model-task results")

best_by_task <- complexity_cells |>
  group_by(task, coding_complexity, compute_class) |>
  slice_max(headline_f1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(panel = "Best API and best local model per task")

best_means <- best_by_task |>
  group_by(coding_complexity, compute_class, panel) |>
  summarise(mean_f1 = mean(headline_f1, na.rm = TRUE), n = n(), .groups = "drop")

complexity_plot_means <- bind_rows(complexity_means, best_means) |>
  mutate(
    panel = factor(panel, levels = c("All model-task results", "Best API and best local model per task")),
    compute_class = factor(compute_class, levels = c("API", "Local"))
  )

complexity_palette <- c("Local" = "#009E73", "API" = "#333333")

p_complexity <- ggplot() +
  geom_line(data = complexity_plot_means,
            aes(x = coding_complexity, y = mean_f1,
                color = compute_class, group = compute_class),
            linewidth = 0.7,
            position = position_dodge(width = 0.42)) +
  geom_point(data = complexity_plot_means,
             aes(x = coding_complexity, y = mean_f1, fill = compute_class),
             shape = 21, size = 4.4, color = "white", stroke = 0.6,
             position = position_dodge(width = 0.42)) +
  geom_text(data = complexity_plot_means,
            aes(x = coding_complexity, y = mean_f1,
                group = compute_class, label = sprintf("%.2f", mean_f1)),
            size = 2.5, vjust = -1.0,
            position = position_dodge(width = 0.42)) +
  facet_wrap(~ panel, nrow = 1) +
  scale_y_continuous(limits = c(0.15, 1.02), breaks = seq(0.2, 1, 0.2)) +
  scale_color_manual(values = complexity_palette, name = NULL) +
  scale_fill_manual(values = complexity_palette, guide = "none") +
  labs(x = "Coding complexity", y = "F1") +
  theme_hanno(fontsize = 10) +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold", size = 8.5),
        panel.grid.minor = element_blank())

ggsave("output/figures/fig-complexity.png", plot = p_complexity,
       width = 6.2, height = 3.45, dpi = 300, bg = "white")

# Appendix figures: additional heterogeneity checks
best_local_api <- df |>
  mutate(compute_class = if_else(is_api, "API", "Local")) |>
  filter(!is.na(headline_f1)) |>
  group_by(task, task_short, family, compute_class) |>
  slice_max(headline_f1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(task, task_short, family, compute_class, model_short, headline_f1) |>
  pivot_wider(
    names_from = compute_class,
    values_from = c(model_short, headline_f1),
    names_sep = "_"
  ) |>
  mutate(
    local_minus_api_best_f1 = headline_f1_Local - headline_f1_API,
    api_minus_local_best_f1 = -local_minus_api_best_f1,
    advantage = case_when(
      api_minus_local_best_f1 < 0 ~ "Local ahead",
      api_minus_local_best_f1 > 0 ~ "API ahead",
      TRUE ~ "Tie"
    ),
    task_short = as.character(task_short),
    family = factor(family, levels = c("Relevance & Harm", "Position & Tone",
                                       "Events & Actions", "Claims & Relations",
                                       "Issues & Topics"))
  ) |>
  arrange(api_minus_local_best_f1)

write_csv(best_local_api, "output/best_local_api_task_gap.csv")

gap_plot_df <- best_local_api |>
  mutate(
    task_label = fct_reorder(task_short, api_minus_local_best_f1),
    advantage = if_else(api_minus_local_best_f1 < 0, "Local ahead", "API ahead"),
    advantage = factor(advantage, levels = c("Local ahead", "API ahead"))
  )

p_best_local_api <- ggplot(gap_plot_df, aes(y = task_label)) +
  geom_vline(xintercept = 0, color = "grey45", linewidth = 0.45, linetype = "dashed") +
  geom_segment(aes(x = 0, xend = api_minus_local_best_f1,
                   yend = task_label, color = advantage),
               linewidth = 0.45) +
  geom_point(aes(x = api_minus_local_best_f1, fill = advantage),
             shape = 21, size = 2.1, stroke = 0.35, color = "white") +
  scale_x_continuous(labels = label_number(accuracy = 0.01),
                     breaks = seq(-0.12, 0.16, 0.04),
                     expand = expansion(mult = c(0.03, 0.03))) +
  scale_color_manual(values = c("Local ahead" = "#009E73",
                                "API ahead" = "#333333"),
                     guide = "none") +
  scale_fill_manual(values = c("Local ahead" = "#009E73",
                               "API ahead" = "#333333"),
                    name = NULL) +
  labs(x = "API advantage in main F1",
       y = NULL) +
  theme_hanno(fontsize = 10) +
  theme(legend.position = "bottom",
        axis.text.y = element_text(size = 6.6),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank())

ggsave("output/figures/fig-best-local-api-gap.png", plot = p_best_local_api,
       width = 5.8, height = 4.75, dpi = 300, bg = "white")

single_model_means <- df |>
  filter(!is.na(headline_f1)) |>
  group_by(model_short, is_api) |>
  summarise(mean_f1 = mean(headline_f1, na.rm = TRUE),
            n_tasks = n_distinct(task),
            .groups = "drop") |>
  filter(n_tasks == total_task_count)

model_selection_value <- bind_rows(
  single_model_means |>
    slice_max(mean_f1, n = 1, with_ties = FALSE) |>
    transmute(selection_rule = "Best single model",
              mean_f1,
              detail = as.character(model_short)),
  single_model_means |>
    filter(!is_api) |>
    slice_max(mean_f1, n = 1, with_ties = FALSE) |>
    transmute(selection_rule = "Best single local model",
              mean_f1,
              detail = as.character(model_short)),
  single_model_means |>
    filter(is_api) |>
    slice_max(mean_f1, n = 1, with_ties = FALSE) |>
    transmute(selection_rule = "Best single API model",
              mean_f1,
              detail = as.character(model_short)),
  df |>
    filter(!is_api) |>
    group_by(task) |>
    summarise(mean_f1 = max(headline_f1, na.rm = TRUE), .groups = "drop") |>
    summarise(selection_rule = "Oracle task-specific local choice",
              mean_f1 = mean(mean_f1),
              detail = "Best local per task"),
  df |>
    filter(is_api) |>
    group_by(task) |>
    summarise(mean_f1 = max(headline_f1, na.rm = TRUE), .groups = "drop") |>
    summarise(selection_rule = "Oracle task-specific API choice",
              mean_f1 = mean(mean_f1),
              detail = "Best API per task"),
  df |>
    group_by(task) |>
    summarise(mean_f1 = max(headline_f1, na.rm = TRUE), .groups = "drop") |>
    summarise(selection_rule = "Oracle task-specific all-model choice",
              mean_f1 = mean(mean_f1),
              detail = "Best model per task")
) |>
  mutate(
    selection_rule = factor(
      selection_rule,
      levels = c("Best single model",
                 "Best single local model",
                 "Best single API model",
                 "Oracle task-specific local choice",
                 "Oracle task-specific API choice",
                 "Oracle task-specific all-model choice")
    ),
    rule_type = if_else(str_starts(as.character(selection_rule), "Oracle"),
                        "Task-specific upper bound",
                        "Single model")
  )

write_csv(model_selection_value, "output/model_selection_value.csv")

model_selection_print <- model_selection_value |>
  mutate(
    `Selection rule` = recode(as.character(selection_rule),
      "Best single model" = "Best single model",
      "Best single local model" = "Best single local model",
      "Best single API model" = "Best single API model",
      "Oracle task-specific local choice" = "Task-specific upper bound: local",
      "Oracle task-specific API choice" = "Task-specific upper bound: API",
      "Oracle task-specific all-model choice" = "Task-specific upper bound: all models"
    )
  ) |>
  transmute(
    `Selection rule`,
    `Model or rule` = detail,
    `Mean F1` = sprintf("%.3f", mean_f1)
  )

write_latex(
  kable(model_selection_print, format = "latex", align = "llr",
        booktabs = TRUE,
        caption = "Mean F1 under single-model choices and task-specific upper bounds. The task-specific rows choose the best model after observing task-level performance, so they describe an upper bound rather than expected performance from a validation sample. \\label{tab:model-selection-value}") |>
    kable_styling(font_size = 9, latex_options = c("HOLD_position")),
  "output/tables/table-model-selection-value.tex"
)

label_gap_df <- best_local_api |>
  left_join(task_length_raw |>
              select(task, label_kind, label_count, effective_label_count),
            by = "task") |>
  mutate(
    label_structure = case_when(
      label_kind == "binary" | label_count <= 2 ~ "Binary / 2-class",
      label_count == 3 ~ "3-class",
      TRUE ~ "Many-class / multi-label"
    ),
    label_structure = factor(label_structure,
                             levels = c("Binary / 2-class",
                                        "3-class",
                                        "Many-class / multi-label"))
  ) |>
  filter(!is.na(effective_label_count), !is.na(api_minus_local_best_f1))

write_csv(label_gap_df, "output/local_api_gap_by_label_structure.csv")

p_label_gap <- ggplot(
  label_gap_df,
  aes(x = effective_label_count,
      y = api_minus_local_best_f1,
      shape = label_structure)
) +
  geom_hline(yintercept = 0, color = "grey45", linewidth = 0.4, linetype = "dashed") +
  geom_point(size = 2.8, alpha = 0.92, color = "grey25") +
  geom_smooth(data = label_gap_df,
              aes(x = effective_label_count,
                  y = api_minus_local_best_f1),
              inherit.aes = FALSE,
              method = "lm", se = FALSE,
              color = "grey30", linewidth = 0.55, linetype = "solid") +
  scale_x_log10(breaks = c(1, 2, 3, 5, 10, 20),
                labels = label_number(accuracy = 1)) +
  scale_y_continuous(labels = label_number(accuracy = 0.01),
                     breaks = seq(-0.12, 0.14, 0.04)) +
  scale_shape_manual(values = c("Binary / 2-class" = 16,
                                "3-class" = 17,
                                "Many-class / multi-label" = 15),
                     name = NULL) +
  labs(x = "Effective number of labels (log scale)",
       y = "Best API minus best local F1") +
  theme_hanno(fontsize = 10.5) +
  theme(legend.position = "bottom",
        panel.grid.minor = element_blank(),
        plot.margin = margin(8, 10, 8, 24))

ggsave("output/figures/fig-label-structure-gap.png", plot = p_label_gap,
       width = 5.95, height = 3.65, dpi = 300, bg = "white")

# Table: paired task-level bootstrap for cross-task mean differences
set.seed(20260514)
task_bootstrap_iterations <- 10000L
top_model_ids <- c(
  "claude-sonnet-4-6",
  "gpt-5.5",
  "gemma4:31b-it-q4_K_M",
  "deepseek-v4-pro"
)

top_model_labels <- df |>
  filter(model %in% top_model_ids) |>
  distinct(model, model_short) |>
  mutate(model_short = as.character(model_short))

task_mean_wide <- df |>
  filter(model %in% top_model_ids) |>
  select(task, model, headline_f1) |>
  pivot_wider(names_from = model, values_from = headline_f1)

if (nrow(task_mean_wide) != total_task_count ||
    any(!top_model_ids %in% names(task_mean_wide))) {
  stop("Task-level bootstrap input is incomplete.")
}

task_bootstrap <- combn(top_model_ids, 2, simplify = FALSE) |>
  map_dfr(function(pair) {
    per_task_diff <- task_mean_wide[[pair[1]]] - task_mean_wide[[pair[2]]]
    boot_diff <- replicate(
      task_bootstrap_iterations,
      mean(sample(per_task_diff, length(per_task_diff), replace = TRUE),
           na.rm = TRUE)
    )
    label_a <- top_model_labels$model_short[match(pair[1], top_model_labels$model)]
    label_b <- top_model_labels$model_short[match(pair[2], top_model_labels$model)]
    tibble(
      `Model comparison` = paste(label_a, "-", label_b),
      `Mean difference` = mean(per_task_diff, na.rm = TRUE),
      `CI lower` = unname(quantile(boot_diff, 0.025, na.rm = TRUE)),
      `CI upper` = unname(quantile(boot_diff, 0.975, na.rm = TRUE)),
      `Tasks` = sum(!is.na(per_task_diff))
    )
  })

write_csv(task_bootstrap, "output/task_bootstrap_top_model_differences.csv")

task_bootstrap_print <- task_bootstrap |>
  transmute(
    `Model comparison`,
    `Mean difference` = sprintf("%.3f", `Mean difference`),
    `95\\% task-bootstrap interval` = sprintf("[%.3f, %.3f]", `CI lower`, `CI upper`),
    Tasks
  )

write_latex(
  kable(task_bootstrap_print, format = "latex", align = "lrrr",
        escape = FALSE, booktabs = TRUE,
        caption = "Paired task-level bootstrap intervals for cross-task mean-F1 differences among the four strongest models. Each bootstrap resamples the 34 tasks with replacement and recomputes the paired mean difference. These intervals assess sensitivity to benchmark composition; they do not replace the paired-by-item bootstrap intervals used for within-task comparisons. \\label{tab:task-bootstrap}") |>
    kable_styling(font_size = 9, latex_options = c("HOLD_position")),
  "output/tables/table-task-bootstrap-appendix.tex"
)

# Table: models
model_table <- tibble::tribble(
  ~Model, ~Class, ~`Exact ID/tag`, ~`Size/quant.`, ~`Access/runtime`,
  "Gemma 4 26B", "Local Ollama", "gemma4:26b", "26B, Ollama tag default", "Apple M2 Pro, 32 GB; Ollama 0.19.0; macOS 26.1",
  "Gemma 4 31B", "Local Ollama", "gemma4:31b-it-q4_K_M", "31B, Q4_K_M", "Apple M2 Pro, 32 GB; Ollama 0.19.0; macOS 26.1",
  "Qwen3 14B", "Local Ollama", "qwen3:14b-q4_K_M", "14B, Q4_K_M", "Apple M2 Pro, 32 GB; Ollama 0.19.0; macOS 26.1",
  "Qwen3 30B-A3B", "Local Ollama", "qwen3:30b-a3b-q4_K_M", "30B MoE, Q4_K_M", "Apple M2 Pro, 32 GB; Ollama 0.19.0; macOS 26.1",
  "Mistral Sm 24B", "Local Ollama", "mistral-small:24b-instruct-2501-q4_K_M", "24B, Q4_K_M", "Apple M2 Pro, 32 GB; Ollama 0.19.0; macOS 26.1",
  "gpt-5.5", "API", "gpt-5.5", "provider model", "OpenAI, May 2026",
  "gpt-5.4-nano", "API", "gpt-5.4-nano", "provider model", "OpenAI, May 2026",
  "DeepSeek V4 Pro", "API", "deepseek-v4-pro", "provider model", "DeepSeek, May 2026",
  "Claude Sonnet 4.6", "API", "claude-sonnet-4-6", "provider model", "Anthropic, May 2026"
)

write_latex(
  kable(model_table, format = "latex", align = "lllll", booktabs = TRUE,
        caption = "Models and runtime settings. The table reports exact local Ollama tags and API model IDs used in the benchmark. No context-window override was set; output caps and output constraints are described in the text. \\label{tab:models}") |>
    kable_styling(font_size = 8, latex_options = c("HOLD_position", "scale_down")) |>
    column_spec(3, width = "15em") |>
    column_spec(5, width = "16em"),
  "output/tables/table-models.tex"
)

# Table: tasks
tasks_table <- tibble::tribble(
  ~Task, ~Description, ~Source, ~Type, ~Labels, ~Provenance, ~`Prompt`,
  "Gilardi relevance", "Tweets, classified as about content moderation or not.", "Gilardi et al. 2023", "Relevance & Harm", "binary", "Verbatim", "short",
  "Brandt relevance", "News articles, classified as conflict- or politics-relevant or not.", "Brandt et al. 2025", "Relevance & Harm", "binary", "Derived", "short",
  "Burnham PolNLI", "Premise-hypothesis pairs, classified for entailment.", "Burnham et al. 2025", "Claims & Relations", "binary", "Derived", "short",
  "Ballard incivility", "Tweets by U.S. members of Congress, classified for incivility.", "Ballard 2022", "Relevance & Harm", "binary", "Derived", "short",
  "Line of Fire incivility", "Political tweets, classified for incivility.", "Rheault et al. 2019", "Relevance & Harm", "binary", "Derived", "short",
  "Theocharis incivility", "Political tweets, classified for incivility.", "Theocharis et al. 2020", "Relevance & Harm", "binary", "Derived", "short",
  "Gilardi stance", "Tweets about content moderation, classified by stance toward moderation.", "Gilardi et al. 2023", "Position & Tone", "3-class", "Verbatim", "short",
  "SemEval stance", "Tweets toward five named political targets, classified by stance.", "Chae & Davidson 2026", "Position & Tone", "3-class", "Derived", "short",
  "SCOTUS sentiment", "Tweets about U.S. Supreme Court rulings, classified by sentiment.", "Ornstein et al. 2025", "Position & Tone", "3-class", "Derived", "short",
  "Wesleyan ad tone", "U.S. Meta political ads, classified by tone toward a candidate.", "Zhang et al. 2025", "Position & Tone", "3-class", "Derived", "short",
  "CCC protest type", "News stories about U.S. protest events, classified by event type.", "Halterman & Keith 2026", "Events & Actions", "4-class", "Verbatim-adapted", "long codebook",
  "BFRS violence", "News stories about Pakistani political violence, classified by event type.", "Halterman & Keith 2026", "Events & Actions", "12-class", "Verbatim-adapted", "long codebook",
  "PAPEA protest forms", "Protest snippets, classified by one of seven common source protest-form labels.", "Haunss et al. 2025", "Events & Actions", "7-class", "Derived", "long codebook",
  "ICBe event type", "Crisis-event sentences, classified by sentence event type.", "Douglass et al. 2024", "Events & Actions", "5-class", "Derived", "short",
  "CMP manifestos", "Quasi-sentences from political party manifestos, classified by policy domain.", "Halterman & Keith 2026; Manifesto Project 2025", "Issues & Topics", "7-class", "Derived", "long codebook",
  "Cross-domain topics", "Political text, classified by broad policy domain.", "Osnabruegge et al. 2023", "Issues & Topics", "8-class", "Derived", "long codebook",
  "Campaign policy area", "Campaign statements, classified by policy area.", "Müller & Fujimura 2025", "Issues & Topics", "12-class", "Derived", "long codebook",
  "BES MII", "Open-ended British Election Study responses, classified by issue topic.", "Mellon et al. 2024", "Issues & Topics", "50-class", "Verbatim-adapted", "long codebook",
  "Spanish protest toxicity", "Spanish protest tweets, classified for toxic content.", "González-Bustamante 2024", "Relevance & Harm", "binary", "Derived", "short",
  "TwitCivility impoliteness", "Political tweets, classified for impoliteness.", "Pendzel et al. 2023", "Relevance & Harm", "binary", "Derived", "short",
  "GTD attack type", "Global Terrorism Database events, classified by primary attack type.", "Brandt et al. 2025", "Events & Actions", "9-class", "Derived", "long codebook",
  "PAPEA protest claims", "Protest snippets, classified by source protest-claim label.", "Haunss et al. 2025", "Issues & Topics", "28-class", "Derived", "long codebook",
  "PLOVER CAMEO event", "Gold-standard CAMEO examples, classified into PLOVER event categories.", "Halterman et al. 2023", "Events & Actions", "18-class", "Derived", "long codebook",
  "PolNLI event entailment", "Event-extraction PolNLI pairs, classified for entailment.", "Burnham et al. 2025", "Claims & Relations", "binary", "Derived", "short",
  "Women's March stance", "Tweets about the Women's March, classified by stance.", "Bestvater & Monroe 2023", "Position & Tone", "binary", "Derived", "short",
  "Trump stance", "Political statements, classified by stance toward Trump.", "Burnham 2025", "Position & Tone", "3-class", "Derived", "short",
  "COVID threat minimization", "Political text, classified for COVID threat minimization.", "Burnham 2025", "Claims & Relations", "binary", "Derived", "short",
  "Kavanaugh stance", "Tweets about Brett Kavanaugh, classified by stance.", "Bestvater & Monroe 2023", "Position & Tone", "binary", "Derived", "short",
  "ATI topics", "Mexican access-to-information requests, classified by request topic.", "Erlich et al. 2022", "Issues & Topics", "7-label", "Derived", "long codebook",
  "CAP party platforms", "Party-platform quasi-statements, classified by CAP major topic.", "Policy Agendas Project 2025", "Issues & Topics", "21-class", "Derived", "long codebook",
  "CAP CRS reports", "Congressional Research Service titles and summaries, classified by CAP major topic.", "Policy Agendas Project 2025; CRS Products", "Issues & Topics", "21-class", "Derived", "long codebook",
  "PolitiCAUSE causal relation", "Political sentences, classified for whether they express a causal relation.", "Garcia Corral et al. 2024", "Claims & Relations", "binary", "Derived", "short",
  "Manifesto populism", "Italian manifesto sentences, classified for populist rhetoric.", "Di Cocco & Monechi 2022", "Claims & Relations", "binary", "Derived", "short",
  "AgoraSpeech criticism/agenda", "Greek campaign-speech paragraphs, classified as criticism or agenda setting.", "Sermpezis et al. 2026", "Claims & Relations", "2-class", "Derived", "short"
)

write_latex(
  kable(tasks_table, format = "latex", align = "lllllll", booktabs = TRUE,
        longtable = TRUE,
        caption = "Thirty-four tasks in the benchmark. 'Type' is the report grouping used in Figure \\ref{fig-family}; it is a broad annotation type rather than a source-provided task family. 'Provenance': Verbatim = prompt unchanged from source paper, Verbatim-adapted = source content with format-only changes, Derived = my prompt based on a source codebook, public label definitions, or task description. 'Prompt': short (~15-35 lines) or long codebook (~47-126 lines). This is the trait that determines local batching feasibility. Source labels correspond to full entries in the references. The CMP task uses the Halterman \\& Keith (2026) domain-collapsed 7-class version of the Manifesto Project (2025) CMP/MARPOR coding scheme, not the full CMP category set. \\label{tab:tasks}") |>
    kable_styling(font_size = 8, latex_options = c("repeat_header"),
                  repeat_header_text = "") |>
    column_spec(2, width = "18em") |>
    column_spec(3, width = "7em") |>
    column_spec(4, width = "8em"),
  "output/tables/table-tasks-appendix.tex"
)

# Table: winners
winners <- df |>
  group_by(family, task_short) |>
  arrange(desc(headline_f1)) |>
  slice(1) |>
  ungroup() |>
  arrange(factor(family, levels = c("Relevance & Harm", "Position & Tone",
                                    "Events & Actions", "Claims & Relations",
                                    "Issues & Topics")),
          task_short) |>
  transmute(
    Type = family,
    Task = as.character(task_short),
    `Top model` = as.character(model_short),
    `Main F1` = sprintf("%.3f", headline_f1),
    Accuracy = sprintf("%.3f", accuracy),
    MCC = sprintf("%.3f", mcc)
  )

write_latex(
  kable(winners, format = "latex", align = "llllll", booktabs = TRUE,
        caption = "Top model per task by the main F1 score across the unified 34-task benchmark, grouped by the report's five annotation types. Accuracy and MCC refer to the same model-task pair. CIs and the full per-(task, model) table are on the GitHub repository. \\label{tab:winners}") |>
    kable_styling(font_size = 8.5, latex_options = c("HOLD_position")) |>
    collapse_rows(columns = 1, valign = "top"),
  "output/tables/table-winners-appendix.tex"
)

# Table: response-format/label failures
malformed_local <- batch_local_df |>
  filter(b10_parse_err_rate >= 0.05) |>
  transmute(
    Task = as.character(task_short),
    Model = as.character(model_short),
    `Batch size` = 10,
    Scope = "Local b=10",
    `Failure rate` = b10_parse_err_rate
  )

malformed_api <- sb |>
  filter(model %in% c("gpt-5.5", "gpt-5.4-nano"),
         batch_size %in% c(10, 20),
         parse_err_rate >= 0.05) |>
  transmute(
    Task = as.character(task_short),
    Model = as.character(model_short),
    `Batch size` = batch_size,
    Scope = "OpenAI diagnostic",
    `Failure rate` = parse_err_rate
  )

malformed <- bind_rows(malformed_local, malformed_api) |>
  arrange(desc(`Failure rate`)) |>
  mutate(`Failure rate` = scales::percent(`Failure rate`, accuracy = 1))

write_latex(
  kable(malformed, format = "latex", align = "llrlr", booktabs = TRUE,
        caption = "Batched model-task pairs with response-format or label failure rates of at least 5 percent, not a complete list of all model-task pairs. Failures include responses that do not parse, responses with the wrong object or array shape, and responses that use labels outside the allowed label set. Local rows come from the full 34-task grid with 10 items per prompt for the five local models (170 pairs). OpenAI rows come from a separate 10-task prompt-batching diagnostic with 10 and 20 items per prompt (40 pairs); these rows do not enter the main one-item-at-a-time benchmark. Claude Sonnet and DeepSeek API prompt batching are not covered here. \\label{tab:malformed}") |>
    kable_styling(font_size = 8, latex_options = c("HOLD_position")),
  "output/tables/table-malformed-appendix.tex"
)

# Table: alternative metrics
altmet <- df |>
  group_by(model_short) |>
  summarise(`Observed tasks` = n_distinct(task),
            `Missing tasks` = total_task_count - `Observed tasks`,
            `Accuracy tasks` = sum(!is.na(accuracy)),
            `MCC tasks` = sum(!is.na(mcc)),
            `Mean F1` = if_else(`Observed tasks` == total_task_count,
                                      mean(headline_f1, na.rm = TRUE),
                                      NA_real_),
            `Mean accuracy` = if_else(`Observed tasks` == total_task_count,
                                      mean(accuracy, na.rm = TRUE),
                                      NA_real_),
            `Mean MCC` = if_else(`Observed tasks` == total_task_count,
                                 mean(mcc, na.rm = TRUE),
                                 NA_real_),
            .groups = "drop") |>
  arrange(is.na(`Mean F1`), desc(`Mean F1`), desc(`Observed tasks`)) |>
  transmute(
    Model = as.character(model_short),
    `F1 tasks` = paste0(`Observed tasks`, "/", total_task_count),
    `Accuracy tasks` = paste0(`Accuracy tasks`, "/", total_task_count),
    `MCC tasks` = paste0(`MCC tasks`, "/", total_task_count),
    `Mean F1` = if_else(is.na(`Mean F1`), "", sprintf("%.3f", `Mean F1`)),
    `Mean accuracy` = if_else(is.na(`Mean accuracy`), "", sprintf("%.3f", `Mean accuracy`)),
    `Mean MCC` = if_else(is.na(`Mean MCC`), "", sprintf("%.3f", `Mean MCC`))
  )

write_latex(
  kable(altmet, format = "latex", align = "lrrrrrr", booktabs = TRUE,
        caption = "Unified 34-task model averages. F1 coverage is complete for all nine models. Accuracy and MCC are averaged over tasks where the metric is defined in the same way across outputs; the multi-binary ATI task is excluded from those two columns. The main F1 score is the task-specific summary metric. \\label{tab:altmetrics}") |>
    kable_styling(font_size = 9, latex_options = c("HOLD_position")),
  "output/tables/table-altmetrics-appendix.tex"
)

message("Wrote report figures and tables.")
