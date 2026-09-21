#!/usr/bin/env Rscript

set.seed(20260901)

library(dplyr)
library(ggplot2)
library(readr)
library(scales)

input_dir <- "output/sidecar/frontier_2026/twitter_claims"

BLUE <- "#2f6f9f"
DARK <- "#1d4f77"
GREY <- "grey55"
LIGHT <- "grey85"
RED <- "#a64b45"

theme_clara_twitter <- function(fontsize = 13) {
  theme_bw(base_size = fontsize, base_family = "Helvetica") +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(color = "grey70", linewidth = 0.45),
      axis.ticks = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = fontsize - 1),
      legend.position = "none",
      plot.title = element_text(face = "bold", size = fontsize + 6),
      plot.subtitle = element_text(color = "grey30", margin = margin(b = 12)),
      plot.caption = element_text(
        color = "grey35", size = fontsize - 3, hjust = 0,
        lineheight = 1.15, margin = margin(t = 12)
      ),
      plot.margin = margin(12, 34, 10, 10)
    )
}

save_fig <- function(plot, stem, width, height) {
  pdf_path <- file.path(input_dir, paste0(stem, ".pdf"))
  png_path <- file.path(input_dir, paste0(stem, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, device = cairo_pdf)
  ggsave(png_path, plot, width = width, height = height, dpi = 450, bg = "white")
  stopifnot(file.exists(pdf_path), file.info(pdf_path)$size > 10000)
  stopifnot(file.exists(png_path), file.info(png_path)$size > 10000)
}

# Figure 1: first-to-latest change within three recent model sequences.
lineage_ids <- c(
  "qwen1_5_32b_chat_hive", "qwen2_5_32b_instruct_bf16_hive",
  "qwen3_32b_bf16_hive", "qwen3_6_27b_fp8_hive",
  "qwen3_30b_a3b_bf16_hive", "qwen3_next_80b_a3b_fp8_hive",
  "qwen3_5_35b_a3b_fp8_hive",
  "gemma3_27b_it_fp8_dynamic_hive", "gemma4_31b_it_qat_w4a16_hive"
)

lineages <- read_csv(
  "output/sidecar/frontier_2026/multi_model_patterns/model_summary.csv",
  show_col_types = FALSE
) %>%
  filter(checkpoint_id %in% lineage_ids) %>%
  transmute(
    checkpoint_id,
    lineage = case_when(
      checkpoint_id %in% c(
        "qwen1_5_32b_chat_hive", "qwen2_5_32b_instruct_bf16_hive",
        "qwen3_32b_bf16_hive", "qwen3_6_27b_fp8_hive"
      ) ~ "Qwen dense 27–32B",
      checkpoint_id %in% c(
        "qwen3_30b_a3b_bf16_hive", "qwen3_next_80b_a3b_fp8_hive",
        "qwen3_5_35b_a3b_fp8_hive"
      ) ~ "Qwen MoE A3B",
      TRUE ~ "Gemma dense 27–31B"
    ),
    release_date = as.Date(plot_date),
    mean_f1,
    series_color = case_when(
      lineage == "Qwen dense 27–32B" ~ "dense_qwen",
      lineage == "Qwen MoE A3B" ~ "moe_qwen",
      TRUE ~ "context"
    )
  ) %>%
  arrange(lineage, release_date)

stopifnot(nrow(lineages) == 9L, n_distinct(lineages$lineage) == 3L)
stopifnot(setequal(lineages$checkpoint_id, lineage_ids))

lineage_order <- c("Qwen dense 27–32B", "Qwen MoE A3B", "Gemma dense 27–31B")
lineages <- lineages %>%
  mutate(
    lineage = factor(lineage, levels = lineage_order),
    panel_label = case_when(
      lineage == "Qwen dense 27–32B" ~ "Qwen dense 27–32B",
      lineage == "Qwen MoE A3B" ~ "Qwen MoE · 3B active",
      TRUE ~ "Gemma dense 27–31B"
    ),
    panel_label = factor(
      panel_label,
      levels = c(
        "Qwen dense 27–32B",
        "Qwen MoE · 3B active",
        "Gemma dense 27–31B"
      )
    )
  ) %>%
  group_by(lineage) %>%
  mutate(
    endpoint = row_number() %in% c(1L, n()),
    endpoint_side = case_when(
      row_number() == 1L ~ "first",
      row_number() == n() ~ "last",
      TRUE ~ NA_character_
    ),
    change_pp = 100 * (last(mean_f1) - first(mean_f1))
  ) %>%
  ungroup()

lineage_changes <- lineages %>%
  group_by(lineage) %>%
  summarize(change_pp = first(change_pp), .groups = "drop")
stopifnot(
  abs(lineage_changes$change_pp[[1]] - 4.832) < 0.002,
  abs(lineage_changes$change_pp[[2]] - 6.369) < 0.002,
  abs(lineage_changes$change_pp[[3]] - 1.018) < 0.002
)

p_lineages <- ggplot(
  lineages,
  aes(x = release_date, y = 100 * mean_f1, group = lineage, color = series_color)
) +
  geom_line(linewidth = 1.05) +
  geom_point(size = 2.6) +
  geom_text(
    data = filter(lineages, endpoint_side == "first"),
    aes(label = sprintf("%.1f", 100 * mean_f1)),
    hjust = 1.25, size = 3.2, color = "grey25", show.legend = FALSE
  ) +
  geom_label(
    data = filter(lineages, endpoint_side == "last"),
    aes(label = sprintf("%+.1f pp", change_pp)),
    hjust = 0.5, vjust = -1.6, size = 3.3,
    fontface = "bold", fill = "white", linewidth = 0,
    label.padding = unit(0.05, "lines"), show.legend = FALSE
  ) +
  facet_wrap(~panel_label, nrow = 1) +
  scale_color_manual(values = c(
    dense_qwen = DARK, moe_qwen = BLUE, context = GREY
  )) +
  scale_x_date(
    limits = as.Date(c("2023-10-01", "2026-10-01")),
    breaks = as.Date(c("2024-01-01", "2025-01-01", "2026-01-01")),
    date_labels = "%Y", expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(58, 70), breaks = c(58, 62, 66, 70),
    labels = function(x) sprintf("%d", x), expand = c(0, 0)
  ) +
  labs(
    title = "Performance gains differ across model lineages",
    x = NULL, y = "Mean F1",
    caption = "Mean F1 across 18 social science text classification tasks."
  ) +
  theme_clara_twitter(13) +
  theme(panel.spacing = unit(1.1, "lines"))

save_fig(p_lineages, "04_dense_lineages", 9.2, 4.6)
write_csv(
  lineages,
  file.path(input_dir, "04_recent_lineages_plot_data.csv")
)

# Figure 2: cumulative medium- and large-compute frontiers.
frontier_raw <- read_csv(
  file.path(input_dir, "05_compute_class_staircase.csv"), show_col_types = FALSE
) %>%
  mutate(plot_date = as.Date(plot_date))

stopifnot(n_distinct(frontier_raw$checkpoint_id) >= 10L)

all_dates <- sort(unique(frontier_raw$plot_date))
frontier_classes <- c(
  "Low (≤10B active)", "Medium (10–40B active)", "Large (>40B active)"
)
frontier_steps <- bind_rows(lapply(frontier_classes, function(class_name) {
  class_data <- filter(frontier_raw, compute_class == class_name)
  valid_dates <- all_dates[all_dates >= min(class_data$plot_date)]
  tibble(
    compute_class = class_name,
    plot_date = valid_dates,
    frontier_f1 = vapply(
      valid_dates,
      function(date) max(class_data$mean_f1[class_data$plot_date <= date]),
      numeric(1)
    )
  )
}))

frontier_setters <- frontier_raw %>% filter(class_frontier_setter)
frontier_ends <- frontier_steps %>%
  group_by(compute_class) %>%
  mutate(gain_pp = 100 * (last(frontier_f1) - first(frontier_f1))) %>%
  slice_tail(n = 1) %>%
  ungroup() %>%
  mutate(
    endpoint_label = case_when(
      grepl("Low", compute_class) ~ sprintf(
        "Low  %.1f  (%+.1f)\nup to 10B active", 100 * frontier_f1, gain_pp
      ),
      grepl("Medium", compute_class) ~ sprintf(
        "Medium  %.1f  (%+.1f)\n10–40B active", 100 * frontier_f1, gain_pp
      ),
      TRUE ~ sprintf(
        "Large  %.1f  (%+.1f)\nover 40B active", 100 * frontier_f1, gain_pp
      )
    ),
    plot_date = as.Date("2026-06-20"),
    label_y = case_when(
      grepl("Medium", compute_class) ~ 68.45,
      grepl("Large", compute_class) ~ 67.35,
      TRUE ~ 65.75
    )
  )

stopifnot(
  abs(frontier_ends$frontier_f1[grepl("Low", frontier_ends$compute_class)] - 0.657482) < 1e-6,
  abs(frontier_ends$frontier_f1[grepl("Medium", frontier_ends$compute_class)] - 0.680465) < 1e-6,
  abs(frontier_ends$frontier_f1[grepl("Large", frontier_ends$compute_class)] - 0.678665) < 1e-6
)

frontier_annotations <- frontier_raw %>%
  filter(checkpoint_id %in% c(
    "llama3_70b_instruct_fp8_hive",
    "qwen2_5_32b_instruct_bf16_hive",
    "gpt_oss_120b_mxfp4_hive"
  )) %>%
  transmute(
    plot_date,
    mean_f1,
    label = case_when(
      checkpoint_id == "llama3_70b_instruct_fp8_hive" ~ "Llama 3 70B",
      checkpoint_id == "qwen2_5_32b_instruct_bf16_hive" ~ "Qwen2.5 32B",
      TRUE ~ "GPT-OSS\n5.1B active"
    ),
    hjust = case_when(
      checkpoint_id == "llama3_70b_instruct_fp8_hive" ~ 1.1,
      TRUE ~ -0.12
    ),
    vjust = case_when(
      checkpoint_id == "gpt_oss_120b_mxfp4_hive" ~ -0.7,
      TRUE ~ -0.75
    )
  )

p_frontier <- ggplot(
  frontier_steps,
  aes(x = plot_date, y = 100 * frontier_f1, color = compute_class)
) +
  geom_step(linewidth = 1.15, direction = "hv") +
  geom_point(
    data = frontier_setters,
    aes(x = plot_date, y = 100 * mean_f1, color = compute_class),
    size = 2.6, show.legend = FALSE
  ) +
  geom_text(
    data = frontier_annotations,
    aes(
      x = plot_date, y = 100 * mean_f1, label = label,
      hjust = hjust, vjust = vjust
    ),
    inherit.aes = FALSE, size = 3.0, lineheight = 0.95,
    color = "grey30"
  ) +
  geom_text(
    data = frontier_ends,
    aes(y = label_y, label = endpoint_label),
    hjust = -0.08, vjust = 0.5, size = 3.5,
    lineheight = 0.95, fontface = "bold", show.legend = FALSE
  ) +
  annotate(
    "text", x = as.Date("2026-04-21"), y = 68.05,
    label = "Qwen3.6 27B", hjust = 1.08, vjust = -0.8,
    size = 3.3, color = DARK
  ) +
  scale_color_manual(values = c(
    "Low (≤10B active)" = "grey72",
    "Medium (10–40B active)" = DARK,
    "Large (>40B active)" = GREY
  )) +
  scale_x_date(
    limits = as.Date(c("2024-01-01", "2027-02-01")),
    breaks = as.Date(c("2024-01-01", "2025-01-01", "2026-01-01")),
    date_labels = "%Y", expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(58, 69), breaks = c(60, 64, 68),
    labels = function(x) sprintf("%d", x), expand = c(0, 0)
  ) +
  labs(
    title = "A medium model reached the large-model frontier",
    x = NULL, y = "Best mean F1 so far",
    caption = "21 checkpoints evaluated on 18 social science text classification tasks."
  ) +
  theme_clara_twitter(13)

save_fig(p_frontier, "05_compute_class_staircase", 8.4, 5.1)

# Figure 3: positive-label propensity across the 21-checkpoint roster.
selectivity <- read_csv(
  file.path(input_dir, "06_binary_positive_rate.csv"), show_col_types = FALSE
) %>%
  mutate(release_date = as.Date(release_date)) %>%
  arrange(release_date)

stopifnot(nrow(selectivity) == 21L)
stopifnot(all(abs(
  selectivity$predicted_positive_rate - selectivity$gold_positive_rate -
    selectivity$positive_rate_gap
) < 1e-10))

date_years <- as.numeric(selectivity$release_date - min(selectivity$release_date)) / 365.25
trend <- lm(positive_rate_gap ~ date_years, data = selectivity)
trend_per_year <- unname(coef(trend)[["date_years"]])
stopifnot(abs(trend_per_year - (-0.049564)) < 1e-5)
gold_rate <- median(selectivity$gold_positive_rate)
trend_dates <- range(selectivity$release_date)
trend_values <- predict(
  trend,
  newdata = data.frame(
    date_years = as.numeric(trend_dates - min(selectivity$release_date)) / 365.25
  )
)
trend_segment <- tibble(
  x = trend_dates[[1]], xend = trend_dates[[2]],
  y = 100 * trend_values[[1]],
  yend = 100 * trend_values[[2]]
)

p_selectivity <- ggplot(
  selectivity, aes(x = release_date, y = 100 * positive_rate_gap)
) +
  geom_hline(
    yintercept = 0, linewidth = 0.75,
    linetype = "22", color = GREY
  ) +
  geom_segment(
    data = trend_segment,
    aes(x = x, xend = xend, y = y, yend = yend),
    inherit.aes = FALSE, linewidth = 1.15, color = DARK
  ) +
  geom_point(size = 2.7, color = DARK) +
  scale_x_date(
    limits = as.Date(c("2024-01-01", "2026-07-01")),
    breaks = as.Date(c("2024-01-01", "2025-01-01", "2026-01-01")),
    date_labels = "%Y", expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(-11, 11), breaks = c(-10, -5, 0, 5, 10),
    labels = function(x) sprintf("%+d", x), expand = c(0, 0)
  ) +
  labs(
    title = "Later models tend to underpredict positive cases",
    x = NULL, y = "Predicted positive share minus actual positive share (pp)",
    caption = "21 checkpoints evaluated on eight binary social science text classification tasks."
  ) +
  theme_clara_twitter(13)

save_fig(p_selectivity, "06_binary_positive_rate", 8.4, 5.1)

# Additional figure 1: overall frontier versus the medium-Qwen dense lineage.
frontier_all <- read_csv(
  file.path(input_dir, "01_frontier_and_efficiency.csv"), show_col_types = FALSE
) %>%
  mutate(plot_date = as.Date(plot_date)) %>%
  arrange(plot_date)

qwen_medium <- lineages %>%
  filter(lineage == "Qwen dense 27–32B") %>%
  select(plot_date = release_date, mean_f1)

stopifnot(nrow(frontier_all) == 21L, nrow(qwen_medium) == 4L)
stopifnot(abs(100 * (last(qwen_medium$mean_f1) - first(qwen_medium$mean_f1)) - 4.832) < 0.002)

p_combined <- ggplot() +
  geom_step(
    data = frontier_all,
    aes(x = plot_date, y = 100 * frontier_f1),
    color = GREY, linewidth = 1.0, direction = "hv"
  ) +
  geom_line(
    data = qwen_medium,
    aes(x = plot_date, y = 100 * mean_f1),
    color = DARK, linewidth = 1.05
  ) +
  geom_point(
    data = qwen_medium,
    aes(x = plot_date, y = 100 * mean_f1),
    color = DARK, size = 2.7
  ) +
  annotate(
    "text", x = as.Date("2024-07-08"), y = 67.65,
    label = "Frontier", hjust = 0, vjust = -0.65,
    color = GREY, fontface = "bold", size = 3.4
  ) +
  annotate(
    "text", x = as.Date("2025-10-01"), y = 64.8,
    label = "Qwen 27–32B: +4.8 pp", hjust = 0.5,
    color = DARK, fontface = "bold", size = 3.4
  ) +
  scale_x_date(
    limits = as.Date(c("2024-01-01", "2026-09-01")),
    breaks = as.Date(c("2024-01-01", "2025-01-01", "2026-01-01")),
    date_labels = "%Y", expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(62, 69), breaks = c(62, 64, 66, 68),
    expand = c(0, 0)
  ) +
  labs(
    title = "The frontier barely moved; medium Qwen caught up",
    x = NULL, y = "Mean F1 across 18 tasks",
    caption = "Grey: best tested score by each date. Blue: comparable dense Qwen checkpoints."
  ) +
  theme_clara_twitter(13)

save_fig(p_combined, "01_frontier_and_efficiency", 9.4, 5.2)

# Additional figure 2: task-category differences for the endpoint model pair.
categories <- read_csv(
  file.path(input_dir, "02_capability_reallocation.csv"), show_col_types = FALSE
) %>%
  mutate(
    category_label = recode(
      category,
      "Events and protest" = "Events and protest",
      "Policy and topics" = "Policy and topics",
      "Stance and sentiment" = "Stance and sentiment",
      "Claims and relations" = "Claims and relations",
      "Relevance and tone" = "Relevance and tone"
    ),
    category_label = factor(category_label, levels = category_label[order(delta_f1_points)])
  )

stopifnot(nrow(categories) == 5L, sum(categories$task_count) == 34L)

p_categories <- ggplot(categories, aes(x = delta_f1_points, y = category_label)) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = 0.6) +
  geom_segment(
    aes(x = 0, xend = delta_f1_points, yend = category_label),
    color = LIGHT, linewidth = 1.0
  ) +
  geom_point(
    aes(color = delta_f1_points > 0), size = 3.0
  ) +
  geom_text(
    aes(
      label = sprintf("%+.1f", delta_f1_points),
      hjust = if_else(delta_f1_points > 0, -0.35, 1.35)
    ),
    size = 3.5, fontface = "bold", color = "grey20"
  ) +
  scale_color_manual(values = c(`TRUE` = DARK, `FALSE` = GREY)) +
  scale_x_continuous(
    limits = c(-3.1, 5.2), breaks = c(-2, 0, 2, 4),
    expand = c(0, 0)
  ) +
  labs(
    title = "Task-category gains were uneven",
    x = "Qwen3.6 27B minus Llama 3.1 70B (F1 points)", y = NULL,
    caption = paste(
      "Thirty-four repository tasks grouped by coding target.\n",
      "Exploratory: the event advantage reversed in the external task set."
    )
  ) +
  theme_clara_twitter(13)

save_fig(p_categories, "02_capability_reallocation", 8.4, 5.2)

# Additional figure 3: sensitivity of the endpoint comparison to task selection.
task_sets <- read_csv(
  file.path(input_dir, "03_task_set_sensitivity.csv"), show_col_types = FALSE
) %>%
  mutate(
    display_label = factor(display_label, levels = rev(display_label)),
    set_type = recode(set_type, repository = "Repository", external = "External")
  )

external_mean <- mean(task_sets$delta_f1_points[task_sets$set_type == "External"])
stopifnot(nrow(task_sets) == 6L, abs(external_mean - (-2.568)) < 0.002)

p_sensitivity <- ggplot(task_sets, aes(x = delta_f1_points, y = display_label)) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = 0.6) +
  geom_segment(
    aes(x = 0, xend = delta_f1_points, yend = display_label),
    color = LIGHT, linewidth = 1.0
  ) +
  geom_point(aes(color = set_type), size = 3.0) +
  geom_text(
    aes(
      label = sprintf("%+.1f", delta_f1_points),
      hjust = if_else(delta_f1_points >= 0, -0.4, 1.4)
    ),
    color = "grey20", size = 3.5, fontface = "bold"
  ) +
  scale_color_manual(values = c(Repository = DARK, External = GREY)) +
  scale_x_continuous(
    limits = c(-6.5, 1.4), breaks = c(-6, -4, -2, 0),
    expand = c(0, 0)
  ) +
  labs(
    title = "The model ranking changes with the task set",
    x = "Qwen3.6 27B minus Llama 3.1 70B (F1 points)", y = NULL,
    caption = sprintf(
      "Blue: repository task sets. Grey: independently selected external tasks (mean %.1f points).",
      external_mean
    )
  ) +
  theme_clara_twitter(13)

save_fig(p_sensitivity, "03_task_set_sensitivity", 8.4, 5.6)

message(sprintf(
  "Saved six CLARA-style Twitter figures; selectivity trend = %.2f pp/year.",
  100 * trend_per_year
))
