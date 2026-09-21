# Render every figure on the benchmark page in the CLARA house style.
# Inputs: the release directory and the standard-rate cost table. Python
# (build_refresh_figures.py) writes figure-data.json first; this script adds
# each figure's file, title, caption and plotted values to it.
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(jsonlite)
  library(ggrepel); library(scales)
})
set.seed(20260921)

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[1] else "output/sidecar/refresh_20260910_release_33"
cost_path <- if (length(args) >= 2) args[2] else "output/sidecar/jev_sidecar/cost_performance.csv"
out <- file.path(root, "preview/llm-benchmark/figures")
manifest <- fromJSON(file.path(out, "figure-data.json"), simplifyVector = FALSE)
release <- fromJSON(file.path(root, "release.json"))

# CLARA palette: open-weight models are the focal group, API models the comparison.
CLARA_BLUE <- "#2f6f9f"
CLARA_DARK <- "#1d4f77"
CLARA_GREY <- "grey55"
CLARA_GUIDE <- "grey85"
group_colors <- c("Open weights" = CLARA_BLUE, "API" = CLARA_GREY)

theme_clara <- function(base_size = 11) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(color = "black", face = "bold", size = base_size - 1, hjust = 0),
      panel.border = element_rect(color = "grey70"),
      axis.ticks = element_blank(),
      legend.position = "none",
      plot.margin = margin(8, 14, 8, 8)
    )
}

models <- bind_rows(lapply(manifest$models, as_tibble)) |>
  mutate(group = if_else(kind == "API", "API", "Open weights"),
         label = if_else(hardware_tier == "multi-gpu", paste0(label, " (2 GPUs)"), label),
         short_label = if_else(hardware_tier == "multi-gpu", paste0(short_label, " (2 GPUs)"), short_label))
featured <- unlist(manifest$featured_models)
stopifnot(all(featured %in% models$model))
meta <- bind_rows(lapply(manifest$task_metadata, function(x) tibble(
  task = x$task, paper_family = x$paper_family, complexity = x$complexity,
  effective_labels = x$effective_labels)))
n_tasks <- nrow(meta)
scores <- release$tasks |> select(model, task, headline_f1) |>
  inner_join(models |> select(model, group, hardware_tier), by = "model") |>
  left_join(meta, by = "task")
stopifnot(nrow(scores) == nrow(models) * n_tasks, !anyNA(scores$headline_f1))

entries <- list(featured = list(), more = list())
save_figure <- function(p, stem, title, caption, values, width, height, section) {
  for (ext in c("svg", "pdf", "png")) {
    ggsave(file.path(out, paste0(stem, ".", ext)), p, width = width, height = height,
           dpi = 300, bg = "white")
  }
  entries[[section]][[length(entries[[section]]) + 1]] <<- list(
    file = stem, title = title, caption = caption, data = values)
}

# Ranked dot plot with the 95% task-bootstrap interval as a pale segment.
overview <- function(ids, label_col) {
  d <- models |> filter(model %in% ids) |> arrange(mean_task_f1) |>
    mutate(name = factor(.data[[label_col]], levels = .data[[label_col]]))
  right <- max(d$task_ci_high) + 0.03
  ggplot(d, aes(y = name)) +
    geom_segment(aes(x = task_ci_low, xend = task_ci_high, yend = name),
                 color = CLARA_GUIDE, linewidth = 1.1) +
    geom_point(aes(x = mean_task_f1, color = group), size = 2.4) +
    geom_text(aes(x = right, label = sprintf("%.3f", mean_task_f1)),
              hjust = 1, size = 3, color = "grey20") +
    scale_color_manual(values = group_colors) +
    scale_x_continuous(breaks = seq(0.5, 0.8, 0.05)) +
    coord_cartesian(xlim = c(min(d$task_ci_low) - 0.01, right), clip = "off") +
    labs(x = sprintf("Mean F1 across %d tasks", n_tasks), y = NULL) +
    theme_clara()
}
overview_caption <- sprintf(paste(
  "Each row shows one model. Points show mean F1 across the %d tasks, printed on the right,",
  "and grey bars show 95%% intervals from resampling tasks. Blue marks open-weight models",
  "and grey marks API models."), n_tasks)

p <- overview(featured, "short_label")
save_figure(p, "fig-recent-mean-f1", "Overall performance", overview_caption,
            models |> filter(model %in% featured) |> select(model, mean_task_f1, task_ci_low, task_ci_high),
            6.5, 0.22 * length(featured) + 0.9, "featured")

# Cost against performance for the API models, with the best one-GPU open model as a reference.
costs <- read.csv(cost_path) |>
  select(model, cost_per_1k_items, cost_basis) |>
  inner_join(models |> filter(kind == "API") |> select(model, short_label, mean_task_f1), by = "model") |>
  mutate(estimated = cost_basis %in% c("token_estimate", "provider_tokens"))
stopifnot(nrow(costs) == sum(models$kind == "API"))
best_open <- models |> filter(hardware_tier == "single-gpu") |> slice_max(mean_task_f1, n = 1)
p <- ggplot(costs, aes(cost_per_1k_items, mean_task_f1)) +
  geom_hline(yintercept = best_open$mean_task_f1, color = CLARA_BLUE, linetype = "dashed", linewidth = 0.5) +
  annotate("text", x = max(costs$cost_per_1k_items), y = best_open$mean_task_f1 - 0.004,
           label = sprintf("Best open model on one GPU (%s), %.3f", best_open$short_label, best_open$mean_task_f1),
           hjust = 1, vjust = 1, size = 3, color = CLARA_BLUE) +
  geom_point(aes(shape = estimated), color = CLARA_DARK, size = 2.4, stroke = 0.8) +
  geom_text_repel(aes(label = short_label), size = 3, color = "grey20", min.segment.length = 0,
                  segment.color = "grey70", box.padding = 0.4, seed = 20260921) +
  scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1)) +
  scale_x_log10(breaks = c(0.03, 0.1, 0.3, 1, 3, 10), labels = c("$0.03", "$0.10", "$0.30", "$1", "$3", "$10")) +
  labs(x = "Cost per 1,000 texts at standard prices (USD, log scale)",
       y = sprintf("Mean F1 across %d tasks", n_tasks)) +
  theme_clara()
save_figure(p, "fig-cost-f1", "Cost and performance",
            sprintf(paste(
              "Each point shows one API model: its mean F1 across the %d tasks against its cost per",
              "1,000 texts at standard prices in September 2026. Hollow points mark costs estimated from",
              "token counts rather than taken from provider bills. The dashed line marks the best open-weight",
              "model that runs on one GPU, which has no per-text charge."), n_tasks),
            costs |> select(model, cost_per_1k_items, mean_task_f1, estimated), 7, 4.6, "featured")

# Best API score minus best one-GPU open score, task by task.
task_gap <- function(ids) {
  best <- scores |> filter(model %in% ids, hardware_tier %in% c("api", "single-gpu")) |>
    group_by(task, group) |> summarise(f1 = max(headline_f1), .groups = "drop") |>
    pivot_wider(names_from = group, values_from = f1) |>
    mutate(gap = API - `Open weights`, winner = if_else(gap < 0, "Open weights", "API"),
           name = reorder(gsub("_", " ", task), gap))
  lim <- max(abs(best$gap)) + 0.02
  ggplot(best, aes(y = name)) +
    geom_vline(xintercept = 0, color = "grey70", linewidth = 0.4) +
    geom_segment(aes(x = 0, xend = gap, yend = name), color = CLARA_GUIDE, linewidth = 0.9) +
    geom_point(aes(x = gap, color = winner), size = 2.2) +
    annotate("text", x = lim, y = Inf, label = "API model better", hjust = 1, vjust = -0.6, size = 3, color = "grey30") +
    annotate("text", x = -lim, y = Inf, label = "Open model better", hjust = 0, vjust = -0.6, size = 3, color = CLARA_BLUE) +
    scale_color_manual(values = group_colors) +
    coord_cartesian(xlim = c(-lim, lim), clip = "off") +
    labs(x = "Best API F1 minus best one-GPU open F1", y = NULL) +
    theme_clara() + theme(plot.margin = margin(18, 14, 8, 8))
}
gap_counts <- function(ids) {
  sum(models$model %in% ids & models$hardware_tier == "api")
}
p <- task_gap(featured)
save_figure(p, "fig-recent-task-gap", "API and open models by task",
            sprintf(paste(
              "Each row shows one task. Points show the best score among the %d API models minus the best",
              "score among the %d open-weight models that run on one GPU. Blue points left of zero mark",
              "tasks where an open model performs better. Because the best model is chosen after observing",
              "the results, these gaps describe the best case for each group."),
              gap_counts(featured), sum(models$model %in% featured & models$hardware_tier == "single-gpu")),
            NULL, 7, 7.8, "featured")

# Annotation types as small multiples, with models in the same overall order in every panel.
family_levels <- c("Relevance & Harm", "Position & Tone", "Events & Actions", "Claims & Relations", "Issues & Topics")
family_plot <- function(ids, label_col, ncol) {
  d <- scores |> filter(model %in% ids) |> group_by(model, group, paper_family) |>
    summarise(mean_f1 = mean(headline_f1), tasks = n(), .groups = "drop") |>
    left_join(models |> select(model, name = all_of(label_col), overall = mean_task_f1), by = "model") |>
    mutate(name = reorder(name, overall),
           paper_family = factor(paper_family, levels = family_levels))
  strip <- d |> distinct(paper_family, tasks) |> arrange(paper_family) |>
    mutate(text = sprintf("%s (%d tasks)", paper_family, tasks))
  d <- d |> mutate(panel = factor(sprintf("%s (%d tasks)", paper_family, tasks), levels = strip$text))
  ggplot(d, aes(mean_f1, name)) +
    geom_point(aes(color = group), size = 1.9) +
    facet_wrap(~panel, ncol = ncol) +
    scale_color_manual(values = group_colors) +
    labs(x = "Mean F1 within the annotation type", y = NULL) +
    theme_clara(base_size = 10) + theme(panel.spacing = unit(0.8, "lines"))
}
p <- family_plot(featured, "short_label", 3)
save_figure(p, "fig-recent-family", "Performance by annotation type",
            paste(
              "Mean F1 within the five annotation types used in the paper, with each task weighted equally.",
              "Models appear in the same order in every panel, ranked by their mean across all tasks.",
              "These types differ from the categories in the task selector below."),
            NULL, 8.5, 0.34 * length(featured) + 1.2, "featured")

# Coding complexity: all scores and the best model per task, API against open.
complexity_plot <- function(ids) {
  d <- scores |> filter(model %in% ids, hardware_tier %in% c("api", "single-gpu"))
  best <- d |> group_by(task, complexity, group) |> summarise(f1 = max(headline_f1), .groups = "drop")
  lines <- bind_rows(
    d |> group_by(complexity, group) |> summarise(f1 = mean(headline_f1), .groups = "drop") |>
      mutate(panel = "Average model"),
    best |> group_by(complexity, group) |> summarise(f1 = mean(f1), .groups = "drop") |>
      mutate(panel = "Best model on each task")) |>
    mutate(complexity = factor(complexity, levels = c("Low", "Medium", "High")),
           panel = factor(panel, levels = c("Average model", "Best model on each task")))
  ends <- lines |> filter(complexity == "High") |>
    mutate(text = if_else(group == "API", "API", "Open, one GPU"))
  ggplot(lines, aes(complexity, f1, group = group, color = group)) +
    geom_line(linewidth = 0.65) + geom_point(size = 2) +
    geom_text(data = ends, aes(label = text), hjust = 0, nudge_x = 0.12, size = 3) +
    facet_wrap(~panel, ncol = 2) +
    scale_color_manual(values = group_colors) +
    scale_x_discrete(expand = expansion(add = c(0.3, 1.1))) +
    coord_cartesian(clip = "off") +
    labs(x = "Coding complexity", y = "Mean F1") +
    theme_clara()
}
complexity_caption <- paste(
  "Tasks grouped by coding complexity, following the paper. High-complexity tasks allow several",
  "labels per text or use at least eight labels in practice; medium-complexity tasks use at least three",
  "labels or have a prompt of 300 words or more; the remaining tasks are low complexity. The number of",
  "labels in practice is the exponential of the entropy of the gold labels, which counts rare labels",
  "less than common ones. Open models are those that run on one GPU.")
p <- complexity_plot(featured)
save_figure(p, "fig-recent-complexity", "Coding complexity", complexity_caption, NULL, 7.5, 3.4, "featured")

# Further figures: every model.
all_ids <- models$model
p <- overview(all_ids, "label")
save_figure(p, "fig-mean-f1", "Overall performance, all models", overview_caption,
            models |> select(model, mean_task_f1, task_ci_low, task_ci_high),
            7.5, 0.22 * length(all_ids) + 0.9, "more")
p <- task_gap(all_ids)
save_figure(p, "fig-best-local-api-gap", "API and open models by task, all models",
            "Same as the task-gap figure above, for every API model and every open model that runs on one GPU.",
            NULL, 7, 7.8, "more")
p <- family_plot(all_ids, "label", 3)
save_figure(p, "fig-family", "Performance by annotation type, all models",
            "Same as the annotation-type figure above, for all models.",
            NULL, 10, 0.3 * length(all_ids) + 1.2, "more")
p <- complexity_plot(all_ids)
save_figure(p, "fig-complexity", "Coding complexity, all models",
            "Same as the complexity figure above, for every API model and every open model that runs on one GPU.",
            NULL, 7.5, 3.4, "more")

# Label structure: the task gap against the number of labels in practice.
structure <- scores |> filter(hardware_tier %in% c("api", "single-gpu")) |>
  group_by(task, effective_labels, group) |> summarise(f1 = max(headline_f1), .groups = "drop") |>
  pivot_wider(names_from = group, values_from = f1) |> mutate(gap = API - `Open weights`)
p <- ggplot(structure, aes(effective_labels, gap)) +
  geom_hline(yintercept = 0, color = "grey70", linetype = "dashed", linewidth = 0.4) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = CLARA_GREY, linewidth = 0.6) +
  geom_point(color = CLARA_DARK, size = 2) +
  scale_x_log10(breaks = c(1, 2, 3, 5, 10, 20)) +
  labs(x = "Number of labels in practice (log scale)", y = "Best API F1 minus best one-GPU open F1") +
  theme_clara()
save_figure(p, "fig-label-structure-gap", "Label structure",
            paste(
              "Each point shows one task: the best API score minus the best score among open models that run",
              "on one GPU, against the number of labels in practice. The dashed line marks equal performance,",
              "and the solid line is a linear fit on the log scale."),
            structure |> select(task, effective_labels, gap), 6.5, 4, "more")

# Speed of the open models that ran on identical texts and settings.
tp <- bind_rows(lapply(manifest$throughput$models, as_tibble)) |>
  left_join(models |> select(model, short_label), by = "model")
p <- ggplot(tp, aes(seconds_per_item, mean_f1)) +
  geom_point(color = CLARA_BLUE, size = 2.4) +
  geom_text_repel(aes(label = short_label), size = 3, color = "grey20", min.segment.length = 0,
                  segment.color = "grey70", box.padding = 0.4, seed = 20260921) +
  scale_x_log10() +
  labs(x = "Generation seconds per text (log scale)", y = sprintf("Mean F1 across %d tasks", n_tasks)) +
  theme_clara()
speed_caption <- sprintf(paste(
  "Each point shows one open-weight model: mean F1 against generation time per text. All models coded",
  "the same %s texts on one RTX PRO 6000 GPU with identical settings. Load and queue times are excluded."),
  format(manifest$throughput$items, big.mark = ","))
save_figure(p, "fig-speed", "Quality and generation time", speed_caption,
            tp |> select(model, seconds_per_item, mean_f1), 7, 4.4, "more")
runtime <- tp |> mutate(minutes = seconds_per_item * 1000 / 60) |> arrange(minutes) |>
  mutate(name = factor(short_label, levels = rev(short_label)))
p <- ggplot(runtime, aes(minutes, name)) +
  geom_segment(aes(x = 0, xend = minutes, yend = name), color = CLARA_GUIDE, linewidth = 0.9) +
  geom_point(color = CLARA_BLUE, size = 2.2) +
  geom_text(aes(label = sprintf("%.1f", minutes)), hjust = -0.5, size = 3, color = "grey20") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = "Generation minutes per 1,000 texts", y = NULL) +
  theme_clara()
save_figure(p, "fig-local-runtime-per-1000", "Generation time per 1,000 texts",
            "Generation minutes per 1,000 texts, from the same runs as the previous figure.",
            runtime |> select(model, minutes), 6.5, 0.3 * nrow(runtime) + 1, "more")

manifest$featured_figures <- entries$featured
manifest$figures <- entries$more
write_json(manifest, file.path(out, "figure-data.json"), auto_unbox = TRUE, pretty = TRUE,
           digits = NA, null = "null")
