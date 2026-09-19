#!/usr/bin/env Rscript

# Supervised-baseline learning curves against the zero-shot LLM reference lines.
#
# Reads output/supervised_baseline_{tfidf,e5}.csv (produced by
# code/build_supervised_baseline.py) and output/summary.csv, and writes:
#   output/figures/fig-supervised-curve.png
#   output/supervised_crossover.csv
#
# The LLM reference lines are the same headline_f1 values the paper reports, so
# the vertical distance in the figure is directly interpretable: it is the gap
# between a classifier trained on n hand-coded labels and a model given none.

suppressPackageStartupMessages({
  library(tidyverse)
  library(haschaR)
})

file_arg <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", file_arg[grepl("^--file=", file_arg)][1])
if (is.na(script_file)) script_file <- "code/plot_supervised_baseline.R"
repo <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
setwd(repo)

API_MODELS <- c("claude-sonnet-4-6", "deepseek-v4-pro", "gpt-5.4-nano", "gpt-5.5")

read_method <- function(method) {
  path <- sprintf("output/supervised_baseline_%s.csv", method)
  if (!file.exists(path)) return(NULL)
  read_csv(path, show_col_types = FALSE) %>% mutate(method = method)
}

sup <- bind_rows(read_method("tfidf"), read_method("e5")) %>%
  filter(!is.na(headline_f1))
stopifnot(nrow(sup) > 0)

# LLM reference lines, restricted to the tasks the supervised run covers so the
# two sides describe the same task set.
llm <- read_csv("output/summary.csv", show_col_types = FALSE) %>%
  filter(task %in% unique(sup$task)) %>%
  mutate(class = if_else(model %in% API_MODELS, "Best API model", "Best local model")) %>%
  group_by(task, class) %>%
  summarise(f1 = max(headline_f1, na.rm = TRUE), .groups = "drop")

llm_ref <- llm %>% group_by(class) %>% summarise(f1 = mean(f1), .groups = "drop")

method_labels <- c(tfidf = "TF-IDF + logistic regression",
                   e5 = "multilingual-E5 + logistic regression")

# Average over seeds within task, then over tasks, so every task counts once
# regardless of how many draws it supported.
curve <- sup %>%
  group_by(task, method, n_train) %>%
  summarise(f1 = mean(headline_f1), .groups = "drop") %>%
  group_by(method, n_train) %>%
  summarise(f1 = mean(f1), n_tasks = n(), .groups = "drop") %>%
  filter(n_tasks >= 10) %>%    # drop sizes only a handful of tasks can support
  mutate(method = recode(method, !!!method_labels))

p <- ggplot(curve, aes(x = n_train, y = f1, color = method)) +
  geom_hline(data = llm_ref, aes(yintercept = f1, linetype = class),
             color = "grey35", linewidth = 0.45) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.1) +
  scale_x_log10(breaks = c(50, 100, 250, 500, 1000, 2000),
                labels = c("50", "100", "250", "500", "1,000", "2,000")) +
  scale_color_manual(values = c("TF-IDF + logistic regression" = "grey45",
                                "multilingual-E5 + logistic regression" = "#1f78b4")) +
  scale_linetype_manual(values = c("Best API model" = "dashed",
                                   "Best local model" = "dotted")) +
  labs(x = "Hand-coded training examples per task (log scale)",
       y = "Mean F1 across tasks", color = NULL, linetype = NULL) +
  theme_hanno(fontsize = 11) +
  theme(legend.position = "bottom", legend.box = "vertical",
        legend.margin = margin(t = -4), panel.grid.minor = element_blank())

ggsave("output/figures/fig-supervised-curve.png", plot = p,
       width = 6.4, height = 4.4, dpi = 200)

# Crossover: the smallest training size at which a method's task-mean overtakes
# each LLM reference line. NA means it never does within the sizes tested.
per_task <- sup %>%
  group_by(task, method, n_train) %>%
  summarise(f1 = mean(headline_f1), .groups = "drop") %>%
  left_join(llm %>% pivot_wider(names_from = class, values_from = f1), by = "task")

crossover <- per_task %>%
  arrange(task, method, n_train) %>%
  group_by(task, method) %>%
  summarise(
    crosses_local = suppressWarnings(min(n_train[f1 >= `Best local model`])),
    crosses_api   = suppressWarnings(min(n_train[f1 >= `Best API model`])),
    best_f1 = max(f1),
    best_local_llm = first(`Best local model`),
    best_api_llm = first(`Best API model`),
    .groups = "drop"
  ) %>%
  mutate(across(starts_with("crosses_"), ~ if_else(is.finite(.x), .x, NA_real_)))

write_csv(crossover, "output/supervised_crossover.csv")

cat("wrote output/figures/fig-supervised-curve.png\n")
cat("wrote output/supervised_crossover.csv\n\n")
crossover %>%
  group_by(method) %>%
  summarise(
    tasks = n(),
    beat_local_ever = sum(!is.na(crosses_local)),
    beat_api_ever = sum(!is.na(crosses_api)),
    median_crossover_local = median(crosses_local, na.rm = TRUE),
    median_crossover_api = median(crosses_api, na.rm = TRUE)
  ) %>%
  print(n = Inf)
