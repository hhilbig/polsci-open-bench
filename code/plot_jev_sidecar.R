#!/usr/bin/env Rscript

# Accuracy against cost, and accuracy against latency, for the nine models on
# the refresh panel, with Jev 1.13 highlighted.
#
# Compares nine commercial API models on the same 33 political-science coding
# tasks (100 frozen items each), on mean task F1 against dollar cost per 1,000
# items and against median per-request latency.
#
# Reads output/sidecar/jev_sidecar/cost_performance.csv, built by
# code/build_jev_sidecar.py. Writes output/figures/fig-jev-cost-latency.{pdf,png}.
#
# Cost is on a log axis, latency is linear. Cost spans 120x, and on a linear
# axis the five cheap models collapse against the left edge, which is where the
# ordering that matters happens. DeepSeek is shown at standard rates, not at the
# discounted off-peak price its run happened to be billed at.
#
# The vertical axis is task-demeaned F1, not absolute F1. Intervals on absolute
# F1 spanned 0.11 and overlapped for every pair of models, because task
# difficulty dominates and moves all nine together. Subtracting the across-model
# mean within each task removes it and leaves intervals 3.6x tighter, which is
# what makes the vertical dimension readable at all.
#
# The trade is that the axis no longer carries an absolute capability number.
# And a specific pair still should not be read off these intervals: two models
# stay correlated after demeaning, so the paired tests in the release's
# pairs.csv remain the right test, and the caption reports them.

set.seed(20260919)

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggrepel)
  library(patchwork)
})

file_arg <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", file_arg[grepl("^--file=", file_arg)][1])
if (is.na(script_file)) script_file <- "code/plot_jev_sidecar.R"
repo <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
setwd(repo)

CLARA_DARK <- "#1d4f77"
CLARA_GREY <- "grey55"

theme_clara <- function(base_size = 13) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(color = "black", face = "bold", size = base_size - 1),
      panel.border = element_rect(color = "grey70"),
      axis.ticks = element_blank(),
      legend.position = "none",
      plot.margin = margin(8, 14, 8, 8)
    )
}

save_clara <- function(plot, path_stem, width, height) {
  ggsave(paste0(path_stem, ".pdf"), plot, width = width, height = height,
         device = cairo_pdf)
  ggsave(paste0(path_stem, ".png"), plot, width = width, height = height,
         dpi = 450, bg = "white")
}

d <- read_csv("output/sidecar/jev_sidecar/cost_performance.csv",
              show_col_types = FALSE) |>
  left_join(read_csv("output/sidecar/jev_sidecar/task_demeaned.csv",
                     show_col_types = FALSE), by = "model") |>
  # DeepSeek is plotted at standard rates. The run happened to land off-peak,
  # and DeepSeek's published off-peak price is exactly half its peak price in
  # every token category, so doubling the invoiced figure recovers the standard
  # price without assuming anything about the token mix. Every other model was
  # billed at its ordinary rate already.
  mutate(focus = model == "jev-1.13.0",
         cost_plotted = coalesce(cost_per_1k_items_peak, cost_per_1k_items))
stopifnot(nrow(d) > 0, "median_latency_s" %in% names(d),
          "cost_is_invoiced" %in% names(d), !any(is.na(d$demeaned_f1)))

# Hollow markers in the cost panel are the models still priced from tokens
# rather than from a provider invoice. Drawing them identically to the invoiced
# points would assert a precision the data does not have. Latency was measured
# the same way for every model, so the right panel fills all of them.
panel <- function(mapping, xlab, xbreaks, xlabels, xlimits, mark_estimates,
                  nudge_y = 0, log_x = FALSE, batch_ticks = FALSE) {
  solid <- if (mark_estimates) filter(d, cost_is_invoiced) else d
  open <- if (mark_estimates) filter(d, !cost_is_invoiced) else d[0, ]
  batch <- if (batch_ticks) filter(d, has_batch_endpoint) else d[0, ]
  ggplot(d, mapping) +
    # Same model at batch rates, a flat 50% discount at OpenAI, Anthropic and
    # Google. DeepSeek and Jev have no batch endpoint and get no tick, which is
    # the substantive point: they cannot reach these prices.
    geom_segment(data = batch,
                 aes(x = cost_per_1k_items_batch, xend = cost_per_1k_items,
                     y = demeaned_f1, yend = demeaned_f1),
                 color = "grey80", linewidth = 0.35) +
    geom_point(data = batch, aes(x = cost_per_1k_items_batch, y = demeaned_f1),
               shape = 124, size = 1.8, color = "grey65") +
    # Zero is the panel average on any given task, so it is the reference the
    # vertical position is read against.
    geom_hline(yintercept = 0, linewidth = 0.3, color = "grey80") +
    geom_errorbar(aes(ymin = demeaned_ci_low, ymax = demeaned_ci_high),
                  width = 0, linewidth = 0.4, color = "grey75") +
    geom_point(data = solid, aes(color = focus, size = focus), shape = 16) +
    geom_point(data = open, aes(color = focus, size = focus), shape = 21,
               fill = "white", stroke = 0.9) +
    geom_text_repel(aes(label = model, color = focus, fontface = ifelse(focus, "bold", "plain")),
                    size = 2.9, min.segment.length = 0.1, seed = 20260919,
                    max.overlaps = Inf, segment.color = "grey75",
                    segment.size = 0.25, box.padding = 0.28, point.padding = 0.12,
                    force = 1, force_pull = 3, max.iter = 20000,
                    nudge_y = nudge_y) +
    scale_color_manual(values = c(`FALSE` = CLARA_GREY, `TRUE` = CLARA_DARK)) +
    scale_size_manual(values = c(`FALSE` = 2.1, `TRUE` = 3.4)) +
    scale_x_continuous(breaks = xbreaks, labels = xlabels, limits = xlimits,
                       transform = if (log_x) "log10" else "identity") +
    # Dropping the intervals collapsed the y range onto the points, leaving Jev
    # pinned in the corner. Pad it so the extreme models sit inside the panel.
    scale_y_continuous(expand = expansion(mult = 0.09)) +
    labs(x = xlab, y = "Task-demeaned F1 (0 = panel average)") +
    theme_clara()
}

# Log cost axis. Cost spans 120x, and on a linear axis the five cheap models
# collapse against the left edge, which is where most of the ordering that
# matters happens. The log axis is the only way to read that region.
p_cost <- panel(
  aes(cost_plotted, demeaned_f1),
  "Cost per 1,000 items (USD, log scale)",
  c(0.03, 0.1, 0.3, 1, 3, 10), c("$0.03", "$0.10", "$0.30", "$1", "$3", "$10"),
  c(0.026, 14),
  mark_estimates = TRUE, log_x = TRUE, batch_ticks = TRUE
)

p_lat <- panel(
  aes(median_latency_s, demeaned_f1),
  "Median latency per request (seconds)",
  seq(0, 2.5, 0.5), paste0(seq(0, 2.5, 0.5), "s"), c(0, 2.65),
  mark_estimates = FALSE
) + labs(y = NULL)

# Short by request. The full qualification lives in the sidecar summary.json
# and the writeup, per the CLARA rule that the claim belongs in surrounding
# prose rather than on the figure.
combined <- (p_cost | p_lat) +
  plot_annotation(
    caption = paste(
      "33 tasks, 100 items each. Vertical axis: each model's F1 minus the eleven-model average on the same task.",
      "Ticks mark batch rates, a 50% discount; DeepSeek and Jev have no batch option. Hollow: cost estimated, not invoiced.",
      sep = "\n"
    ),
    theme = theme(plot.caption = element_text(size = 7.5, hjust = 0, color = "grey35"))
  )

save_clara(combined, "output/figures/fig-jev-cost-latency", width = 8.6, height = 4.3)
cat("wrote output/figures/fig-jev-cost-latency.pdf and .png\n")
