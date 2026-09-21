#!/usr/bin/env Rscript

# Cost against performance twice: once at real-time rates, once at batch rates.
#
# Both panels share the same axes, so the only thing that moves between them is
# price. Models with a batch endpoint slide left by exactly half. Models without
# one do not move, because their standard rate is the only rate they have, and
# that is the price you would actually pay for an offline job. Those are marked
# with an asterisk rather than dropped, since their staying put is the point.
#
# Reads output/sidecar/jev_sidecar/cost_performance.csv, built by
# code/build_jev_sidecar.py. Writes output/figures/fig-cost-realtime-batch.{pdf,png}.

set.seed(20260921)

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggrepel)
  library(patchwork)
})

file_arg <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", file_arg[grepl("^--file=", file_arg)][1])
if (is.na(script_file)) script_file <- "code/plot_cost_panels.R"
repo <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
setwd(repo)

CLARA_DARK <- "#1d4f77"
CLARA_GREY <- "grey55"

theme_clara <- function(base_size = 13) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid = element_blank(),
      strip.background = element_blank(),
      panel.border = element_rect(color = "grey70"),
      axis.ticks = element_blank(),
      legend.position = "none",
      plot.margin = margin(8, 14, 8, 8)
    )
}

d <- read_csv("output/sidecar/jev_sidecar/cost_performance.csv",
              show_col_types = FALSE) |>
  left_join(read_csv("output/sidecar/jev_sidecar/task_demeaned.csv",
                     show_col_types = FALSE), by = "model") |>
  mutate(focus = model == "jev-1.13.0",
         # No batch endpoint means the standard rate is what an offline job
         # costs, so the model appears in the batch panel at that price.
         cost_batch = coalesce(cost_per_1k_items_batch, cost_per_1k_items),
         label_batch = if_else(has_batch_endpoint, model, paste0(model, "*")))
stopifnot(nrow(d) > 0, "cost_per_1k_items_batch" %in% names(d),
          !any(is.na(d$demeaned_f1)))

BREAKS <- c(0.03, 0.1, 0.3, 1, 3, 10)
LABELS <- c("$0.03", "$0.10", "$0.30", "$1", "$3", "$10")
LIMITS <- c(0.026, 16)

# At batch rates luna falls to $0.081 and lands almost on top of
# deepseek-v4-flash at $0.101, and they are within 0.005 F1 of each other too.
# Vertical-only repulsion has nowhere to put them, so those two labels are
# placed by hand in that panel only.
panel <- function(xvar, labelvar, xlab, nudges = c()) {
  nudge <- unname(ifelse(d$model %in% names(nudges), nudges[d$model], 0))
  ggplot(d, aes(.data[[xvar]], demeaned_f1)) +
    geom_hline(yintercept = 0, linewidth = 0.3, color = "grey80") +
    geom_errorbar(aes(ymin = demeaned_ci_low, ymax = demeaned_ci_high),
                  width = 0, linewidth = 0.4, color = "grey75") +
    geom_point(data = filter(d, cost_is_invoiced),
               aes(color = focus, size = focus), shape = 16) +
    geom_point(data = filter(d, !cost_is_invoiced),
               aes(color = focus, size = focus), shape = 21,
               fill = "white", stroke = 0.9) +
    geom_text_repel(aes(label = .data[[labelvar]], color = focus,
                        fontface = ifelse(focus, "bold", "plain")),
                    # Vertical-only repulsion. The x position carries meaning
                    # and the labels are long, so letting them drift sideways on
                    # a log axis put them over neighbouring points.
                    size = 2.9, min.segment.length = 0.1, seed = 20260921,
                    max.overlaps = Inf, segment.color = "grey75",
                    segment.size = 0.25, box.padding = 0.35, point.padding = 0.15,
                    direction = "y", force = 8, force_pull = 0.3, max.iter = 40000,
                    nudge_y = nudge) +
    scale_color_manual(values = c(`FALSE` = CLARA_GREY, `TRUE` = CLARA_DARK)) +
    scale_size_manual(values = c(`FALSE` = 2.1, `TRUE` = 3.4)) +
    scale_x_continuous(transform = "log10", breaks = BREAKS,
                       labels = LABELS, limits = LIMITS) +
    scale_y_continuous(expand = expansion(mult = 0.16)) +
    labs(x = xlab, y = "Task-demeaned F1 (0 = panel average)") +
    theme_clara()
}

combined <- (panel("cost_per_1k_items", "model",
                   "Real-time cost per 1,000 items (USD, log scale)") |
             (panel("cost_batch", "label_batch",
                    "Batch cost per 1,000 items (USD, log scale)",
                    nudges = c("gpt-5.6-luna" = 0.012,
                               "deepseek-v4-flash" = -0.012)) + labs(y = NULL))) +
  plot_annotation(
    caption = paste(
      "33 tasks, 100 items each. Vertical axis: each model's F1 minus the eleven-model average on the same task.",
      "Same scales in both panels, so only price moves. * no batch option, shown at its standard rate. Hollow: estimated cost.",
      sep = "\n"),
    theme = theme(plot.caption = element_text(size = 7.5, hjust = 0, color = "grey35")))

ggsave("output/figures/fig-cost-realtime-batch.pdf", combined, width = 9.2, height = 4.6,
       device = cairo_pdf)
ggsave("output/figures/fig-cost-realtime-batch.png", combined, width = 9.2, height = 4.6,
       dpi = 450, bg = "white")
cat("wrote output/figures/fig-cost-realtime-batch.pdf and .png\n")
