#!/usr/bin/env Rscript

# Does Jev fall further behind as tasks get harder?
#
# One point per task. Horizontal position is how well the other eight models did
# on that task, which is the task's difficulty measured without Jev in it.
# Vertical position is Jev's own F1. The diagonal is where Jev would match the
# panel exactly, so vertical distance below it is Jev's gap on that task.
#
# The question is whether the fitted line is parallel to the diagonal. A slope
# of 1 means a constant gap; steeper than 1 means the gap widens on hard tasks.
#
# Reads output/sidecar/jev_sidecar/complexity_{points,slopes}.csv, built by
# code/build_jev_sidecar.py. Writes output/figures/fig-jev-complexity.{pdf,png}.

set.seed(20260920)

suppressPackageStartupMessages({
  library(tidyverse)
})

file_arg <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", file_arg[grepl("^--file=", file_arg)][1])
if (is.na(script_file)) script_file <- "code/plot_jev_complexity.R"
repo <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
setwd(repo)

CLARA_DARK <- "#1d4f77"

pts <- read_csv("output/sidecar/jev_sidecar/complexity_points.csv", show_col_types = FALSE)
slopes <- read_csv("output/sidecar/jev_sidecar/complexity_slopes.csv", show_col_types = FALSE)
raw <- slopes[slopes$test == "slope_vs_panel_raw", ]
stopifnot(nrow(pts) > 0, nrow(raw) == 1)

p <- ggplot(pts, aes(others_mean_f1, focus_f1)) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.4, color = "grey70") +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE, linewidth = 0.9,
              color = CLARA_DARK, fill = CLARA_DARK, alpha = 0.12) +
  geom_point(size = 2, color = CLARA_DARK, alpha = 0.85) +
  annotate("text", x = 0.38, y = 0.97, hjust = 0, size = 3.1, color = CLARA_DARK,
           label = sprintf("slope %.2f [%.2f, %.2f]", raw$slope, raw$ci_low, raw$ci_high)) +
  coord_equal(xlim = c(0.35, 1), ylim = c(0.35, 1)) +
  labs(x = "Mean F1 of the other eight models (task difficulty)",
       y = "Jev 1.13 F1",
       caption = paste(
         "One point per task, 33 tasks, 100 items each. The diagonal is where Jev matches the panel.",
         "The fitted slope does not differ from 1, so Jev's gap is a roughly constant offset rather than",
         "one that widens on harder tasks. On a logit scale the slope is 0.89 [0.77, 0.98], which would",
         "instead put Jev slightly further behind on the easiest tasks.",
         sep = "\n")) +
  theme_bw(base_size = 13) +
  theme(
    panel.grid = element_blank(),
    panel.border = element_rect(color = "grey70"),
    axis.ticks = element_blank(),
    legend.position = "none",
    plot.caption = element_text(size = 7.5, hjust = 0, color = "grey35"),
    plot.margin = margin(8, 14, 8, 8)
  )

ggsave("output/figures/fig-jev-complexity.pdf", p, width = 5.6, height = 5.6,
       device = cairo_pdf)
ggsave("output/figures/fig-jev-complexity.png", p, width = 5.6, height = 5.6,
       dpi = 450, bg = "white")
cat("wrote output/figures/fig-jev-complexity.pdf and .png\n")
