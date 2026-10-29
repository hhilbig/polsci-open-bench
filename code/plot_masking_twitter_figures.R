#!/usr/bin/env Rscript
# Social-media figures for the party-masking experiment
# (code/analyze_masking_experiment.py). Run from the repository root after the
# analysis script has run for both the open-weight models (default) and the
# commercial models (--run api). Writes PDF and PNG to
# output/sidecar/task_ext_20261010_masked/twitter/.

set.seed(20261010)

library(dplyr)
library(ggplot2)
library(readr)
library(scales)

input_dir <- "output/sidecar/task_ext_20261010_masked/analysis"
api_dir <- "output/sidecar/task_ext_20261011_masked_api/analysis"
out_dir <- "output/sidecar/task_ext_20261010_masked/twitter"
dir.create(out_dir, showWarnings = FALSE)

DARK <- "#1d4f77"
GREY <- "grey55"
LIGHT <- "grey85"

theme_clara_twitter <- function(fontsize = 13) {
  theme_bw(base_size = fontsize, base_family = "Helvetica") +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(color = "grey70", linewidth = 0.45),
      axis.ticks = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = fontsize - 1, hjust = 0),
      legend.position = "none",
      plot.title.position = "plot",
      plot.caption.position = "plot",
      plot.title = element_text(face = "bold", size = fontsize + 6),
      plot.subtitle = element_text(color = "grey30", lineheight = 1.15, margin = margin(b = 12)),
      plot.caption = element_text(
        color = "grey35", size = fontsize - 3, hjust = 0,
        lineheight = 1.15, margin = margin(t = 12)
      ),
      plot.margin = margin(12, 34, 10, 10)
    )
}

save_fig <- function(plot, stem, width, height) {
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, device = cairo_pdf)
  ggsave(png_path, plot, width = width, height = height, dpi = 300, bg = "white")
  stopifnot(file.info(pdf_path)$size > 5000, file.info(png_path)$size > 10000)
}

benoit <- read_csv(file.path(input_dir, "benoit_masking.csv"), show_col_types = FALSE)
tweets <- bind_rows(
  read_csv(file.path(api_dir, "tweet_masking.csv"), show_col_types = FALSE) %>%
    mutate(group = "Commercial"),
  read_csv(file.path(input_dir, "tweet_masking.csv"), show_col_types = FALSE) %>%
    mutate(group = "Open-weight")
)
tweet_items <- read_csv(file.path(api_dir, "tweet_items.csv"), show_col_types = FALSE)

model_order <- benoit %>%
  filter(domain == "economic", passage == "named") %>%
  arrange(con_lab_error_gap_orig) %>%
  pull(model)

# Figure 1: Republican-minus-Democratic error gap against the human raters
# (model minus mean human rating), tweets that name someone, as written and
# masked; tweets without names as the reference.
excess <- bind_rows(
  tweets %>% filter(passage == "named") %>%
    transmute(group, model, version = "Names shown", est = rep_dem_error_gap_orig,
              lo = gap_orig_lo, hi = gap_orig_hi),
  tweets %>% filter(passage == "named") %>%
    transmute(group, model, version = "Names masked", est = rep_dem_error_gap_mask,
              lo = gap_mask_lo, hi = gap_mask_hi),
  tweets %>% filter(passage != "named") %>%
    transmute(group, model, version = "Tweets without names", est = rep_dem_error_gap_orig,
              lo = gap_orig_lo, hi = gap_orig_hi)
)
excess_order <- excess %>%
  filter(version == "Names shown") %>%
  arrange(group, est) %>%
  pull(model)
excess <- excess %>%
  mutate(
    model = factor(model, levels = rev(excess_order)),
    version = factor(version, levels = c("Names shown", "Names masked", "Tweets without names")),
    y = as.numeric(model) + c(0.25, 0, -0.25)[as.integer(version)]
  )
fig_ex <- ggplot(excess, aes(est, y, color = version, shape = version)) +
  geom_vline(xintercept = 0, color = "grey60", linewidth = 0.5) +
  geom_segment(aes(x = lo, xend = hi, yend = y), linewidth = 0.8) +
  geom_point(size = 2.9, fill = "white", stroke = 1.1) +
  facet_grid(group ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_color_manual(values = c("Names shown" = DARK, "Names masked" = GREY,
                                "Tweets without names" = "grey65")) +
  scale_shape_manual(values = c("Names shown" = 16, "Names masked" = 16,
                                "Tweets without names" = 21)) +
  scale_y_continuous(breaks = seq_along(excess_order), labels = rev(excess_order),
                     expand = expansion(add = 0.5)) +
  scale_x_continuous(expand = expansion(mult = 0.04)) +
  labs(
    title = "With names in the tweet, LLMs read more party into it",
    subtitle = paste0(
      "How much further apart the model places Republican and Democratic tweets\n",
      "than human raters did (points on a 0-100 left-right scale)"
    ),
    x = "Extra Republican-Democratic distance, model minus human raters (points)", y = NULL,
    caption = paste0(
      "144 tweets by members of Congress that name a party or politician (Le Mens and Gallego). Human raters saw the tweets\n",
      "with the names. Masked: names replaced by [PARTY] or [POLITICIAN], everything else identical. Tweets without names:\n",
      "89 (commercial) and 356 (open-weight) sampled tweets. Not-political answers scored as 50. 95% intervals, bootstrap over tweets."
    )
  ) +
  theme_clara_twitter() +
  theme(
    strip.placement = "outside", strip.text.y.left = element_text(angle = 0, vjust = 1),
    legend.position = "top", legend.justification = "left", legend.title = element_blank(),
    legend.text = element_text(size = 12), legend.margin = margin(0, 0, 0, 0)
  )
save_fig(fig_ex, "01_tweets_excess_separation", width = 9.5, height = 6.4)

# Figure 2: two example tweets, commercial models with the name shown and masked
# against the mean human rating.
examples <- tibble(
  item_id = c("lemens_tweet_150", "lemens_tweet_226"),
  text = c(
    paste0("Democrat: \"@DeptVetAffairs has implemented @VetAffairsDems legislation ensuring\n",
           "all veterans experiencing a suicidal crisis can receive no-cost emergency\n",
           "mental health care!\""),
    paste0("Republican: \"Grocery prices increased by 10% for the NINTH month in a row.\n",
           "Bottom line: Democrats' failed economic policies are hurting you.\"")
  )
)
ex <- tweet_items %>%
  inner_join(examples, by = "item_id") %>%
  select(text, model, human_position_mean, score_orig, score_mask)
stopifnot(nrow(ex) == 8)
ex_long <- bind_rows(
  ex %>% distinct(text, score = human_position_mean) %>% mutate(row = "Human raters (name shown)"),
  ex %>% transmute(text, model, score = score_orig, row = "LLMs, name shown"),
  ex %>% transmute(text, model, score = score_mask, row = "LLMs, name masked")
) %>%
  mutate(
    row = factor(row, levels = rev(c("Human raters (name shown)", "LLMs, name shown",
                                     "LLMs, name masked"))),
    text = factor(text, levels = examples$text)
  )
ex_mean <- ex_long %>% group_by(text, row) %>% summarise(score = mean(score), .groups = "drop")

fig_example <- ggplot(ex_long, aes(score, row, color = row)) +
  geom_vline(data = ex_mean %>% filter(row == "Human raters (name shown)"),
             aes(xintercept = score), color = "grey80", linewidth = 0.5) +
  geom_point(data = ex_long %>% filter(!is.na(model)), size = 2, alpha = 0.45) +
  geom_point(data = ex_mean, size = 4.2, shape = 18) +
  geom_text(data = ex_mean, aes(label = round(score)), vjust = -1, size = 3.8,
            fontface = "bold", family = "Helvetica") +
  facet_wrap(~ text, ncol = 1) +
  scale_color_manual(values = c("Human raters (name shown)" = "black",
                                "LLMs, name shown" = DARK, "LLMs, name masked" = GREY)) +
  scale_x_continuous(limits = c(0, 100), breaks = c(0, 25, 50, 75, 100),
                     labels = c("0\nleft", "25", "50", "75", "100\nright")) +
  scale_y_discrete(expand = expansion(add = 0.7)) +
  labs(
    title = "Same tweet, name masked: the LLMs land where people did",
    x = NULL, y = NULL,
    caption = paste0(
      "Large marks: mean of four commercial models (Gemini 3.8 Flash, DeepSeek V4 Pro, GPT-5.6 Terra, Claude Sonnet 5);\n",
      "small marks: each model. Masked: the name replaced by [PARTY]. Human raters: mean of 18-20 ratings (Le Mens and Gallego)."
    )
  ) +
  theme_clara_twitter() +
  theme(strip.text = element_text(face = "plain", size = 11.5, lineheight = 1.15,
                                  margin = margin(t = 8, b = 4)))
save_fig(fig_example, "02_example_tweets", width = 9, height = 6.2)

# Figure 3: Conservative-minus-Labour error gap in passages that name a party or
# politician, as written and with names masked.
gap <- benoit %>%
  filter(domain == "economic", passage == "named") %>%
  select(model, con_lab_error_gap_orig, gap_orig_lo, gap_orig_hi,
         con_lab_error_gap_mask, gap_mask_lo, gap_mask_hi)
gap_long <- bind_rows(
  gap %>% transmute(model, version = "Names shown", est = con_lab_error_gap_orig,
                    lo = gap_orig_lo, hi = gap_orig_hi),
  gap %>% transmute(model, version = "Names masked", est = con_lab_error_gap_mask,
                    lo = gap_mask_lo, hi = gap_mask_hi)
) %>%
  mutate(
    model = factor(model, levels = model_order),
    version = factor(version, levels = c("Names masked", "Names shown")),
    y = as.numeric(model) + if_else(version == "Names shown", 0.15, -0.15)
  )
labels_1 <- gap_long %>% filter(model == first(model_order))

fig1 <- ggplot(gap_long, aes(est, y, color = version)) +
  geom_vline(xintercept = 0, color = "grey60", linewidth = 0.5) +
  geom_segment(aes(x = lo, xend = hi, yend = y), linewidth = 0.8) +
  geom_point(size = 3.2) +
  geom_text(
    data = labels_1, aes(x = hi + 0.03, label = version), hjust = 0, size = 4,
    fontface = "bold", family = "Helvetica"
  ) +
  scale_color_manual(values = c("Names shown" = DARK, "Names masked" = GREY)) +
  scale_y_continuous(breaks = seq_along(model_order), labels = model_order,
                     expand = expansion(add = 0.6)) +
  labs(
    title = "Party names pull Labour and Conservative codes apart",
    subtitle = paste0(
      "How much further right the model codes Conservative than Labour manifesto sentences\n",
      "that experts placed at the same economic position (steps on a 5-point scale)"
    ),
    x = "Conservative minus Labour (scale steps)", y = NULL,
    caption = paste0(
      "192 of 500 sampled sentences whose passage names a party or politician. Masked: names replaced by [PARTY] or [POLITICIAN],\n",
      "everything else identical. Experts saw the same passages, names included, but not which manifesto they came from\n",
      "(Benoit et al. 2016). Gap net of the expert code and expert mean. 95% intervals, bootstrap over manifestos."
    )
  ) +
  theme_clara_twitter()
save_fig(fig1, "03_party_gap_named", width = 9, height = 5.4)
