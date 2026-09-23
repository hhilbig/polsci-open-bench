# Updating the public benchmark page

The page at <https://www.hannohilbig.com/llm-benchmark/> is generated from the
frozen release. Everything that changes between updates is in one file,
[`experiments/benchmark_page.yaml`](../experiments/benchmark_page.yaml), and one
command runs the pipeline.

## The short version

```bash
python3 code/update_benchmark_page.py all     # release, figures, page, checks
python3 code/update_benchmark_page.py stage   # copy into the homepage repo
```

`stage` stops before committing. It prints the homepage repo's `git status`;
review it, then commit and push there yourself. That repo usually holds
unrelated CV work, so commit only `index.html`, `sitemap.xml` and
`llm-benchmark/`.

## What each step does

| Step | What runs | Writes |
|---|---|---|
| `release` | `refresh_gemini.py manifest`, `build_refresh_release.py --active-only`, `build_jev_sidecar.py` | the release directory and the cost table |
| `figures` | `build_refresh_figures.py`, which calls the R renderer | `preview/llm-benchmark/figures/` |
| `page` | `build_refresh_preview.py` | `preview/llm-benchmark/index.html` and the download files |
| `check` | the page verifier and the page tests | nothing |
| `stage` | copies the page, styles, script and referenced figures | the homepage repo |

Each step runs on its own, so a text-only change needs `page` then `stage`.

## Making a change

**Change the wording.** The prose is in `code/build_refresh_preview.py`, in the
one HTML string inside `render()`. Figure captions are in
`code/render_refresh_paper_figures.R`, next to the code that draws each figure.
Rebuild with `page` (or `figures` for a caption) and `stage`.

**Add or drop a model.** Add its predictions to the release inputs, then in
`experiments/benchmark_page.yaml` add a `labels` entry, and a `featured_models`
entry if it belongs above the fold. Run `all`. The build stops if a featured
model is missing from the release or a scored model has no label.

**New release of the whole panel.** Point `inputs.release_dir` and the other
input paths at the new directories, update `site.data_url` to the new folder on
GitHub, and update the `runs:` block, which is what the page states under "Where
the models ran". Run `all`, then `stage`.

**New speed runs.** Add a group under `inputs.speed_runs` with a label, the
concurrency in plain words, and one entry per model pointing at the run
directory. Models in a group must share texts, settings and GPU model, which the
build checks; each group becomes one panel of the speed figure.

## What is checked automatically

- Every configured path exists, every featured model is in the release, and
  every scored model has a label (`code/page_config.py`).
- The page reproduces the release's numbers, and the figures were built from the
  same release (`build_refresh_preview.py: verify`).
- Staging strips exactly three preview markers: the `noindex` tag, the preview
  notice and the preview sentence in the footer. It publishes only the figure
  files the page references, and running it twice changes nothing
  (`tests/test_update_benchmark_page.py`).

## Numbers stated in the page and README

Counts and comparisons on the page are computed at build time, so they cannot go
stale. The README repeats a few of them as prose; check that section after a
release that moves the headline numbers.

## Where the files live

- Config: `experiments/benchmark_page.yaml`
- Entry point: `code/update_benchmark_page.py`
- Page text and layout: `code/build_refresh_preview.py`
- Figure data: `code/build_refresh_figures.py`; drawing: `code/render_refresh_paper_figures.R`
- Release build: `code/build_refresh_release.py`; costs: `code/build_jev_sidecar.py`
