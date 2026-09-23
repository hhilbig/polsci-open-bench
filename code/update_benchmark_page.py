"""Build and stage the public benchmark page.

Everything that changes between updates is in experiments/benchmark_page.yaml.
This is the only entry point; the steps below call the existing builders.

    python3 code/update_benchmark_page.py all      # release, figures, page, check
    python3 code/update_benchmark_page.py page     # rebuild the page only
    python3 code/update_benchmark_page.py stage    # copy it into the homepage repo

`stage` copies the page and the figures it references, removes the preview-only
markers, and adds the homepage link and sitemap entry if they are missing. It
stops before committing: review the printed git status, then commit and push in
the homepage repo yourself.

Runbook with the full sequence: docs/update_benchmark_page.md
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from page_config import ROOT, load_page_config

# Markers that exist only in the local preview and must not be published.
PREVIEW_MARKERS = [
    (r'<meta name="robots" content="noindex">', ''),
    (r'<p class="notice">Local preview, not published\..*?</p>', ''),
    (r'\. This is a local preview\. Publication requires approval\.</footer>', '.</footer>'),
]


def run(command, **kwargs):
    print('$', ' '.join(str(c) for c in command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT, **kwargs)


def build_release(config):
    """Rebuild the scored release: API manifests, the release itself, then costs."""
    run([sys.executable, 'code/refresh_gemini.py', 'manifest'])
    command = [sys.executable, 'code/build_refresh_release.py', '--active-only',
               '--open-manifest', config.inputs['open_manifest'],
               '--output', config.inputs['release_dir']]
    for manifest in config.inputs['api_manifests']:
        command += ['--api-manifest', manifest]
    run(command)
    run([sys.executable, 'code/build_jev_sidecar.py'])


def build_figures(config):
    run([sys.executable, 'code/build_refresh_figures.py', '--release-dir', config.inputs['release_dir']])


def build_page(config):
    run([sys.executable, 'code/build_refresh_preview.py', '--release-dir', config.inputs['release_dir']])


def check(config):
    """Re-verify the built page against the release and run the page tests."""
    run([sys.executable, 'code/build_refresh_preview.py', '--release-dir', config.inputs['release_dir'],
         '--verify-only'])
    run([sys.executable, '-m', 'unittest', '-q',
         'tests.test_page_config', 'tests.test_update_benchmark_page',
         'tests.test_refresh_preview', 'tests.test_refresh_release'])


def published_page(preview_dir):
    """The page as published: the preview markers removed."""
    page = (preview_dir / 'index.html').read_text()
    for pattern, replacement in PREVIEW_MARKERS:
        page, count = re.subn(pattern, replacement, page, flags=re.S)
        if count != 1:
            raise ValueError(f'preview marker {pattern!r} appears {count} times, expected once')
    return page


def referenced_figures(page):
    """Only the figure files the page actually shows, plus the share image."""
    return sorted({f'{stem}.{ext}' for stem, ext in re.findall(r'figures/(fig-[a-z0-9-]+)\.(svg|png)', page)})


def add_once(text, marker, addition, where):
    """Insert `addition` unless `marker` is already there; returns (text, added)."""
    if marker in text:
        return text, False
    if where not in text:
        raise ValueError(f'cannot place the entry: {where!r} not found')
    return text.replace(where, addition, 1), True


def stage(config, target=None):
    preview = config.preview_dir
    destination = Path(target).expanduser() if target else config.homepage_dir
    page = published_page(preview)
    figures = referenced_figures(page)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'index.html').write_text(page)
    for name in ('styles.css', 'benchmark.js'):
        shutil.copy(preview / name, destination / name)
    figure_dir = destination / 'figures'
    if figure_dir.exists():
        shutil.rmtree(figure_dir)
    figure_dir.mkdir()
    for name in figures:
        shutil.copy(preview / 'figures' / name, figure_dir / name)
    site = destination.parent
    added = []
    index = site / 'index.html'
    data_section = '<h2 id="data">Data</h2>\n      <ol start="1">\n'
    if index.exists():
        entry = (f'        <li>\n          <b><a href="{config.site["target_dir"]}/">Political Science LLM '
                 'Benchmark</a></b>\n          2026. Matched evaluation of 31 open-weight and commercial '
                 'language models on 33\n          political science text-coding tasks, with cost and '
                 'generation speed.\n        </li>\n')
        text, did = add_once(index.read_text(), f'href="{config.site["target_dir"]}/"',
                             data_section + entry, data_section)
        if did:
            index.write_text(text)
            added.append('homepage Data entry')
    sitemap = site / 'sitemap.xml'
    if sitemap.exists():
        text, did = add_once(sitemap.read_text(), config.site['page_url'],
                             f'''  <url>
    <loc>{config.site['page_url']}</loc>
    <priority>0.8</priority>
  </url>
</urlset>''', '</urlset>')
        if did:
            sitemap.write_text(text)
            added.append('sitemap entry')
    print(f'staged {destination} ({len(figures)} figures)')
    print('added:', ', '.join(added) if added else 'nothing new, the links were already there')
    if (site / '.git').exists():
        subprocess.run(['git', 'status', '--short'], cwd=site, check=False)
        print('\nReview the diff above, then commit and push in', site)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['release', 'figures', 'page', 'check', 'stage', 'all'])
    parser.add_argument('--config', default=None, help='defaults to experiments/benchmark_page.yaml')
    parser.add_argument('--to', default=None, help='stage into this directory instead of the homepage repo')
    args = parser.parse_args()
    config = load_page_config(args.config) if args.config else load_page_config()
    config.check_against_release()
    steps = {'release': build_release, 'figures': build_figures, 'page': build_page, 'check': check}
    if args.step == 'stage':
        stage(config, args.to)
    elif args.step == 'all':
        for name in ('release', 'figures', 'page', 'check'):
            steps[name](config)
        print('\nBuilt and checked. Run "stage" to copy the page into the homepage repo.')
    else:
        steps[args.step](config)


if __name__ == '__main__':
    main()
