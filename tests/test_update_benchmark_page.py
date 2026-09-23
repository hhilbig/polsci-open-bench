"""Staging must publish exactly the page, strip the preview markers, and repeat safely."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import update_benchmark_page as updater  # noqa: E402
from page_config import load_page_config  # noqa: E402

CONFIG = load_page_config()


def test_published_page_strips_every_preview_marker():
    page = updater.published_page(CONFIG.preview_dir)
    assert 'noindex' not in page
    assert 'Local preview, not published' not in page
    assert 'Publication requires approval' not in page
    # The rest of the page survives.
    assert '<h2 id="hardware">Where the models ran</h2>' in page
    assert CONFIG.site['page_url'] in page


def test_published_page_fails_when_a_marker_is_missing():
    with tempfile.TemporaryDirectory() as folder:
        preview = Path(folder)
        page = (CONFIG.preview_dir / 'index.html').read_text().replace('<meta name="robots" content="noindex">', '')
        (preview / 'index.html').write_text(page)
        with unittest.TestCase().assertRaisesRegex(ValueError, 'appears 0 times'):
            updater.published_page(preview)


def test_only_referenced_figures_are_published():
    page = updater.published_page(CONFIG.preview_dir)
    figures = updater.referenced_figures(page)
    assert figures, 'the page shows no figures'
    for name in figures:
        assert (CONFIG.preview_dir / 'figures' / name).is_file()
        assert f'figures/{name}' in page
    # Figures left over from earlier builds stay behind.
    on_disk = {p.name for p in (CONFIG.preview_dir / 'figures').glob('fig-*')}
    assert set(figures) <= on_disk and len(figures) < len(on_disk)


def test_stage_writes_the_page_and_is_idempotent():
    with tempfile.TemporaryDirectory() as folder:
        site = Path(folder)
        (site / 'index.html').write_text('<h2 id="data">Data</h2>\n      <ol start="1">\n      </ol>\n')
        (site / 'sitemap.xml').write_text('<urlset>\n</urlset>')
        target = site / CONFIG.site['target_dir']
        updater.stage(CONFIG, target)
        published = (target / 'index.html').read_text()
        assert 'noindex' not in published
        assert (target / 'styles.css').is_file() and (target / 'benchmark.js').is_file()
        names = sorted(p.name for p in (target / 'figures').iterdir())
        assert names == updater.referenced_figures(published)
        assert (site / 'index.html').read_text().count('href="llm-benchmark/"') == 1
        assert (site / 'sitemap.xml').read_text().count(CONFIG.site['page_url']) == 1
        before = {p: p.read_bytes() for p in site.rglob('*') if p.is_file()}
        updater.stage(CONFIG, target)
        after = {p: p.read_bytes() for p in site.rglob('*') if p.is_file()}
        assert before == after, 'staging twice changed the site'


def test_stage_leaves_other_site_files_alone():
    with tempfile.TemporaryDirectory() as folder:
        site = Path(folder)
        (site / 'index.html').write_text('<h2 id="data">Data</h2>\n      <ol start="1">\n      </ol>\n')
        keep = site / 'papers'
        keep.mkdir()
        (keep / 'draft.pdf').write_bytes(b'unchanged')
        updater.stage(CONFIG, site / CONFIG.site['target_dir'])
        assert (keep / 'draft.pdf').read_bytes() == b'unchanged'


def test_add_once_refuses_an_unknown_anchor():
    with unittest.TestCase().assertRaisesRegex(ValueError, 'not found'):
        updater.add_once('<html></html>', 'marker', 'entry', '<h2 id="data">')


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in globals().items() if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
