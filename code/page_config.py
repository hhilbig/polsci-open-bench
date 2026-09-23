"""Load and validate the benchmark page's configuration.

One YAML file (experiments/benchmark_page.yaml) names the release the page
reports, the model roster, the site URLs and the facts the page states about the
runs. Everything that changes between updates lives there, so the page builder,
the figure builder and the release build read it instead of repeating literals.

The loader fails closed: a missing file, an unreadable path, a featured model
that is not in the release, or a ranked model without a display name all raise
rather than silently falling back.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'experiments/benchmark_page.yaml'
REQUIRED_INPUTS = ['release_dir', 'panel', 'taxonomy', 'cost_table', 'api_source',
                   'open_manifest', 'api_manifests', 'speed_runs']
REQUIRED_SITE = ['page_url', 'data_url', 'code_url', 'author_url', 'homepage_repo',
                 'target_dir', 'share_image']
REQUIRED_RUNS = ['api_window', 'batch_providers', 'gpu', 'gpu_memory', 'cluster', 'runtime',
                 'multi_gpu_note', 'max_output_tokens', 'token_limit_exception', 'not_run_open']


class PageConfig:
    """Validated view of experiments/benchmark_page.yaml with absolute paths."""

    def __init__(self, data, path):
        self.path = Path(path)
        self.page_id = data['page_id']
        self.inputs = data['inputs']
        self.site = data['site']
        self.runs = data['runs']
        self.featured_models = frozenset(data['featured_models'])
        self.labels = dict(data['labels'])

    def resolve(self, key):
        """Absolute path for one of the configured inputs."""
        return ROOT / self.inputs[key]

    @property
    def release_dir(self):
        return self.resolve('release_dir')

    @property
    def preview_dir(self):
        return self.release_dir / 'preview/llm-benchmark'

    @property
    def api_manifests(self):
        return [ROOT / p for p in self.inputs['api_manifests']]

    @property
    def homepage_dir(self):
        """Where `stage` writes the published copy."""
        return Path(self.site['homepage_repo']).expanduser() / self.site['target_dir']

    def label(self, model):
        return self.labels.get(model, model.replace('_', ' '))

    def check_against_release(self):
        """Every featured model must be scored, and every scored model needs a label."""
        release = json.loads((self.release_dir / 'release.json').read_text())
        scored = {m['model'] for m in release['models']}
        missing = sorted(self.featured_models - scored)
        if missing:
            raise ValueError(f'featured_models not in the release: {", ".join(missing)}')
        unlabelled = sorted(scored - set(self.labels))
        if unlabelled:
            raise ValueError(f'models without a label: {", ".join(unlabelled)}')
        return release


def load_page_config(path=DEFAULT_CONFIG, *, check_paths=True):
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if data.get('schema_version') != 1:
        raise ValueError(f'{path}: unsupported schema_version')
    for section, required in (('inputs', REQUIRED_INPUTS), ('site', REQUIRED_SITE), ('runs', REQUIRED_RUNS)):
        missing = [key for key in required if key not in (data.get(section) or {})]
        if missing:
            raise ValueError(f'{path}: {section} is missing {", ".join(missing)}')
    for key in ('page_id', 'featured_models', 'labels'):
        if not data.get(key):
            raise ValueError(f'{path}: {key} is missing')
    config = PageConfig(data, path)
    if check_paths:
        for key in ('release_dir', 'panel', 'taxonomy', 'api_source', 'open_manifest'):
            if not config.resolve(key).exists():
                raise ValueError(f'{path}: inputs.{key} does not exist: {config.inputs[key]}')
        for group in config.inputs['speed_runs']:
            for model, folder in group['models'].items():
                if not (ROOT / folder / model / 'run_metadata.json').is_file():
                    raise ValueError(f'{path}: no run metadata for {model} in {folder}')
    return config
