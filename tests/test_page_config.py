"""The page config must fail closed: a bad path or roster stops the build."""
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from page_config import DEFAULT_CONFIG, load_page_config  # noqa: E402


def config_data():
    return yaml.safe_load(DEFAULT_CONFIG.read_text())


def write(data):
    handle = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
    yaml.safe_dump(data, handle)
    handle.close()
    return handle.name


def test_real_config_loads_and_matches_the_release():
    config = load_page_config()
    release = config.check_against_release()
    scored = {m['model'] for m in release['models']}
    assert config.featured_models <= scored
    assert set(config.labels) >= scored
    assert config.release_dir.is_dir() and config.preview_dir.name == 'llm-benchmark'
    assert config.site['page_url'].endswith('/')


def test_missing_input_path_is_rejected():
    data = config_data()
    data['inputs']['panel'] = 'output/sidecar/does-not-exist/panel.json'
    with unittest.TestCase().assertRaisesRegex(ValueError, 'inputs.panel does not exist'):
        load_page_config(write(data))


def test_missing_section_key_is_rejected():
    data = config_data()
    del data['runs']['gpu']
    with unittest.TestCase().assertRaisesRegex(ValueError, 'runs is missing gpu'):
        load_page_config(write(data))


def test_unknown_schema_version_is_rejected():
    data = config_data()
    data['schema_version'] = 99
    with unittest.TestCase().assertRaisesRegex(ValueError, 'schema_version'):
        load_page_config(write(data))


def test_featured_model_outside_the_release_is_rejected():
    data = config_data()
    data['featured_models'].append('model-that-was-never-run')
    config = load_page_config(write(data))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'featured_models not in the release'):
        config.check_against_release()


def test_scored_model_without_a_label_is_rejected():
    data = config_data()
    data['labels'].pop('claude-opus-5')
    config = load_page_config(write(data))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'models without a label'):
        config.check_against_release()


def test_speed_run_without_metadata_is_rejected():
    data = config_data()
    data['inputs']['speed_runs'][0]['models']['glm4_7_flash'] = 'output/sidecar/not-a-run'
    with unittest.TestCase().assertRaisesRegex(ValueError, 'no run metadata for glm4_7_flash'):
        load_page_config(write(data))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in globals().items() if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
