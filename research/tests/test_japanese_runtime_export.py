"""The legacy exporter must copy runtime sources, never the checkout root."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('legacy_exporter', Path(__file__).resolve().parents[1] / 'tools/export_japanese_runtime.py')
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def test_export_from_reorganized_checkout_copies_only_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            project, base, env, output = (root / n for n in ('project', 'base', 'env', 'output'))
            for folder, names in ((project, ('sakuratts/__init__.py', 'tools/synthesize_japanese.py',
                        'requirements/mlx-japanese.txt', 'docs/third-party/LICENSE', '.git/private', '.env', 'outputs/private.wav')),
                    (base, ('bin/python3.11', 'lib/python3.11/os.py')),
                    (env, ('pyvenv.cfg', 'lib/python3.11/site-packages/runtime.py'))):
                for name in names:
                    path = folder / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('fixture')
            args = SimpleNamespace(output=output, venv=env, include_install_tools=False, python_mode='bundled')
            for name, format in exporter.PACKAGE_FORMATS.items():
                package = root / name
                package.mkdir()
                (package / 'manifest.json').write_text(json.dumps({'format': format}))
                setattr(args, name + '_package', package)
            def probe(executable, **kwargs):
                if executable == env / 'bin/python':
                    return {'version': [3, 11, 15], 'base_prefix': str(base), 'base_executable': str(base / 'bin/python3.11')}
                prefix = output / ('venv' if executable == output / 'venv/bin/python' else 'python')
                return dict.fromkeys(('prefix', 'base_prefix', 'exec_prefix', 'base_exec_prefix'), str(prefix)) | {'sys_path': [], 'paths': {}}
            def create_env(*args, **kwargs):
                (output / 'venv/lib/python3.11/site-packages').mkdir(parents=True)
            with patch.object(exporter, 'PROJECT', project), patch.object(exporter.sys, 'platform', 'darwin'), \
                    patch.object(exporter, 'probe', side_effect=probe), \
                    patch.object(exporter, 'check_distributions', return_value={}), \
                    patch.object(exporter.subprocess, 'run', side_effect=create_env):
                result = exporter.export(args)
            self.assertTrue((output / 'sakuratts/__init__.py').is_file())
            self.assertTrue((output / 'tools/synthesize_japanese.py').is_file())
            self.assertTrue((output / 'requirements/mlx-japanese.txt').is_file())
            self.assertFalse((output / '.git').exists())
            self.assertFalse((output / '.env').exists())
            self.assertFalse((output / 'outputs/private.wav').exists())
            self.assertIn('resources/reference/manifest.json', result['files'])
