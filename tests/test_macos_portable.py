"""Portable Mac dependency selection, binary closure, permissions and launch isolation."""

from email.parser import Parser
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import tomllib
import unittest
from unittest.mock import patch
import zipfile

from test_portable_builder import builder
from sakuratts.TTS_infer_pack.config import read_inference_configuration


class MacPortableTests(unittest.TestCase):
    def test_macos_recipe_selects_cpu_frontend_ort_and_no_cuda_or_torch(self):
        root = Path(__file__).resolve().parents[1]
        recipe = builder.read_recipe(root / 'packaging/recipes/macos-mlx-ja.toml')
        project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
        names = {builder.Requirement(x).name for x in builder.main_requirements(project, recipe)}
        self.assertTrue({'mlx', 'mlx-metal', 'onnxruntime', 'pyopenjtalk-plus', 'fastapi'} <= names)
        self.assertFalse({'torch', 'cupy-cuda12x', 'onnxruntime-directml'} & names)
        installed = {name: (None, Parser().parsestr('Name: ' + name + '\nVersion: 1\n' + deps))
                     for name, deps in [('app', 'Requires-Dist: metal; sys_platform == "darwin"\n'), ('metal', '')]}
        self.assertEqual(builder.dependency_names(installed, ['app', 'windows; sys_platform == "win32"'],
                                                  '3.11', 'macos-arm64'), ['app', 'metal'])

    def test_wheel_selection_rejects_developer_os_and_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for tag, valid in [('cp311-cp311-macosx_14_0_arm64', True), ('py3-none-any', True),
                               ('cp311-cp311-macosx_26_0_arm64', False), ('cp311-cp311-macosx_14_0_x86_64', False),
                               ('cp312-cp312-macosx_14_0_arm64', False), ('cp310-abi3-macosx_14_0_arm64', True)]:
                (root / 'WHEEL').write_text('Tag: ' + tag + '\n')
                with self.subTest(tag=tag):
                    if valid:
                        builder.macos.wheel_compatible(root, '14.0', '3.11')
                    else:
                        with self.assertRaisesRegex(ValueError, 'compatible'):
                            builder.macos.wheel_compatible(root, '14.0', '3.11')

    def binary(self, path, dependency, major=14, cpu=0x100000c):
        name = dependency.encode() + b'\0'
        command = struct.pack('<IIIIII', 0xc, 24 + len(name), 24, 0, 0, 0) + name
        minimum = struct.pack('<IIIIII', 0x32, 24, 1, major << 16, major << 16, 0)
        path.write_bytes(struct.pack('<IiiIIIII', 0xfeedfacf, cpu, 0, 6, 2,
                                     len(command) + len(minimum), 0, 0) + command + minimum)

    def test_binary_audit_rejects_external_libraries_newer_os_and_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'binary'
            for dependency, major, cpu, error in [('/usr/lib/libSystem.B.dylib', 14, 0x100000c, None),
                    ('/opt/homebrew/lib/libaudio.dylib', 14, 0x100000c, 'outside'),
                    ('@rpath/missing.dylib', 14, 0x100000c, 'outside'),
                    ('/usr/lib/libSystem.B.dylib', 26, 0x100000c, 'newer'),
                    ('/usr/lib/libSystem.B.dylib', 14, 0x1000007, 'arm64')]:
                with self.subTest(dependency=dependency, major=major, cpu=cpu):
                    self.binary(source, dependency, major, cpu)
                    plan = builder.Plan()
                    plan.add(source, 'bin/python3', 'cpython')
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            builder.macos.audit(plan, '14.0', 'bin/python3')
                    else:
                        self.assertEqual(builder.macos.audit(plan, '14.0', 'bin/python3')['mach_o_files'], 1)

    def test_mac_assembly_keeps_executable_modes_and_uses_relative_runtime(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory).resolve()
            binary = directory / 'python'
            binary.write_bytes(b'fixture')
            binary.chmod(0o755)
            wheel = directory / 'product.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('sakuratts/__init__.py', '')
            plan = builder.Plan()
            plan.add(binary, 'runtime/main/bin/python3', 'cpython')
            plan.site_target = 'runtime/main/lib/python3.11/site-packages'
            plan.release = dict(target='macos-arm64', backend='mlx', backends=['mlx'], languages=['ja'],
                                services=['http'], preparation=False, minimum_macos='14.0',
                                python_executable='runtime/main/bin/python3')
            args = SimpleNamespace(root=root, output=directory / '樱花 空格', wheel=wheel, ffmpeg='ffmpeg')
            with patch.object(builder.subprocess, 'run', return_value=SimpleNamespace(stderr='', stdout='license')):
                builder.assemble(args, plan)
            self.assertTrue((args.output / plan.site_target / 'sakuratts/__init__.py').is_file())
            for name in ('sakuratts.command', 'start-server.command', 'check-runtime.command', 'runtime/main/bin/python3'):
                self.assertTrue((args.output / name).is_file())
                if os.name == 'posix':
                    self.assertTrue((args.output / name).stat().st_mode & 0o111)
            self.assertFalse(list(args.output.rglob('pyvenv.cfg')))
            self.assertFalse(list(args.output.rglob('*._pth')))
            config_path = args.output / 'configs/tts_infer.example.yaml'
            model, settings = read_inference_configuration(tts_config=config_path)
            self.assertIsNone(model)
            self.assertEqual(settings['backend'], 'mlx')
            self.assertEqual(settings['gpt_checkpoint'], 'models/your-gpt.ckpt')
            self.assertEqual(settings['sovits_checkpoint'], 'models/your-sovits.pth')
            self.assertNotIn('version:', config_path.read_text())

    def test_launcher_uses_unicode_bundle_and_clears_external_paths(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('mac_launcher', Path(__file__).resolve().parents[1] / 'scripts/portable/launcher.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / '樱花 空格'
            (root / 'runtime').mkdir(parents=True)
            (root / 'runtime/portable.json').write_text(json.dumps({'release': {'backend': 'mlx',
                'target': 'macos-arm64', 'minimum_macos': '14.0', 'python_executable': 'runtime/main/bin/python3'}}))
            with patch.object(launcher, 'ROOT', root), patch.object(sys, 'executable', str(root / 'runtime/main/bin/python3')), \
                    patch.object(launcher.platform, 'system', return_value='Darwin'), \
                    patch.object(launcher.platform, 'machine', return_value='arm64'), \
                    patch.object(launcher.platform, 'mac_ver', return_value=('14.0', (), '')), \
                    patch.object(launcher.os, 'chdir'), patch.dict(os.environ, {'PYTHONPATH': '/bad', 'DYLD_LIBRARY_PATH': '/bad'}, clear=True):
                launcher.configure()
                self.assertNotIn('PYTHONPATH', os.environ)
                self.assertNotIn('DYLD_LIBRARY_PATH', os.environ)
                self.assertEqual(os.environ['TMPDIR'], str(root / 'cache/tmp'))
                self.assertEqual(os.environ['PATH'], os.pathsep.join(map(str, [root / 'runtime/bin', Path('/usr/bin'), Path('/bin')])))
