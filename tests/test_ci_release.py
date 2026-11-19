"""发行包替换与两平台发布入口的行为检查。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from datetime import datetime
import zipfile

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import rebuild_portable
import publish_modelscope


class Api:
    def __init__(self, files=()):
        self.files = dict(files)
        self.uploads = []
    def list_repo_files(self, *_):
        return [SimpleNamespace(path=path, size=size) for path, size in self.files.items()]
    def upload_file(self, repo, kind, data, path, **_):
        self.uploads.append(path)
        self.files[path] = len(data) if isinstance(data, bytes) else Path(data).stat().st_size


class ReleaseTests(unittest.TestCase):
    def test_rebuild_replaces_preparation_resources_and_syncs_inventories(self):
        from test_preparation_builder import I18N_SOURCE, put
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            prep = root / 'runtime/preparation'
            controller = 'Lib/site-packages:threadpoolctl'
            licenses = {'components': {controller: {}, 'public-analysis-resource': {}}}
            base = {'licenses.json': (json.dumps(licenses), 'generated'),
                    'official/tools/i18n/i18n.py': (I18N_SOURCE, 'official'),
                    'official/tools/i18n/locale/en_US.json': ('{"hello":"Hello"}', 'official'),
                    'official/chinese/bert.bin': ('obsolete weights', 'chinese-frontend-resources'),
                    'Lib/site-packages/threadpoolctl.py': ('old', controller),
                    'Lib/site-packages/threadpoolctl-3.1.0.dist-info/METADATA': ('old', controller),
                    'Lib/site-packages/numpy.py': ('preserved', 'numpy')}
            files = {}
            bundle = {'files': {}, 'components': {}}
            for name, (data, owner) in base.items():
                path = put(prep, name, data)
                rebuild_portable.record(prep, path, files, owner)
                rebuild_portable.record(root, path, bundle['files'], 'preparation:' + owner)
            # A previous attempt already removed the old wheel metadata.
            (prep / 'Lib/site-packages/threadpoolctl-3.1.0.dist-info/METADATA').unlink()
            manifest = put(prep, 'preparation-manifest.json', json.dumps({'files': files,
                           'components': {controller: {'name': 'threadpoolctl'}}}))
            english, chinese = root / 'english', root / 'chinese'
            for name in ('checkpoint.npz', 'english.md', 'CMUdict-README.txt', 'GPT-SoVITS-LICENSE.txt', 'Apache-2.0.txt'):
                put(english, name)
            put(english, 'source.json', '{"sources": {"upstream": "sha256"}}')
            names = ['g2pw.onnx', 'bert.onnx', 'tokenizer.json', 'MONOPHONIC_CHARS.txt', 'POLYPHONIC_CHARS.txt',
                     'bopomofo_to_pinyin_wo_tune_dict.json', 'traditional_to_simplified.json', 'corrections.json',
                     'pinyin-symbols.json', 'symbols-v2.json', 'notices/model.txt']
            for name in names:
                put(chinese, name)
            put(chinese, 'manifest.json', json.dumps({'format': 'sakuratts-chinese-v2-resources-v1',
                                                       'files': dict.fromkeys(names, {})}))
            wheel = root / 'threadpoolctl.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('threadpoolctl.py', 'new')
                archive.writestr('threadpoolctl-3.6.0.dist-info/METADATA', 'Name: threadpoolctl\nVersion: 3.6.0\n')
                archive.writestr('threadpoolctl-3.6.0.dist-info/LICENSE', 'license')
            args = SimpleNamespace(threadpoolctl_wheel=wheel, english_resources=english, chinese_resources=chinese)
            auxiliary = put(root, 'auxiliary-model-sources.json', '{}')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                rebuild_portable.update_preparation(root, bundle, args, auxiliary)
            digest = rebuild_portable.builder.digest
            def fixture_digest(path):
                return ('43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb'
                        if path == wheel else digest(path))
            with patch.object(rebuild_portable.builder, 'digest', side_effect=fixture_digest):
                rebuild_portable.update_preparation(root, bundle, args, auxiliary)

            updated = json.loads(manifest.read_text())
            self.assertEqual(updated['bytes'], sum(row['bytes'] for row in updated['files'].values()))
            self.assertEqual(set(bundle['files']), {'runtime/preparation/' + name for name in
                             [*updated['files'], 'preparation-manifest.json']})
            for directory, rows in ((prep, updated['files']), (root, bundle['files'])):
                for name, row in rows.items():
                    self.assertEqual(row['sha256'], digest(directory / name))
                    self.assertEqual(row['bytes'], (directory / name).stat().st_size)
            self.assertFalse((prep / 'official/chinese/bert.bin').exists())
            self.assertEqual((prep / 'official/chinese/bert.onnx').read_text(), 'fixture')
            self.assertEqual((prep / 'official/english/checkpoint.npz').read_text(), 'fixture')
            self.assertEqual(bundle['components']['preparation:english-frontend-resources']['sources'],
                             {'upstream': 'sha256'})
            self.assertEqual((prep / 'Lib/site-packages/numpy.py').read_text(), 'preserved')
            self.assertEqual((prep / 'Lib/site-packages/threadpoolctl.py').read_text(), 'new')
            self.assertEqual(bundle['components']['preparation:' + controller]['version'], '3.6.0')
            licenses = json.loads((prep / 'licenses.json').read_text())['components']
            for owner in ('english-frontend-resources', 'chinese-frontend-resources', controller):
                for name in licenses[owner]['notices']:
                    self.assertTrue((prep / name).is_file())
            spec = importlib.util.spec_from_file_location('rebuilt_i18n', prep / 'official/tools/i18n/i18n.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.load_language_list('en_US'), {'hello': 'Hello'})

    def test_missing_dependencies_are_added_without_overwriting_base_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site, source = root / 'runtime/main/site', root / 'dependencies'
            for directory, version in ((site, '1'), (source, '2')):
                metadata = directory / ('existing-' + version + '.dist-info')
                metadata.mkdir(parents=True)
                (metadata / 'METADATA').write_text('Name: existing\nVersion: ' + version + '\n')
            metadata = source / 'added-1.dist-info'
            metadata.mkdir()
            (metadata / 'METADATA').write_text('Name: added\nVersion: 1\nRequires-Dist: existing>=1\n')
            (metadata / 'WHEEL').write_text('Tag: py3-none-any\n')
            (metadata / 'LICENSE').write_text('license')
            (source / 'added.py').write_text('added')
            names = ['added.py', *['added-1.dist-info/' + name for name in ('METADATA', 'WHEEL', 'LICENSE', 'RECORD')]]
            (metadata / 'RECORD').write_text(''.join(name + ',,\n' for name in names))
            manifest = {'files': {}, 'components': {}}
            rebuild_portable.record(root, site / 'existing-1.dist-info/METADATA', manifest['files'],
                                    'runtime/main/site:existing')
            with patch.object(rebuild_portable.builder, 'main_requirements', return_value=['added==1']):
                rebuild_portable.install_language_dependencies(root, site, source, {}, {},
                    {'python': '3.12', 'target': 'windows-x64'}, manifest)
            self.assertEqual((site / 'added.py').read_text(), 'added')
            self.assertTrue((site / 'existing-1.dist-info').exists())
            self.assertFalse((site / 'existing-2.dist-info').exists())
            self.assertIn('runtime/main/site/added-1.dist-info/LICENSE', manifest['files'])
            for name, row in manifest['files'].items():
                self.assertEqual(row['sha256'], rebuild_portable.builder.digest(root / name))

    def test_rebuild_recovers_language_dependency_copied_before_manifest_was_saved(self):
        from test_preparation_builder import put
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site, source = root / 'runtime/main/site', root / 'dependencies'
            metadata = 'reader-1.dist-info/METADATA'
            files = {metadata: 'Name: reader\nVersion: 1\n', 'reader.py': 'installed',
                     'reader-1.dist-info/WHEEL': 'Tag: py3-none-any\n'}
            record = 'reader-1.dist-info/RECORD'
            files[record] = ''.join(name + ',,\n' for name in [*files, record])
            for name, content in files.items():
                put(source, name, content)
            put(site, metadata, files[metadata])
            manifest = {'files': {}, 'components': {}}
            with patch.object(rebuild_portable.builder, 'main_requirements', return_value=['reader==1']):
                rebuild_portable.install_language_dependencies(root, site, source, {}, {},
                    {'python': '3.12', 'target': 'windows-x64'}, manifest)
            self.assertEqual((site / 'reader.py').read_text(), 'installed')
            self.assertEqual(set(manifest['files']), {'runtime/main/site/' + name for name in files})
            self.assertEqual(manifest['components']['runtime/main/site:reader']['version'], '1')

    def test_published_names_include_date_and_daily_sequence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'SakuraTTS-windows-x64.7z'
            archive.write_bytes(b'archive')
            archive.with_name(archive.name + '.sha256').write_text('abc  ' + archive.name)
            (root / 'compression-report.json').write_text('{"unpacked_bytes":20}')
            api = Api()
            with patch.object(publish_modelscope, 'datetime') as clock:
                clock.now.return_value = datetime(2026, 10, 5)
                for index, suffix in enumerate(('', '-1', '-2')):
                    output = root / 'result.json'
                    publish_modelscope.publish_platform(api, root, 'windows-x64', str(index), 'commit', output)
                    name = f'SakuraTTS-windows-x64-20261005{suffix}.7z'
                    record = json.loads(output.read_text())
                    self.assertEqual(record['path'], f'previews/{index}/{name}')
                    self.assertEqual(api.uploads[-1], record['path'] + '.sha256')
                    self.assertEqual(api.files[api.uploads[-1]], len(f'abc  {name}\n'.encode()))

    def test_new_wheel_removes_deleted_product_files_and_preserves_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site = root / 'site'; old = site / 'sakuratts/removed.py'
            old.parent.mkdir(parents=True);old.write_text('old')
            dependency = site / 'runtime.dll';dependency.write_bytes(b'runtime')
            wheel = root / 'product.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('sakuratts/new.py', 'new')
            inventory = {'site/sakuratts/removed.py': {}, 'site/runtime.dll': {}}
            rebuild_portable.install_product(root, site, wheel, inventory)
            self.assertFalse(old.exists())
            self.assertNotIn('site/sakuratts/removed.py', inventory)
            self.assertEqual((site/'sakuratts/new.py').read_text(), 'new')
            self.assertEqual(dependency.read_bytes(), b'runtime')

    def test_index_waits_for_both_verified_platforms(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);api=Api()
            for target in ('windows-x64','macos-arm64'):
                extension = '7z' if target == 'windows-x64' else 'tar.gz'
                row={'platform':target,'path':f'previews/run/{target}.{extension}','bytes':12,
                     'url':'https://example.invalid/'+target,'releaseId':'run','sourceCommit':'commit'}
                (root/(target+'.json')).write_text(json.dumps(row))
                if target == 'windows-x64':api.files[row['path']]=12
            with self.assertRaisesRegex(ValueError,'缺失'):
                publish_modelscope.publish_index(api,root,'run','commit')
            self.assertEqual(api.uploads,[])
            api.files['previews/run/macos-arm64.tar.gz']=12
            publish_modelscope.publish_index(api,root,'run','commit')
            self.assertEqual(api.uploads,['latest-preview.json','README.md'])

    def test_index_rejects_mixed_source_commits(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);api=Api()
            for target in ('windows-x64','macos-arm64'):
                (root/(target+'.json')).write_text(json.dumps({'platform':target,
                     'releaseId':'run','sourceCommit':target}))
            with self.assertRaisesRegex(ValueError,'不同'):
                publish_modelscope.publish_index(api,root,'run','commit')
            self.assertEqual(api.uploads,[])

    def test_chinese_download_preserves_default_desktop_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            api = Api()
            published = {}
            def upload(repo, kind, data, path, **_):
                published[path] = data
            api.upload_file = upload
            for platform, languages in [('windows-x64', ['ja', 'en']), ('macos-arm64', ['ja', 'en']),
                                        ('windows-x64', ['zh', 'ja', 'en'])]:
                name = platform + '-' + '-'.join(languages)
                record = {'platform': platform, 'languages': languages, 'path': name, 'url': name,
                          'bytes': 1, 'sourceCommit': 'commit', 'releaseId': 'run'}
                (root / (name + '.json')).write_text(json.dumps(record))
                api.files[name] = 1
            publish_modelscope.publish_index(api, root, 'run', 'commit', with_chinese=True)
            index = json.loads(published['latest-preview.json'])
            windows, = [p for p in index['packages'] if p['platform'] == 'windows-x64']
            self.assertEqual(windows['languages'], ['ja', 'en'])
            self.assertEqual(index['variants'][0]['languages'], ['zh', 'ja', 'en'])

    def test_existing_release_archive_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'SakuraTTS.7z').write_bytes(b'archive')
            (root/'compression-report.json').write_text('{"unpacked_bytes":20}')
            api=Api([('previews/run/SakuraTTS.7z',7)])
            with self.assertRaisesRegex(ValueError,'已存在'):
                publish_modelscope.publish_platform(api,root,'windows-x64','run','commit',root/'result.json')
            self.assertEqual(api.uploads,[])

    def test_platform_uploads_archive_and_checksum_for_its_format(self):
        for target, extension in (('windows-x64', '7z'), ('macos-arm64', 'tar.gz')):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                name = f'SakuraTTS-{target}.{extension}'
                (root/name).write_bytes(b'archive')
                (root/(name+'.sha256')).write_text('checksum')
                other = 'tar.gz' if extension == '7z' else '7z'
                (root/f'unrelated.{other}').write_bytes(b'not for this platform')
                (root/'compression-report.json').write_text('{"unpacked_bytes":20}')
                api = Api()
                output = root/'published'/f'{target}.json'
                with patch.object(publish_modelscope, 'datetime') as clock:
                    clock.now.return_value = datetime(2026, 10, 5)
                    publish_modelscope.publish_platform(api,root,target,'run','commit',output)
                path = f'previews/run/SakuraTTS-{target}-20261005.{extension}'
                self.assertEqual(api.uploads, [path, path+'.sha256'])
                record = json.loads(output.read_text(encoding='utf-8'))
                self.assertEqual(record['path'], path)
                self.assertEqual(record['url'], publish_modelscope.download_url(path))
                self.assertEqual(record['bytes'], 7)


if __name__ == '__main__':
    unittest.main()
