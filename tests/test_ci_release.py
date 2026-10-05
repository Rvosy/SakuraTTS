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
