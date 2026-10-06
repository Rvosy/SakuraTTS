"""从固定运行环境基包和当前源码 wheel 装配预览版。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tomllib
import zipfile

import build_portable as builder


def install_english_dependencies(bundle, site, source, project, recipe, release, manifest):
    installed = builder.distributions(site)
    available = builder.distributions(source)
    selected = builder.dependency_names(dict(available, **installed),
        builder.main_requirements(project, recipe), release['python'], release['target'])
    plan = builder.Plan()
    for name in selected:
        if name in installed:
            continue
        directory, metadata = available[name]
        if release['target'] == 'macos-arm64':
            builder.macos.wheel_compatible(directory, recipe['minimum_macos'], release['python'])
        else:
            builder.windows_wheel_compatible(directory, release['python'])
        plan.package(source, directory, metadata, site.relative_to(bundle).as_posix())
    for name, row in plan.files.items():
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row['source'], target)
        record(bundle, target, manifest['files'], row['component'])
    manifest['components'].update(plan.components)


def install_english_resources(bundle, source, inventory, components):
    prep = bundle / 'runtime/preparation'
    manifest_path = prep / 'preparation-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    component = 'english-frontend-resources'
    provenance = json.loads((source / 'source.json').read_text(encoding='utf-8'))
    manifest['components'][component] = provenance
    components['preparation:' + component] = provenance
    for path in builder.tree(source):
        target = prep / 'official/english' / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        record(prep, target, manifest['files'], component)
        record(bundle, target, inventory, 'preparation:' + component)
    licenses_path = prep / 'licenses.json'
    licenses = json.loads(licenses_path.read_text(encoding='utf-8'))
    licenses['components'][component] = {'notices': ['official/english/' + name for name in
        ('english.md', 'CMUdict-README.txt', 'GPT-SoVITS-LICENSE.txt', 'Apache-2.0.txt')]}
    licenses_path.write_text(json.dumps(licenses, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    record(prep, licenses_path, manifest['files'], manifest['files']['licenses.json']['component'])
    manifest['bytes'] = sum(row['bytes'] for row in manifest['files'].values())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for path in (licenses_path, manifest_path):
        record(bundle, path, inventory, 'preparation-inventory')


def upgrade_preparation_threadpoolctl(bundle, wheel, inventory, components):
    """Replace the base bundle's pure Python controller, including its inventories."""
    if builder.digest(wheel) != '43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb':
        raise ValueError('Expected the pinned threadpoolctl 3.6.0 wheel (SHA256 mismatch)')
    prep = bundle / 'runtime/preparation'
    manifest_path = prep / 'preparation-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    license_path = prep / 'licenses.json'
    licenses = json.loads(license_path.read_text(encoding='utf-8'))
    with zipfile.ZipFile(wheel) as archive:
        payload = {item.filename: archive.read(item) for item in archive.infolist() if not item.is_dir()}
    old = next(name for name, row in manifest['components'].items() if row.get('name') == 'threadpoolctl')
    site = old.rsplit(':', 1)[0]
    # The old RECORD owns both the module and its versioned metadata.
    for name in list(manifest['files']):
        if manifest['files'][name]['component'] == old:
            target = (prep / name).resolve()
            if prep.resolve() not in target.parents:
                raise ValueError('Preparation inventory input escaped its root: ' + name)
            target.unlink()
            del manifest['files'][name]
            inventory.pop('runtime/preparation/' + name, None)
    component = {'name': 'threadpoolctl', 'version': '3.6.0', 'source': 'wheel',
                 'wheel_sha256': builder.digest(wheel)}
    manifest['components'][old] = component
    components['preparation:' + old] = component
    for name, data in payload.items():
        target = prep / site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        record(prep, target, manifest['files'], old)
        record(bundle, target, inventory, 'preparation:' + old)
    licenses['components'][old] = {**component, 'notices': [site + '/threadpoolctl-3.6.0.dist-info/' + name
                                                          for name in ('LICENSE', 'METADATA')]}
    license_path.write_text(json.dumps(licenses, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    record(prep, license_path, manifest['files'], manifest['files']['licenses.json']['component'])
    manifest['bytes'] = sum(row['bytes'] for row in manifest['files'].values())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for target in (license_path, manifest_path):
        record(bundle, target, inventory, 'preparation-inventory')


def install_product(bundle, site, wheel, inventory):
    # 旧 wheel 的模块必须整体替换，避免已删除源码残留在下一版。
    for directory in [site / 'sakuratts', *site.glob('sakuratts-*.dist-info')]:
        if directory.exists():
            prefix = directory.relative_to(bundle).as_posix() + '/'
            for name in list(inventory):
                if name.startswith(prefix):
                    del inventory[name]
            shutil.rmtree(directory)
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            path = Path(item.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in item.filename or ':' in item.filename or path.suffix == '.pth':
                raise ValueError('Unsafe product wheel entry: ' + item.filename)
            target = site / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))
            record(bundle, target, inventory, 'product-or-generated')


def record(bundle, path, inventory, component='product-or-generated'):
    inventory[path.relative_to(bundle).as_posix()] = {
        'bytes': path.stat().st_size, 'sha256': builder.digest(path), 'component': component}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'wheel', 'threadpoolctl-wheel', 'english-dependencies', 'english-resources', 'ffmpeg', 'ffmpeg-source', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / 'bundle-manifest.json').read_text())
    release = manifest['release']
    apple = release['target'] == 'macos-arm64'
    recipe = builder.read_recipe(root / 'packaging/recipes' / ('macos-mlx-ja.toml' if apple else 'windows-x64.toml'))
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    if release['backends'] != recipe['backends'] or not release['preparation']:
        raise ValueError('运行环境基包与当前发行组合不匹配')
    site = bundle / ('runtime/main/lib/python3.11/site-packages' if apple else 'runtime/main/Lib/site-packages')
    install_english_dependencies(bundle, site, args.english_dependencies, project, recipe, release, manifest)
    builder.dependency_names(builder.distributions(site), builder.main_requirements(project, recipe),
                             release['python'], release['target'])
    inventory = manifest['files']
    install_english_resources(bundle, args.english_resources, inventory, manifest['components'])
    upgrade_preparation_threadpoolctl(bundle, args.threadpoolctl_wheel, inventory, manifest['components'])
    install_product(bundle, site, args.wheel, inventory)
    for name in builder.launch_files(recipe):
        target = bundle / ('README.md' if name == 'README-macos.md' else name)
        shutil.copy2(root / 'scripts/portable' / name, target)
        if target.suffix == '.command':
            target.chmod(0o755)
        record(bundle, target, inventory)
    ffmpeg = bundle / ('runtime/bin/ffmpeg' if apple else 'runtime/bin/ffmpeg.exe')
    shutil.copy2(args.ffmpeg, ffmpeg)
    ffmpeg.chmod(0o755)
    record(bundle, ffmpeg, inventory, 'ffmpeg-source-build')
    license_root = bundle / 'licenses/ffmpeg'
    license_root.mkdir(exist_ok=True)
    for source, name in ((args.ffmpeg_source, 'ffmpeg-7.1.2.tar.xz'),
                         (root / 'scripts/build_ffmpeg.sh', 'build_ffmpeg.sh'),
                         (args.ffmpeg_source.parent / 'libogg-1.3.5.tar.xz', 'libogg-1.3.5.tar.xz'),
                         (args.ffmpeg_source.parent / 'libvorbis-1.3.7.tar.xz', 'libvorbis-1.3.7.tar.xz')):
        target = license_root / name
        shutil.copy2(source, target)
        record(bundle, target, inventory, 'ffmpeg-corresponding-source')
    result = subprocess.run([str(ffmpeg), '-L'], capture_output=True, text=True, check=True)
    notice = bundle / 'licenses/FFmpeg-build-and-license.txt'
    notice.write_text(result.stdout + result.stderr, encoding='utf-8')
    record(bundle, notice, inventory, 'ffmpeg-license')
    # 辅助权重保持原字节，补入对应上游模型卡和分发说明。
    for source in (root / 'docs/third-party').glob('*'):
        if source.is_file():
            target = bundle / 'licenses/sakuratts' / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            record(bundle, target, inventory, 'third-party-notices')
    prep = bundle / 'runtime/preparation'
    license_file = prep / 'licenses.json'
    licenses = json.loads(license_file.read_text())
    analysis = licenses['components']['public-analysis-resource']
    analysis['license_status'] = 'GPT-SoVITS distribution declaration and original model cards are included; see licenses/sakuratts/portable-resources.md at the bundle root.'
    license_file.write_text(json.dumps(licenses, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    prep_manifest_file = prep / 'preparation-manifest.json'
    prep_manifest = json.loads(prep_manifest_file.read_text())
    auxiliary = prep / 'auxiliary-model-sources.json'
    shutil.copy2(root / 'packaging/auxiliary-model-sources.json', auxiliary)
    record(bundle, auxiliary, inventory, 'preparation-inventory')
    for path in (license_file, auxiliary):
        prep_manifest['files'][path.name].update(bytes=path.stat().st_size, sha256=builder.digest(path))
    prep_manifest['bytes'] = sum(row['bytes'] for row in prep_manifest['files'].values())
    prep_manifest_file.write_text(json.dumps(prep_manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for path in (license_file, prep_manifest_file):
        record(bundle, path, inventory, 'preparation-inventory')
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise ValueError('CI 发行必须使用干净的源码提交')
    release.update(version=project['version'], source_commit=commit, source_dirty=False, languages=recipe['languages'])
    marker = bundle / 'runtime/portable.json'
    value = json.loads(marker.read_text())
    value['release'] = release
    marker.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    record(bundle, marker, inventory)
    manifest.update(wheel_sha256=builder.digest(args.wheel), bytes=sum(row['bytes'] for row in inventory.values()))
    (bundle / 'bundle-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    bundle.rename(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
