"""从固定运行环境基包和当前源码 wheel 装配预览版。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tomllib
import zipfile

import build_portable as builder
from build_preparation import patch_i18n


def install_language_dependencies(bundle, site, source, project, recipe, release, manifest):
    # 中断后，磁盘上可能已经出现下一个 wheel 的 METADATA，基包清单仍是上次保存的版本。
    installed = {name: package for name, package in builder.distributions(site).items()
                 if (package[0] / 'METADATA').relative_to(bundle).as_posix() in manifest['files']}
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
        manifest['files'][name] = {key: row[key] for key in ('bytes', 'sha256', 'component')}
    manifest['components'].update(plan.components)


def remove_obsolete(prep, files, component, keep):
    for name in [name for name, row in files.items() if row['component'] == component and name not in keep]:
        target = (prep / name).resolve()
        if prep not in target.parents:
            raise ValueError('Preparation inventory input escaped its root: ' + name)
        target.unlink(missing_ok=True)
        del files[name]


def install_language_resources(prep, manifest, licenses, source, language):
    component = language + '-frontend-resources'
    if language == 'english':
        provenance = json.loads((source / 'source.json').read_text(encoding='utf-8'))
        resources = builder.tree(source)
        notices = ('english.md', 'CMUdict-README.txt', 'GPT-SoVITS-LICENSE.txt', 'Apache-2.0.txt')
    else:
        provenance, resources = builder.chinese_resources(source)
        notices = sorted(name for name in provenance['files'] if name.startswith('notices/'))
    manifest['components'][component] = provenance
    prefix = 'official/' + language + '/'
    sources = {prefix + path.relative_to(source).as_posix(): path for path in resources}
    remove_obsolete(prep, manifest['files'], component, sources)
    for name, path in sources.items():
        target = prep / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        record(prep, target, manifest['files'], component)
    licenses['components'][component] = {'notices': [prefix + name for name in notices]}


def upgrade_preparation_threadpoolctl(prep, manifest, licenses, wheel):
    checksum = builder.digest(wheel)
    if checksum != '43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb':
        raise ValueError('Expected the pinned threadpoolctl 3.6.0 wheel (SHA256 mismatch)')
    old = next(name for name, row in manifest['components'].items() if row.get('name') == 'threadpoolctl')
    site = old.rsplit(':', 1)[0]
    with zipfile.ZipFile(wheel) as archive:
        entries = {site + '/' + item.filename: item for item in archive.infolist() if not item.is_dir()}
        remove_obsolete(prep, manifest['files'], old, entries)
        for name, item in entries.items():
            target = prep / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))
            record(prep, target, manifest['files'], old)
    component = {'name': 'threadpoolctl', 'version': '3.6.0', 'source': 'wheel', 'wheel_sha256': checksum}
    manifest['components'][old] = component
    licenses['components'][old] = {**component, 'notices': [site + '/threadpoolctl-3.6.0.dist-info/' + name
                                                          for name in ('LICENSE', 'METADATA')]}


def update_preparation(bundle, bundle_manifest, args, auxiliary_source):
    prefix = 'runtime/preparation/'
    prep = bundle / prefix
    manifest_path = prep / 'preparation-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    license_path = prep / 'licenses.json'
    licenses = json.loads(license_path.read_text(encoding='utf-8'))
    upgrade_preparation_threadpoolctl(prep, manifest, licenses, args.threadpoolctl_wheel)
    install_language_resources(prep, manifest, licenses, args.english_resources, 'english')
    if args.chinese_resources:
        install_language_resources(prep, manifest, licenses, args.chinese_resources, 'chinese')

    files = manifest['files']
    path = patch_i18n(prep)
    record(prep, path, files, files[path.relative_to(prep).as_posix()]['component'])
    auxiliary = prep / 'auxiliary-model-sources.json'
    shutil.copy2(auxiliary_source, auxiliary)
    record(prep, auxiliary, files, 'generated')
    licenses['components']['public-analysis-resource']['license_status'] = 'GPT-SoVITS distribution declaration and original model cards are included; see licenses/sakuratts/portable-resources.md at the bundle root.'
    builder.write(license_path, json.dumps(licenses, ensure_ascii=False, indent=2) + '\n')
    record(prep, license_path, files, files['licenses.json']['component'])
    manifest['bytes'] = sum(row['bytes'] for row in files.values())
    builder.write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')

    inventory = bundle_manifest['files']
    for name in [name for name in inventory if name.startswith(prefix)]:
        del inventory[name]
    for name, row in files.items():
        inventory[prefix + name] = {**row, 'component': 'preparation:' + row['component']}
    record(bundle, manifest_path, inventory, 'preparation-inventory')
    for name, component in manifest['components'].items():
        bundle_manifest['components']['preparation:' + name] = component


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
    for name in ('bundle', 'wheel', 'threadpoolctl-wheel', 'english-resources', 'ffmpeg', 'ffmpeg-source', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--language-dependencies', '--english-dependencies', required=True, type=Path)
    parser.add_argument('--chinese-resources', type=Path)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--allow-dirty', action='store_true', help='Build a local candidate from uncommitted source')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / 'bundle-manifest.json').read_text(encoding='utf-8'))
    release = manifest['release']
    apple = release['target'] == 'macos-arm64'
    recipe = builder.read_recipe(args.recipe or root / 'packaging/recipes' / ('macos-mlx-ja.toml' if apple else 'windows-x64.toml'))
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    if release['target'] != recipe['target'] or release['backends'] != recipe['backends'] or not release['preparation']:
        raise ValueError('运行环境基包与当前发行组合不匹配')
    if ('zh' in recipe['languages']) != bool(args.chinese_resources):
        raise ValueError('中日英配方需要 --chinese-resources 指向完整的离线中文资源')
    if 'zh' in release['languages'] and 'zh' not in recipe['languages']:
        raise ValueError('请使用日英基包构建日英版，以免包含中文资源')
    if args.output.exists():
        raise FileExistsError(args.output)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip())
    if dirty and not args.allow_dirty:
        raise ValueError('CI 发行必须使用干净的源码提交')
    site = bundle / ('runtime/main/lib/python3.11/site-packages' if apple else 'runtime/main/Lib/site-packages')
    install_language_dependencies(bundle, site, args.language_dependencies, project, recipe, release, manifest)
    inventory = manifest['files']
    update_preparation(bundle, manifest, args, root / 'packaging/auxiliary-model-sources.json')
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
    release.update(version=project['version'], source_commit=commit, source_dirty=dirty, languages=recipe['languages'])
    marker = bundle / 'runtime/portable.json'
    value = json.loads(marker.read_text(encoding='utf-8'))
    value['release'] = release
    marker.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    record(bundle, marker, inventory)
    manifest.update(wheel_sha256=builder.digest(args.wheel), bytes=sum(row['bytes'] for row in inventory.values()))
    (bundle / 'bundle-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bundle.rename(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
