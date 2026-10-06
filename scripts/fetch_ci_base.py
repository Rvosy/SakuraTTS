"""下载固定版本的运行环境输入，不依赖开发机路径。"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, choices=['windows-x64', 'macos-arm64'])
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    names = (['windows-base.tar.gz.part1', 'windows-base.tar.gz.part2'] if args.target == 'windows-x64'
             else ['SakuraTTS-macOS-plugin-preview.tar.gz'])
    for name in names:
        subprocess.run(['gh', 'release', 'download', 'runtime-base-20261004', '--repo', 'Rvosy/SakuraTTS',
                        '--pattern', name, '--dir', str(args.directory)], check=True)
    archive = args.directory / names[0]
    if len(names) > 1:
        archive = args.directory / 'base.tar.gz'
        with archive.open('xb') as output:
            for name in names:
                part = args.directory / name
                with part.open('rb') as source:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                part.unlink()
    extracted = args.directory / 'extracted'
    with tarfile.open(archive) as source:
        source.extractall(extracted, filter='data')
    archive.unlink()
    candidates = list(extracted.glob('*/bundle-manifest.json'))
    if len(candidates) != 1:
        raise ValueError('基包缺少唯一的发行清单')
    candidates[0].parent.rename(args.directory / 'bundle')
    extracted.rmdir()
    name = 'english-frontend-20261006.tar.gz'
    subprocess.run(['gh', 'release', 'download', 'runtime-base-20261004', '--repo', 'Rvosy/SakuraTTS',
                    '--pattern', name, '--dir', str(args.directory)], check=True)
    archive = args.directory / name
    if hashlib.sha256(archive.read_bytes()).hexdigest() != 'ca7f81e9fb25ca63a7b138a5e7355cbaecc463dd8610a0f0327ce4cb6833a169':
        raise ValueError('English frontend archive SHA256 mismatch')
    with tarfile.open(archive) as source:
        source.extractall(args.directory, filter='data')
    archive.unlink()


if __name__ == '__main__':
    main()
