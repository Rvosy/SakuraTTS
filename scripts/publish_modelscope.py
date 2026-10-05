"""发布平台产物；全部平台到齐后再更新预览入口。"""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

REPO = 'SuzushimaArisu/SakuraTTS'
TARGETS = {'windows-x64', 'macos-arm64'}


def download_url(path):
    return f'https://modelscope.cn/models/{REPO}/resolve/master/' + quote(path, safe='/')


def verify_files(api, records):
    files = {row.path: row.size for row in api.list_repo_files(REPO, 'model')}
    for record in records:
        if files.get(record['path']) != record['bytes']:
            raise ValueError('魔搭文件缺失或尺寸不匹配：' + record['path'])


def publish_platform(api, folder, target, release_id, commit, output):
    archive, = folder.glob('*.tar.gz')
    report = json.loads((folder / 'compression-report.json').read_text())
    path = f'previews/{release_id}/{archive.name}'
    existing = {row.path for row in api.list_repo_files(REPO, 'model')}
    if path in existing:
        raise ValueError('发行文件已存在，请使用新的发布编号：' + path)
    api.upload_file(REPO, 'model', archive, path, commit_message=f'上传 {release_id} {target}', disable_tqdm=True)
    api.upload_file(REPO, 'model', archive.with_name(archive.name + '.sha256'), path + '.sha256',
                    commit_message=f'上传 {release_id} {target} 校验文件', disable_tqdm=True)
    record = {'platform': target, 'path': path, 'url': download_url(path), 'bytes': archive.stat().st_size,
              'unpackedBytes': report['unpacked_bytes'], 'sourceCommit': commit, 'releaseId': release_id,
              'validation': 'cpu-synthesis-and-idle' if target == 'windows-x64' else 'imports-and-managed-http'}
    verify_files(api, [record])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def publish_index(api, directory, release_id, commit):
    records = [json.loads(path.read_text()) for path in directory.rglob('*.json')]
    if len(records) != 2 or {r['platform'] for r in records} != TARGETS:
        raise ValueError('Windows 与 macOS 产物必须全部上传成功')
    if any(r['releaseId'] != release_id or r['sourceCommit'] != commit for r in records):
        raise ValueError('两个平台的发布编号或源码提交不同')
    verify_files(api, records)
    index = {'channel': 'preview', 'releaseId': release_id, 'sourceCommit': commit, 'packages': records}
    api.upload_file(REPO, 'model', json.dumps(index, ensure_ascii=False, indent=2).encode(),
                    'latest-preview.json', commit_message=f'更新预览版 {release_id}')
    rows = '\n'.join(f"| {r['platform']} | [下载整合包]({r['url']}) | {r['bytes'] / 1e9:.2f} GB |" for r in sorted(records, key=lambda r:r['platform']))
    card = f'''---
license: other
---
# SakuraTTS 整合包

用于 Sakura 桌宠的本地语音引擎。空闲时释放推理进程，对话开始时可提前唤醒。

当前预览版：`{release_id}`。源码提交：[Rvosy/SakuraTTS@{commit[:12]}](https://github.com/Rvosy/SakuraTTS/tree/{commit})。

| 平台 | 文件 | 下载大小 |
|---|---|---|
{rows}

Windows x64 同包包含 CPU、NVIDIA CUDA 和 DirectML；macOS 包适用于 Apple silicon。
整合包包含 Python、推理运行库和模型准备组件，不包含角色 GPT / SoVITS 权重及个人参考音频。
在 SakuraTTS 插件设置中导入对应平台的压缩包；再次导入可更新。

Windows CPU 在 CI 验证实际合成和空闲释放；macOS CI 验证导入与 managed HTTP 启动，MLX 硬件合成另在本机验收。
CUDA、DirectML 真机合成和人工听音不属于本次 CI 验收。

项目代码采用 MIT；所携 Python、推理库、辅助模型与音频工具适用各自许可，见包内 `licenses/`。
FFmpeg 对应源码与构建脚本位于 `licenses/ffmpeg/`。
旧版文件保留在 `previews/`，机器可读入口为 `latest-preview.json`。
'''
    api.upload_file(REPO, 'model', card.encode(), 'README.md', commit_message=f'更新 {release_id} 下载说明')
    print(json.dumps(index, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['platform', 'index'])
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--release-id', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--target', choices=sorted(TARGETS))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from modelscope_hub import HubApi
    api = HubApi(token=os.environ['MODELSCOPE_TOKEN'])
    if args.mode == 'platform':
        if not args.target or not args.output:
            parser.error('platform requires --target and --output')
        publish_platform(api, args.directory, args.target, args.release_id, args.commit, args.output)
    else:
        publish_index(api, args.directory, args.release_id, args.commit)


if __name__ == '__main__':
    main()
