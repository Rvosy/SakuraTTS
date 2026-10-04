# 2026-10-04 整合包 CI 接入记录

运行入口与分发结构见 [CI 发布指南](../ci-release.md)。

首次运行 [37190830404](https://github.com/Rvosy/SakuraTTS/actions/runs/37190830404) 使用提交 `88f3f145f345c2e43336b44cdc749312216a16fc`。Windows 2022 与 macOS 14 均在打包契约检查失败：`test_cpu_amd_assembly_generates_relocatable_backend_templates` 和 `test_mac_assembly_keeps_executable_modes_and_uses_relative_runtime` 缺少 `yaml`；`test_portable_runtime` 导入链缺少 `numpy`。

原因是工作流只安装构建工具与上传 SDK，没有安装测试实际使用的产品依赖。本地开发环境已有这些库，不能作为空 runner 的依赖证据。工作流改为同时安装项目的 `.[server]`，版本继续由 `pyproject.toml` 定义；保持原测试与两平台矩阵；第二次运行已通过这两项原失败阶段。

本地 FFmpeg 预检还发现两项构建配置问题：PCM muxer 的 configure 名称应为 `pcm_s16le` / `pcm_f32le`；Vorbis 的旧 Darwin 默认标志 `-force_cpusubtype_ALL` 不受当前链接器支持。修正格式选择、明确使用 `-O2 -fPIC` 构建依赖后，实际 OGG 参考解码与 Ogg、AAC 编解码通过；该结果不代替 Windows runner 验收。

第二次运行 [37190940678](https://github.com/Rvosy/SakuraTTS/actions/runs/37190940678) 使用提交 `f9cb8b72a17a5e7a58dc2bec54cef52146968365`，已通过原测试阶段，但两平台下载固定输入时返回 `release not found`。资产已存在且上传完整；[GitHub Release API 文档](https://docs.github.com/en/rest/releases/releases#list-releases) 说明草稿只对有推送权限的调用者可见。构建任务改用 `contents: write`，仅用于读取私有草稿；最终魔搭入口任务仍使用只读 GitHub 权限。第三次运行的两平台均成功下载固定输入，验证了该权限修复。

第三次运行 [37191031853](https://github.com/Rvosy/SakuraTTS/actions/runs/37191031853) 使用提交 `44aa5be2db3b90c1565891846a22c5d38fc442e8`，两平台构建与最终发布任务全部成功。Windows 打包契约 45 项通过；macOS 44 项通过，1 项 Windows 专用检查跳过。原先缺失测试依赖和无法读取草稿的两个失败阶段均在对应 runner 通过。

## 已发布产物

公开入口：[SuzushimaArisu/SakuraTTS](https://modelscope.cn/models/SuzushimaArisu/SakuraTTS)。版本为 `preview-37191031853-1`。

| 平台 | 压缩包字节数 | 解压后字节数 | CI 验收 |
|---|---:|---:|---|
| Windows x64 | 2,867,271,455 | 5,399,625,387 | CPU 实际合成、空闲卸载与推理进程回收 |
| macOS arm64 | 957,352,381 | 2,376,480,367 | 包内导入、MLX CPU 运算、managed HTTP 启动 |

两平台均通过 Ogg、AAC 编解码与归档内容检查。Windows 生成 44,160 帧、32 kHz 的 WAV；空闲后 `model_loaded=false`、`worker_pid=null`，没有残留推理进程。测试使用公共模型和合成参考信号，不作为音质验收。

发布后匿名访问两个压缩包均返回 HTTP 200，Content-Length 与发布记录一致。公开 `latest-preview.json` 的版本、源码提交、两平台条目与 Actions 发布结果逐项一致，README 下载链接也指向这一批文件。原始验证报告保存在该运行的 `verification-windows-x64`、`verification-macos-arm64` artifact；上传结果保存在对应的 `published-*` artifact。

CUDA、DirectML 真机合成、当前 CI 产物的 Metal 合成和人工听音尚未验收。桌宠插件仍通过离线导入安装或更新，在线下载与更新尚未接入。工作流位于 `codex/portable-modelscope` 分支，尚未合并默认分支。
