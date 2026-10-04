# 2026-10-04 整合包 CI 接入记录

运行入口与分发结构见 [CI 发布指南](../ci-release.md)。

首次运行 [37190830404](https://github.com/Rvosy/SakuraTTS/actions/runs/37190830404) 使用提交 `88f3f145f345c2e43336b44cdc749312216a16fc`。Windows 2022 与 macOS 14 均在打包契约检查失败：`test_cpu_amd_assembly_generates_relocatable_backend_templates` 和 `test_mac_assembly_keeps_executable_modes_and_uses_relative_runtime` 缺少 `yaml`；`test_portable_runtime` 导入链缺少 `numpy`。

原因是工作流只安装构建工具与上传 SDK，没有安装测试实际使用的产品依赖。本地开发环境已有这些库，不能作为空 runner 的依赖证据。工作流改为同时安装项目的 `.[server]`，版本继续由 `pyproject.toml` 定义；保持原测试与两平台矩阵，下一次运行用于验证这项环境修复。

本地 FFmpeg 预检还发现两项构建配置问题：PCM muxer 的 configure 名称应为 `pcm_s16le` / `pcm_f32le`；Vorbis 的旧 Darwin 默认标志 `-force_cpusubtype_ALL` 不受当前链接器支持。修正格式选择、明确使用 `-O2 -fPIC` 构建依赖后，实际 OGG 参考解码与 Ogg、AAC 编解码通过；该结果不代替 Windows runner 验收。

第二次运行 [37190940678](https://github.com/Rvosy/SakuraTTS/actions/runs/37190940678) 使用提交 `f9cb8b72a17a5e7a58dc2bec54cef52146968365`，已通过原测试阶段，但两平台下载固定输入时返回 `release not found`。资产已存在且上传完整；[GitHub Release API 文档](https://docs.github.com/en/rest/releases/releases#list-releases) 说明草稿只对有推送权限的调用者可见。构建任务改用 `contents: write`，仅用于读取私有草稿；最终魔搭入口任务仍使用只读 GitHub 权限。下一次运行验证该权限修复。
