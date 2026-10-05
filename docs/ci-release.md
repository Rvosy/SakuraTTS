# GitHub Actions 整合包预览发布

工作流为 [portable-preview.yml](../.github/workflows/portable-preview.yml)，上传到 [SuzushimaArisu/SakuraTTS](https://modelscope.cn/models/SuzushimaArisu/SakuraTTS)。仓库 Secret `MODELSCOPE_TOKEN` 保存该魔搭仓库的写入令牌；令牌只交给上传步骤，不进入构建输入或产物。

当前支持 Windows x64 统一包和 macOS arm64 包。两平台分别执行打包测试、构建产品 wheel、装配、运行检查、归档与上传，全部成功后更新 `latest-preview.json` 和下载说明。失败保留已上传候选文件与 Actions 日志，不覆盖旧版入口、不自动重跑失败测试。

Windows 使用 Windows runner 自带的 7-Zip 和 [归档脚本](../scripts/archive_portable.py) 的 `maximum` 档位，压缩后运行 `7z t`。macOS 继续使用 tar.gz，保留可执行权限。上传文件名追加北京时间日期，例如 `SakuraTTS-windows-x64-20261005.7z` 和 `SakuraTTS-macos-arm64-20261005.tar.gz`；同日后续版本在日期后追加 `-1`、`-2`。两平台均上传对应的 `.sha256`，其中的文件名与下载文件一致。

## 运行环境输入

`runtime-base-20261004` GitHub Release 草稿保存固定运行环境输入，来源是已完成本机联调的完整包。它不是用户发行版；Windows 输入分成两段，构建时顺序拼接。下载脚本见 [fetch_ci_base.py](../scripts/fetch_ci_base.py)。GitHub 的 Release 草稿仅向具备推送权限的调用者可见，因此构建任务使用自身令牌的 `contents: write` 权限访问草稿；任务不写入 GitHub 仓库，不公开草稿下载地址。

每次发行从检出的源码重新构建 wheel，完整替换旧产品模块和启动器，重新生成版本与源码身份；准备环境和推理依赖来自固定基包。当前依赖声明与基包不匹配时构建失败，需要重新准备基包。这不是从空环境重新编译所有第三方依赖的工作流。

FFmpeg 单独从固定的 7.1.2 源码编译，保留音频解码、PCM、AAC 和 Ogg 输出。源码、构建脚本与实际许可一同进入整合包，详见[辅助资源声明](third-party/portable-resources.md)。原基包的音频工具会被替换。

更新基包时，在对应平台按[Windows 指南](portable-bundle.md)和[macOS 指南](portable-macos.md)重新准备并验收运行环境，保存为新版本输入，再修改下载脚本中的固定版本与文件名。保留原输入，避免历史构建因依赖漂移失效。

## 触发与验收

项目统一在 `main` 维护。提交、推送代码都不会构建或上传整合包；发布只由 Actions 的手动入口触发。

发布步骤：推送需要发布的代码到 `main`，打开仓库 **Actions → 构建并发布整合包预览 → Run workflow**，选择 `main` 并运行。该操作会构建 Windows 和 macOS 包、执行检查并上传魔搭；两平台均成功后更新下载入口。未点击运行时，魔搭保持上一版。

发布过程串行执行，平台构建并行执行。发布编号使用 GitHub run ID 与 attempt，重跑不会覆盖上一轮候选文件。

- Windows 在 CPU 上检查矩阵运算、原始模型转换、合成、空闲卸载和进程退出。公共测试权重与合成参考音频仅用于验收，不进入整合包。
- macOS 在托管 runner 检查包内导入、MLX CPU 运算及 managed HTTP 未加载模型启动。Metal 合成仍需具备相应硬件加速的 Mac 验收。
- CUDA、DirectML 合成、峰值显存和人工听音需要独立真机证据，不以构建成功代替。

运行报告通过 Actions artifact 保存；大型整合包直接上传魔搭。原始错误与首次失败原因必须调查清楚后修复，不设置自动重跑。

## 魔搭目录与更新入口

每次发布保存到 `previews/<发布编号>/`，包含两平台压缩包与归档校验文件。平台上传后核对远端文件尺寸，再保存平台发布结果。最后的发布任务要求两个结果来自同一源码提交和同一发布编号，并再次核对远端文件。

`latest-preview.json` 只在两个平台全部成功后更新，记录平台、文件位置、尺寸、源码提交与验证范围。`README.md` 提供对应下载链接；稳定版入口尚未建立。当前桌宠插件仍使用离线导入，尚未读取该清单自动下载或更新。
