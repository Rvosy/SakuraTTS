# macOS 整合包修复与验收（2026-10-04）

本次验证覆盖 Apple silicon 的 V2Pro／FP32 日文流程：原始权重转换、新参考编码、HTTP、缓存、取消恢复和整合包。构建来自 `fix/macos-portable-delivery` 工作树，基准提交为 `30ab4d5deb8ac149f84b0559824c1f01f28f8345`，清单标记 `source_dirty: true`。这是本地候选包，尚未公开发布。

## 环境与输入

实机为 Apple M4、16 GiB 统一内存，macOS 26.5.2（25F84）。解释器为独立 CPython 3.11.15 arm64。主环境使用 MLX／mlx-metal 0.32.2、NumPy 2.4.6、ONNX Runtime 1.30.0、pyopenjtalk-plus 0.4.1.post9；准备组件使用 CPU PyTorch／TorchAudio 2.7.1。FFmpeg 来自 imageio-ffmpeg 0.6.0 的 arm64 独立二进制，实际构建信息随包保存。

发行目标为 macOS 14.0。MLX 和 mlx-metal 的 wheel 标签分别为 `cp311-cp311-macosx_14_0_arm64`、`py3-none-macosx_14_0_arm64`。构建检查通过：主环境 39 个、准备组件 309 个 Mach-O 文件，均包含 arm64，最低系统版本不高于目标；依赖解析仅使用包内文件和系统库。这些是静态条件，不代替 macOS 14 真机运行。

模型为朱雀院红叶 V2Pro，参考转写为 `じゃあ、私、もっと悪い子になっちゃおうな〜`。原始输入保持不变，也未进入整合包。

| 输入 | SHA256 |
| --- | --- |
| `朱雀院红叶-e15.ckpt` | `010197bfc30b04d991f2bf060f962549932a8278b98c137d92f980e9cca8c0e9` |
| `朱雀院红叶_e8_s38928.pth` | `f8bd92196175f435ae9df2bc01e87b0119a5d94c0af81ada4573a9720536ab38` |
| `VO02_0204.OGG` | `c265a87781115e67d787ed81a7d2a3755a0aff54dd15d6c304b5fe61b01b91b4` |

完整依赖及来源输入记录在本地 `outputs/macos-build/runtime-final-inputs.json`、`outputs/macos-build/preparation-final-inputs.json`；发布包内的清单不包含这些构建输入绝对路径。

## 修复范围

- 原生 MLX 声学转换器进入产品准备模块，公共转换和权重切换按后端选择实现。HTTP 复用现有请求、参考缓存、切换回滚和进程生命周期。
- 准备工具根据原始声学权重识别 V2Pro／V2ProPlus，参考清单记录实际家族。MLX 仍只接受非 LoRA V2Pro 和 FP32。
- Mac recipe 使用独立 CPython、平台匹配的 wheel、包内 FFmpeg 和 `.command` 启动器；主推理和 CPU 准备环境分开装配。构建拒绝过新系统 wheel、错误 Python ABI／架构和未打包的外部动态库。
- 旧日文导出器只复制所需运行源码，修复重复创建输出目录及 requirements 目录缺失。相关测试使用规范化临时路径，避免 `/var` 与 `/private/var` 的别名差异。
- 归档按发行清单选文件，核对输入与归档内内容，保留执行权限；中文归档名的校验和文件使用 UTF-8。

## 自动测试

`PATH="$PWD/outputs/macos-build/test-bin:$PATH" .venv/bin/python -m unittest discover -s tests`：389 项，384 项通过，5 项跳过。使用独立 FFmpeg，以覆盖实际音频编码行为。跳过项为两项 Windows Unicode 转换、一项 Windows Job Object、一项非 Mac 启动诊断，以及本机 Accelerate 不适用的 threadpoolctl BLAS 线程控制检查。

旧导出器回归测试另有 1 项通过。该测试使用临时文件和解释器探测替身，验证导出选择及目录布局，不作为真实推理证据。

原始日志保存在 `outputs/macos-build/logs/`。早期失败日志保留：临时路径断言、宿主 FFmpeg 缺少 Vorbis 编码器，以及中文校验和文件编码错误。最终通过结果不替换这些失败记录。

## 真机首次使用与生命周期

`outputs/macos-first-use/report.json` 记录首个完整候选包验收：空模型／参考缓存开始，原始权重转换、新参考编码、重复请求、managed 重启、direct 启动、休眠与进程退出均通过。四次输出均为非空 32 kHz 单声道 PCM16，PCM SHA256 相同：`307f3c0469f9d655205d99aa4c115d353b3f9c04971b50185e5edd0152ef9176`。整个验收耗时 68.73 秒；这是包含准备、请求和退出的单次流程耗时，不是冷启动性能基准。

`outputs/macos-lifecycle-fragments/report.json` 使用两段显式换行的日文，默认 `cut0`。`fp32`、`low-memory`、`minimum-memory` 各执行两次完整请求，每次实际生成两个片段；在处理中取消后重新请求，PCM 与该策略的基准完全相同。普通推理未导入 Torch。关闭后 MLX cache 均为 0；active 分别为 4100、0、0 字节。这不是总进程内存或统一内存峰值测量。

此前的 `outputs/macos-lifecycle/report.json` 与 `outputs/macos-lifecycle-multisentence/report.json` 也保留。后者虽然输入较长，但 `cut2` 合并后只有一个片段，因此不计入多片段覆盖。最终 harness 明确检查实际片段数。

## 解压、离线隔离与搬迁

最终归档使用 macOS 自带 `/usr/bin/tar` 解压到 `/private/tmp/SakuraTTS-离线验收 20261004/`。17,171 个清单文件的 SHA256 全部一致；三个 `.command`、主／准备解释器和 FFmpeg 的执行权限保留。用解压后的 `check-runtime.command` 实际执行 Metal 矩阵乘法，通过且未导入 Torch。

验收使用 `sandbox-exec` 启动包内准备解释器和同一份首次使用 harness。系统规则禁止外网，仅允许 localhost TCP；禁止读取 `/Users/beyondpower`、`/opt/homebrew`、`/usr/local`，因而开发源码、原始实验目录、venv 和用户缓存均不可用。原始模型与参考复制到独立验收目录。负向检查确认读取开发文件、连接外网均返回 `EPERM`，本地回环连接正常；规则未修改系统安全设置。

在上述约束下，从空模型／参考缓存完成原始转换、新参考编码、四次 HTTP 音频请求、managed 休眠／重启和 direct 模式，整个验收耗时 70.97 秒。四次 PCM 均与前次普通环境验收完全相同，准备和应用进程均使用包内解释器，所有服务正常退出。

随后将整个包连同缓存重命名为 `再次搬迁 SakuraTTS`，同时把原始输入目录重命名为 `迁移后的 角色文件`，旧目录已不存在。在相同沙箱规则下执行 `--reuse-cache`，managed／direct 均通过：模型和参考缓存清单不变，没有重新执行准备，PCM 与搬迁前完全相同。

上述原始报告、音频、日志、沙箱规则和负向检查保存在本地 `outputs/macos-isolated-final/`，汇总为 `summary.json`。这证明当前机器上的解压、独立运行与搬迁，不等同于新操作系统安装或其他硬件验收。

## 候选产物

完整包位于 `dist/SakuraTTS-macOS-AppleSilicon-final/`，清单记录 17,171 个文件、2,410,161,077 字节；该字节数不含清单本身。包内不含用户声线权重、参考音频和测试缓存。

产品 wheel SHA256 为 `83a2ee9772e8df2f3f69a2d50c46f74186d78601e7818edc292684fd99dec0ad`。离线构建同时检查 wheel 与源码包中的产品文件及许可证是否与选择的源码快照一致。

压缩包为 `dist/macos-release-final/SakuraTTS-macOS-AppleSilicon-final.tar.gz`，964,488,441 字节（约 919.8 MiB），压缩耗时 110.56 秒。SHA256：`da8ba681b1eaccad5389fead6a40e693a658bfc1551401f99df83130d2eb8096`。归档含清单在内共 17,172 个文件，逐文件内容校验通过；SHA256 文件与 `compression-report.json` 在同一目录。

## 未验收范围

- macOS 14／15、M1／M2／M3 等其他芯片、较小内存机器及第二台干净机器尚未实测；Intel Mac 不在此 arm64 包范围。
- V2ProPlus、FP16 和其他模型家族尚未接入 MLX；当前验收只使用上述一组模型与参考。
- 未执行人工听音、ASR、完整官方音频对照、长时间压力和峰值内存验收。PCM 可重复不等于内容、音色或自然度合格。
- 未验收浏览器下载后 Gatekeeper／quarantine 的启动行为，未完成签名公证。HuBERT／ERes2Net 权重归属材料及 FFmpeg 对应源码提供方式仍须在公开分发前核对。
- 本次在 macOS 执行共享测试，未重新在 Windows 硬件上构建和验收。

构建和使用方法由 [Mac 整合包指南](../../docs/portable-macos.md)、[Apple 指南](../../docs/apple.md)维护。
