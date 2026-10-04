# V2ProPlus MLX 适配与验证（2026-10-04）

本次实现将 V2ProPlus 接入 Apple silicon 的原生 MLX 路径，沿用公共转换、Engine、HTTP、参考准备和三种内存策略。修改位于 `feat/mlx-v2proplus` 工作树，基准提交为 `da0d6916a34f1cf113307fe8992b7c6184bd804a`；下述候选包由未提交源码构建，尚未发布。代码与外部项目的结构核查见[架构记录](mlx-v2proplus-architecture-20261004.md)，使用方法见[Apple 指南](../../docs/apple.md)。

## 环境与输入

实机为 Apple M4、16 GiB 统一内存，macOS 26.5.2（25F84）。候选包使用独立 CPython 3.11.15、MLX / mlx-metal 0.32.2、NumPy 2.4.6；准备组件为 CPU PyTorch 2.7.1。构建仍以 macOS 14 为目标，实际系统范围仅为本机。

| 输入 | SHA256 |
| --- | --- |
| 朱雀院红叶 GPT `朱雀院红叶-e15.ckpt` | `010197bfc30b04d991f2bf060f962549932a8278b98c137d92f980e9cca8c0e9` |
| 官方 V2ProPlus 声学 `s2Gv2ProPlus.pth` | `d42a22bbbf65fb2bbdd45ad6a66841156977db45c7aabe0a6992ff378d9c7d3b` |
| 参考 `VO02_0204.OGG` | `c265a87781115e67d787ed81a7d2a3755a0aff54dd15d6c304b5fe61b01b91b4` |
| 旧 V2Pro 回归 `朱雀院红叶_e8_s38928.pth` | `f8bd92196175f435ae9df2bc01e87b0119a5d94c0af81ada4573a9720536ab38` |

这是一组已有 GPT 与官方 Plus 基础声学权重的组合，用于验证 Plus 架构和产品流程，不代表某个定制 Plus 声线的音质验收。参考转写为 `じゃあ、私、もっと悪い子になっちゃおうな〜`。原始模型与参考未修改，也未放入发行包。

## 实现与回归

MLX 声码器已按 manifest 读取网络宽度、卷积核与上采样参数，无需另写一套 Plus 网络。加载、参考匹配、诊断和运行报告现均接受实际 Pro/Plus 家族。MLX 与 ONNX 导出器共用 `prepare/sovits_checkpoint.py`，检查权重家族、配置一致性、推理张量与参考维度；原始张量精度及导出布局保留。

前端与参考准备入口改为 `prepare_resources.py`。发行示例只选择后端与权重路径，不再由操作系统指定模型家族。公开 `convert --backend mlx` 的参数遗漏已修复；共享加载器支持准备解释器的隔离执行，避免 Windows `._pth` 下找不到相邻模块。

执行 `PATH="$PWD/outputs/macos-build/test-bin:$PATH" .venv/bin/python -m unittest discover -s tests`：406 项，400 项通过、6 项跳过。另核对了两个导出器在临时工作目录、`python -I -B` 下的入口；缓存回归覆盖共享加载器变化后重新转换。日志为 `outputs/mlx-v2proplus/tests-final.log`。本次未在 Windows / Linux 真机重跑。

旧 V2Pro 重新转换得到的 650 个张量、173,050,368 原始字节，其 dtype、shape 和内容全部与历史原生包一致，整个 `weights.npz` 哈希也相同。配置、输入输出、模块和 WeightNorm 元数据一致。报告在 `outputs/mlx-v2proplus/v2pro-regression/report.json`；该项验证转换结果，未重新运行旧模型合成。

## 官方固定输入数值对照

官方源码固定为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`。使用同一 Plus 权重、新准备的参考条件、语义 token、音素及显式噪声，先运行官方 CPU FP32，再运行 MLX CPU FP32 encoder 与 Metal FP32 flow / decoder，两个计算路径分进程执行。

16 个语义 token 和完整参考的 113 个 token 分别生成 0.64 秒与 4.52 秒波形。每组又分别执行绑定参考投影与普通路径，共四次。12 个中间阶段全部通过项目已有 `atol=1e-4, rtol=1e-5` 检查。两组波形最大绝对误差分别为 `4.00618e-5`、`9.15080e-5`；绑定与未绑定的全部阶段逐元素一致。MLX 回放进程未导入 Torch。

原始数组、执行脚本、官方与 MLX 报告保存在 `outputs/mlx-v2proplus/acoustic-parity/`。这是固定条件的声学数值验证，不是完整官方 TTS 输出、人工听音或性能对照。

## 公共入口与生命周期

公开 CLI 从原始 `.ckpt` / `.pth` 完成 MLX 转换和参考准备，生成的声学包记录 `v2ProPlus`。首次 CLI 运行曾因参数列表不含 `mlx` 被拒绝，修复前日志 `convert.log` 与修复后 `convert-after-cli-fix.log` 均保留。

`scripts/verify_mlx_runtime.py` 对 `fp32`、`low-memory`、`minimum-memory` 各执行两次两段日文请求，实际生成两个片段、共 8.08 秒音频；取消后重新请求，PCM 与该策略基准完全一致。普通推理未导入 Torch。关闭后的 MLX cache 为 0，active 分别为 4100、0、0 字节；这些计数不是进程总内存或统一内存峰值。报告和音频在 `outputs/mlx-v2proplus/lifecycle/`。

完整候选包的 `check-runtime.command` 成功执行 Metal 运算。从空缓存运行 `scripts/verify_portable_first_use.py`，检查原始转换、新参考、重复请求、managed 重启、休眠退出以及 direct 模式，全流程 68.16 秒。四个 HTTP 响应均为 32 kHz 单声道 PCM16、2.38 秒，PCM SHA256 相同：`a43732657290b8e17071cc52129e917a6ad61ec7051aeb6f25eb82f67fa5a129`。记录在 `outputs/mlx-v2proplus/first-use/`。首次流程含准备工作，不能视为推理延迟基准。

## 解压、隔离与搬迁

使用 macOS 自带 `tar` 解压到中文及空格路径，17,172 个清单文件的 SHA256 全部一致，三个启动器、主／准备解释器及 FFmpeg 的执行权限保留。沙箱禁止外网，并禁止读取 `/Users/beyondpower`、`/opt/homebrew`、`/usr/local`；只允许本地回环 TCP。负向检查确认开发文件读取和外网连接均返回 `EPERM`，本地 HTTP 可用。

在上述条件下从空缓存完成首次转换、新参考、四次 HTTP 请求及服务生命周期检查，耗时 73.34 秒。随后同时重命名程序目录和原始模型／参考目录，用 `--reuse-cache` 完成 managed 与 direct 验证，耗时 4.66 秒，搬迁前后的缓存清单一致。普通环境、隔离首次使用及搬迁后三轮的全部响应 PCM 哈希相同。

原始报告、音频、日志、沙箱规则和负向检查在 `outputs/mlx-v2proplus/isolated/`；测试目录位置在 `location.json`。汇总为 `outputs/mlx-v2proplus/summary.json`。这些结果证明当前 M4 上独立解压与搬迁可用，不代表旧系统或第二台设备验收。

## 候选产物

完整包：`dist/SakuraTTS-macOS-V2ProPlus/`。压缩包：`dist/macos-release-v2proplus/SakuraTTS-macOS-V2ProPlus.tar.gz`，964,476,137 字节（约 919.8 MiB）。SHA256：`665c9350e48180606a7e0aa26c7cb19107ad4b4279ccd0a65354605d18aa0bb8`。

主包仍支持 V2Pro，名称只标明本轮验证内容。产品 wheel 的 110 个 Python 文件与当前工作树逐文件一致，wheel SHA256 为 `37c6e60dd6434ccaca0d647cefd91a8899d386213019e13de125fd57315bd590`。归档按清单选择文件，含清单共 17,173 个文件，逐文件内容校验通过；模型、参考和运行缓存不进入压缩包。依赖及来源输入保存在包外 `outputs/mlx-v2proplus/bundle-inputs.json`，源码与 harness 哈希在 `source.json`。

## 未验收范围

- M1 / M2 / M3、其他内存容量和 macOS 14 / 15 尚未真机验证，Intel Mac 不适用此 arm64 包。
- FP16、其他模型家族、其他定制 Plus 权重和英文整链没有新增验收结论。
- 未执行人工听音、ASR、完整官方 TTS 音频对照、长文压力和峰值内存验收。数值一致或 PCM 可重复不等于音色、自然度和内容完整性通过。
- 浏览器下载后的 quarantine / Gatekeeper、签名公证和公开分发材料仍沿用[上一轮记录](macos-portable-20261004.md)的未验收项。
