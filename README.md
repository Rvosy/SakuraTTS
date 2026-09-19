# SakuraTTS

SakuraTTS 是兼容已验证 GPT-SoVITS 权重的独立 GPU 推理项目，用于 Sakura 桌宠的语音生成。目标是在保留模型能力和语音质量的前提下，降低安装体积、显存占用和生成延迟。

现有 GPT / SoVITS 权重通过转换工具生成推理模型包，再由自己的运行时生成音频。转换与验证可以使用 PyTorch；日常推理运行包以不依赖完整 PyTorch 环境为目标。桌宠现有 CPU TTS 方案继续沿用。

## 安装与运行状态

本项目通过根目录 `pyproject.toml` 安装，产品源码位于 `src/sakuratts/`。Windows / NVIDIA 已接通日文原文到完整 WAV：GPT 使用自有 CuPy CUDA 执行器，SoVITS 使用独立 ONNX Runtime CUDA 工作进程。普通运行不导入 PyTorch，也不读取官方、Lite 或 Genie 目录。当前实机模型是 Sakura V2ProPlus，GPU 是 RTX 5060 8 GB；完整验收范围和未解决问题见 [Windows 实测记录](docs/experiments/2026-09-20-windows-nvidia-backend.md)。

先按 [Windows 安装与模型准备](docs/setup-windows-nvidia.md) 离线准备运行环境、模型和参考包，再执行：

```powershell
.venv-windows-runtime\Scripts\sakuratts.exe doctor --nvidia --config models\windows-sakura\runtime.json
.venv-windows-runtime\Scripts\sakuratts.exe synthesize --config models\windows-sakura\runtime.json --reference 中性 --text "おはよう。今日もよろしくね。" --output outputs\sakura-neutral.wav
```

`doctor` 报告依赖、包哈希和参考身份是否通过，不生成音频，也不代表内容或音质验收。转换和官方对照使用单独的开发环境；安装依赖、组件大小和当前双 Python 版本的部署边界在 Windows 指南中说明。Mac 用户继续使用已有的 [MLX 日文入口](docs/japanese-runtime.md)，通用目录说明见[安装指南](docs/setup.md)。

## 已有 Mac 验证

2026-09-19 在 Apple M4 上，使用“朱雀院红叶”V2Pro 完成了官方与 Lite 对照，并保存用户语音问题的消融与复听证据。自有 GPT、采样和完整声学计算已接通：使用准备好的官方条件，十例生成 token、停止和最终波形通过对照；两个原始回归样例的四条官方 / 自有样音经用户确认正常且无明显差异。独立诊断仍有三个采样概率值和一个 MRTE 中间值超差，尚未完成整体数值验收。

已验证的改动包括复用 GPT 权重、降低声码器工作区、可选按请求释放 GPT，以及三个权重归档无损减少 801.37 MiB。资源数据按文本长度和运行策略分别记录，不能用短句的分配器峰值代表完整 TTS。中文音素 / BERT 和日文语言段已通过独立对照，约 203 KiB 的单参考条件包已支持独立重载；日文已接通原始目标文本到 PCM，并在独立环境完成四例、32 次正常请求对照。原始回归日文 WAV 与用户此前确认的样音逐字节相同；其余日文样例未新增试听。当前不宣称全模型兼容或完整运行包交付。

目前可通过[日文命令行入口](docs/japanese-runtime.md)直接输入原文并生成 WAV。新参考准备工具已从原始音频重建同一参考条件，五组数组与旧包相同；普通生成使用独立参考包和 NumPy 随机数，不读取历史实验的目标特征或 token。两个新 seed 已正常生成，离线 ASR 均识别到开头和三句内容，新样音的发音与音色仍待人工试听。

连续请求可[复用模型并释放当前 GPT 状态](docs/experiments/2026-09-19-native-model-lifecycle.md)：四条日文实测每次省约 0.41–0.44 秒，空闲 MLX active 比保留旧 KV 少 96 MiB。再将释放[提前到声学之前](docs/experiments/2026-09-19-gpt-state-before-acoustic.md)，四例请求分配器峰值少约 96 MiB，本轮耗时增加 2–16 毫秒，WAV 仍逐字节一致。RSS 峰值没有改善；需要更低空闲常驻时仍可卸载全部模型。

## 文档

| 文档 | 内容 |
|---|---|
| [Windows 安装与推理](docs/setup-windows-nvidia.md) | 离线运行环境、经典日文前端、V2ProPlus 转换、参考准备和 WAV 命令 |
| [Windows 实测记录](docs/experiments/2026-09-20-windows-nvidia-backend.md) | RTX 5060 官方对照、显存、速度、数值与已知问题 |
| [Lite / Genie 定点研究](docs/research/windows-nvidia-reference-implementations.md) | 固定版本的缓存、采样、资源生命周期和可复用边界 |
| [日文运行入口](docs/japanese-runtime.md) | 当前可运行范围、普通合成命令与离线参考准备 |
| [推理行为与兼容契约](docs/specs/inference-contract.md) | 产品范围、模型兼容、解码语义、流式输出和资源生命周期 |
| [原生 GPU 运行时方案](docs/adr/0001-native-gpu-runtime.md) | 转换器与运行包的边界、模块分工、后端候选及取舍 |
| [研究与证据](docs/research/gpu-inference.md) | 三个参考仓库、相关论文、性能数据的适用范围和待核实线索 |
| [基准与验收协议](docs/specs/benchmark-protocol.md) | 如何公平比较体积、显存、速度、质量与稳定性 |
| [实施路线](docs/roadmap.md) | 分阶段交付物、完成条件、首轮实验和待确定事项 |
| [调用链与资源实验](docs/experiments/2026-09-19-parity-and-lifecycle.md) | 用户语音问题、固定历史数值对照、参考条件释放与证据 |
| [参考资源与 BERT 裁剪](docs/experiments/2026-09-19-bert-and-reference-memory.md) | 输出保持、常驻边界、独立内存采样与测量限制 |
| [声学准备与常驻](docs/experiments/2026-09-19-acoustic-lifecycle.md) | 同波形常驻下降、RSS 代价和完整退出记录 |
| [文本条件与试听](docs/experiments/2026-09-19-asr-review.md) | 消融、离线 ASR、用户对指定样音的复听结果 |
| [采样与停止规则](docs/experiments/2026-09-19-sampling-semantics.md) | 无 PyTorch 采样基线及尚未解决的 Top-p 边界 |
| [扩展回归](docs/experiments/2026-09-19-expanded-regression.md) | 10 条输入的波形保持和英文资源修复 |
| [GPT 缓存与速度](docs/experiments/2026-09-19-gpt-capacity.md) | 正常固定历史测量、容量成本和长句数值失败 |
| [长句数值定位](docs/experiments/2026-09-19-mlx-numerics.md) | Prefill 舍入放大与通用高精度候选 |
| [固定条件声学对照](docs/experiments/2026-09-19-sovits-fixed-conditions.md) | 同语义、音素、音色与显式噪声的 SoVITS 对照 |
| [自有 MLX GPT](docs/experiments/2026-09-19-mlx-gpt.md) | 独立转换、无 PyTorch 的语义计算与 Metal 数值对照 |
| [自有历史生成](docs/experiments/2026-09-19-native-gpt-generation.md) | 共享实采噪声下的自行采样、停止与语义切片 |
| [自有生成接声学](docs/experiments/2026-09-19-native-prepared-speech.md) | 准备好条件后的完整计算链、无插桩请求时间和资源 |
| [十例整链生成](docs/experiments/2026-09-19-expanded-prepared-speech.md) | 扩展输出对照、试听文件保持和计量范围 |
| [十例采样边界](docs/experiments/2026-09-19-expanded-native-generation.md) | token 保持与三个概率值超差 |
| [独立中文 BERT](docs/experiments/2026-09-19-mlx-bert.md) | CPU 数值与正常成本、GPU 未解决误差 |
| [声学解码包](docs/experiments/2026-09-19-sovits-package.md) | 权重转换、参考条件边界与严格重载 |
| [自有声学编码器](docs/experiments/2026-09-19-mlx-sovits-encoder.md) | 码本、相对注意力、MRTE 和分布参数的独立实现 |
| [完整声学计算](docs/experiments/2026-09-19-mlx-sovits-complete.md) | reverse flow、声码器、设备选择、正常速度和缓存代价 |
| [扩展声学回归](docs/experiments/2026-09-19-expanded-acoustic.md) | 十例波形通过与保留的 MRTE 中间值失败 |
| [无损权重存储](docs/experiments/2026-09-19-lossless-weight-storage.md) | 归档体积、恢复后的逐位一致性及读取代价 |
| [GPT 权重复用](docs/experiments/2026-09-19-gpt-prefill-weight-reuse.md) | 去掉 Prefill 重复读取和校验的实测收益 |
| [声码器工作区](docs/experiments/2026-09-19-decoder-workspace.md) | 调整求值时机、波形保持和峰值 / 延迟取舍 |
| [GPT 生命周期](docs/experiments/2026-09-19-gpt-lifecycle.md) | 按请求释放语义模型和重新加载的成本 |
| [声学数值定位](docs/experiments/2026-09-19-attention-softmax-numerics.md) | 首层 softmax 舍入及后续验证方向 |
| [声学精度与成本](docs/experiments/2026-09-19-softmax-candidates.md) | 保留 FP32、显式 CPU 累积选项与十例 120 阶段通过 |
| [文本前端依赖](docs/experiments/2026-09-19-text-frontend-dependencies.md) | 中文 G2PW、日文词典与混合语言能力边界 |
| [独立 tokenizer](docs/experiments/2026-09-19-tokenizer-runtime.md) | 移除 Transformers 后的词元与输入数组对照 |
| [G2PW 输入迁移](docs/experiments/2026-09-19-g2pw-inputs.md) | 查询截断、字符映射和输入准备的官方对照 |
| [G2PW 文本准备](docs/experiments/2026-09-19-g2pw-text.md) | OpenCC、查询上下文和 PyPinyin 回退迁移 |
| [G2PW 模型推理](docs/experiments/2026-09-19-g2pw-onnx.md) | 独立 ONNX 接口、官方概率一致性与实际依赖 |
| [G2PW 拼音闭环](docs/experiments/2026-09-19-g2pw-pinyin.md) | 文本到模型预测再填回拼音的完整对照 |
| [中日文前端](docs/experiments/2026-09-19-chinese-phones.md) / [日文对照](docs/experiments/2026-09-19-japanese-g2p.md) | 语言段迁移、实际能力与依赖证据 |
| [参考条件包](docs/experiments/2026-09-19-reference-condition-package.md) | 单参考身份绑定、最小产物与独立重载 |
| [G2PW 内存映射](docs/experiments/2026-09-19-g2pw-mapped-ort.md) | 同输出降低 CPU RSS、加载与部署边界 |
| [G2PW 释放观察](docs/experiments/2026-09-19-g2pw-lifecycle.md) | 三次创建与关闭后的 RSS 边界 |
| [兼容矩阵](docs/specs/compatibility-matrix.md) | 各模型、语言和功能的实测范围与待验证项 |
| [Mac 首轮验证](docs/experiments/2026-09-19-macos-reference-smoke.md) | 实际环境、运行命令、样音、耗时、资源记录和已知限制 |

建议先读推理契约和运行时方案，再看实施路线；研究资料用于解释技术依据，基准协议用于判断优化是否有效。

## 第一版方向

近期先支持日文，中文已有代码与证据保留，待日文运行链和通用优化完成后单独恢复。生产以 Windows / NVIDIA CUDA 为重点。Windows 复用了 Mac 阶段的模型包、文本与参考准备、采样和生命周期代码，另行适配 Sakura V2ProPlus；Mac 专用后端调优暂缓。

单活动请求、`batch=1`、FP32 和非流式完整 WAV 是当前 Windows 范围。其他精度、权重和模型家族需要独立验证；Mac 的 V2Pro 对照不能替代 Windows 的 V2ProPlus 验证。相同 seed 也不保证不同后端抽到同一序列，固定数值对照使用显式随机输入回放。

当前只维护一条 Windows 计算路径：CuPy CUDA GPT 与 ORT CUDA 声学解码。C++、TensorRT 和量化仍是候选；不以移植更多后端作为完成条件。速度和显存取舍以完整请求实测为准，质量、干净机器安装和宿主集成仍须分别验收。

## 目录结构

```text
SakuraTTS/
├── AGENTS.md       # 项目开发约定
├── src/            # 产品源码
├── tests/          # 自动化测试
├── harness/        # 产品行为验证入口
├── scripts/        # 开发和维护脚本
└── docs/
    ├── specs/      # 行为契约与验收协议，正文注明实现状态
    ├── adr/        # 架构决策及提案
    ├── research/   # 外部证据与研究线索
    ├── experiments/ # 本地实验记录与复现说明
    └── roadmap.md  # 实施阶段与待确定事项
```

安装文件与本地模型目录的完整说明见[项目结构](docs/setup.md#代码和资源放在哪里)。

## 开发

开发约定见 [AGENTS.md](AGENTS.md)。早期 Mac 实验将上游仓库、环境和资源放在 `../SakuraTTS-References/`；这是实验布局，普通合成不要求该目录存在。当前部分模型转换与参考准备工具仍依赖其中的固定上游源码，详见[安装指南](docs/setup.md)。[reference_smoke.py](harness/reference_smoke.py) 只用于调用上游作对照，历史命令见 [Mac 首轮验证](docs/experiments/2026-09-19-macos-reference-smoke.md)。

本地配置使用 `.env`，需要共享配置格式时提供不含凭据的 `.env.example`。
根目录下的 `data/`、`models/` 和 `outputs/` 用于本地运行数据、模型与生成结果，已加入 Git 忽略规则。
