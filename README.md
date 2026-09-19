# SakuraTTS

SakuraTTS 计划提供一套兼容 GPT-SoVITS 模型的轻量 GPU 推理引擎，用于 Sakura 桌宠的实时语音。目标是在保留模型能力和语音质量的前提下，降低安装体积、显存占用和生成延迟。

现有 GPT / SoVITS 权重通过转换工具生成推理模型包，再由自己的运行时生成音频。转换与验证可以使用 PyTorch；日常推理运行包以不依赖完整 PyTorch 环境为目标。桌宠现有 CPU TTS 方案继续沿用。

2026-09-19 在 Apple M4 上，使用“朱雀院红叶”V2Pro 完成了官方与 Lite 对照，并保存用户语音问题的消融与复听证据。自有 GPT、采样和完整声学计算已接通：使用准备好的官方条件，十例生成 token、停止和最终波形通过对照；两个原始回归样例的四条官方 / 自有样音经用户确认正常且无明显差异。独立诊断仍有三个采样概率值和一个 MRTE 中间值超差，尚未完成整体数值验收。

已验证的改动包括复用 GPT 权重、降低声码器工作区、可选按请求释放 GPT，以及三个权重归档无损减少 801.37 MiB。资源数据按文本长度和运行策略分别记录，不能用短句的分配器峰值代表完整 TTS。中文音素 / BERT 和日文语言段已通过独立对照，约 203 KiB 的单参考条件包已支持独立重载；上层路由与原始文本整链仍在接通。当前不宣称全模型兼容或完整运行包交付。

## 文档

| 文档 | 内容 |
|---|---|
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

生产以 Windows / NVIDIA CUDA 为重点。当前 Mac 阶段先完成可迁移的模型包、文本与参考准备、静态权重计算、KV 和资源生命周期；Mac 专用后端调优暂缓。首个样本为“朱雀院红叶”V2Pro，CUDA 性能留待目标设备验证。

当前在 Mac 上实现 MLX / NumPy 计算和必要的 CPU ONNX 模块，按实测接通文本、参考条件和资源生命周期。单活动请求、`batch=1` 是当前范围。FP16 和 NVIDIA CUDA 仍是后续候选，需要保留已验证的高精度对照并独立实测；其他模型逐组验证，不默认兼容。

C++ 原生运行时是拟议方向。TensorRT-RTX、ONNX Runtime CUDA 和少量原生 CUDA 算子需要通过实际模型验证后取舍，尚未选定生产后端。Windows 方向已确定，具体 NVIDIA GPU 与性能预算仍待固定。通用策略完成后，若剩余工作必须依赖 NVIDIA 硬件，就转入 Windows 接续，不继续深挖 Metal 特性。

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

空目录通过 `.gitkeep` 保留，添加实际文件后可移除对应占位文件。

## 开发

开发约定见 [AGENTS.md](AGENTS.md)。三个上游仓库、独立 Python 环境、模型和运行结果放在本机同级目录 `../SakuraTTS-References/`，不随本仓库上传。本仓库的 [reference_smoke.py](harness/reference_smoke.py) 负责调用上游、保存音频及诊断信息；具体运行命令见 [Mac 首轮验证](docs/experiments/2026-09-19-macos-reference-smoke.md)。

本地配置使用 `.env`，需要共享配置格式时提供不含凭据的 `.env.example`。
根目录下的 `data/`、`models/` 和 `outputs/` 用于本地运行数据、模型与生成结果，已加入 Git 忽略规则。
