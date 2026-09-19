# SakuraTTS

SakuraTTS 计划提供一套兼容 GPT-SoVITS 模型的轻量 GPU 推理引擎，用于 Sakura 桌宠的实时语音。目标是在保留模型能力和语音质量的前提下，降低安装体积、显存占用和生成延迟。

现有 GPT / SoVITS 权重通过转换工具生成推理模型包，再由自己的运行时生成音频。转换与验证可以使用 PyTorch；日常推理运行包以不依赖完整 PyTorch 环境为目标。桌宠现有 CPU TTS 方案继续沿用。

仓库已有设计文档和上游参考验证脚本。2026-09-19 在 Apple M4 上，使用“朱雀院红叶”V2Pro 模型完成了官方与 GSV-TTS-Lite 的 MPS / FP32 中、日文合成。自有原生推理引擎和模型转换器尚未实现；这次验证也不代表已达到性能或音质验收标准。

## 文档

| 文档 | 内容 |
|---|---|
| [推理行为与兼容契约](docs/specs/inference-contract.md) | 产品范围、模型兼容、解码语义、流式输出和资源生命周期 |
| [原生 GPU 运行时方案](docs/adr/0001-native-gpu-runtime.md) | 转换器与运行包的边界、模块分工、后端候选及取舍 |
| [研究与证据](docs/research/gpu-inference.md) | 三个参考仓库、相关论文、性能数据的适用范围和待核实线索 |
| [基准与验收协议](docs/specs/benchmark-protocol.md) | 如何公平比较体积、显存、速度、质量与稳定性 |
| [实施路线](docs/roadmap.md) | 分阶段交付物、完成条件、首轮实验和待确定事项 |
| [调用链与资源实验](docs/experiments/2026-09-19-parity-and-lifecycle.md) | 用户语音问题、固定历史数值对照、参考条件释放与证据 |
| [兼容矩阵](docs/specs/compatibility-matrix.md) | 各模型、语言和功能的实测范围与待验证项 |
| [Mac 首轮验证](docs/experiments/2026-09-19-macos-reference-smoke.md) | 实际环境、运行命令、样音、耗时、资源记录和已知限制 |

建议先读推理契约和运行时方案，再看实施路线；研究资料用于解释技术依据，基准协议用于判断优化是否有效。

## 第一版方向

当前按“先 Mac、后 Windows”的顺序验证。Mac 使用 MPS / FP32 跑通上游实现，首个样本已确定为“朱雀院红叶”V2Pro。Windows 验证暂缓。

原生运行时仍以 NVIDIA CUDA、单活动请求、`batch=1` 和 FP16 正确性基线为候选起点，随后优化执行图、缓存、资源复用和流式生成。其他模型逐组验证，不默认兼容。

C++ 原生运行时是拟议方向。TensorRT-RTX、ONNX Runtime CUDA 和少量原生 CUDA 算子需要通过实际模型验证后取舍，尚未选定生产后端。首发系统、目标 NVIDIA GPU 与性能预算仍待确定，Mac 的跑通结果不能替代 CUDA 验证。

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
