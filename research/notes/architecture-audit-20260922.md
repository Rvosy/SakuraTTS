# 架构与上游源码核查记录

记录日期：2026-09-22。以下保留当时对固定上游版本和本地整合包的核查；现行模块职责见[架构说明](../../docs/architecture.md)。

## 与上游的组织方式对照

以下对照在 2026-09-21 核查了固定版本的源码目录与依赖声明，未将上游目录结构视为性能证据。

| 项目 | 已核查的组织方式 | SakuraTTS 的取舍 |
| --- | --- | --- |
| [GPT-SoVITS `48b1a016`](https://github.com/RVC-Boss/GPT-SoVITS/tree/48b1a0169a28582a8984402f82cf438d3bfa6aca) | 根目录提供 `api.py`、`api_v2.py`、`webui.py`、训练及推理 Notebook；`GPT_SoVITS/` 与 `tools/` 支撑推理、训练和数据处理；依赖集中在 requirements | 保留熟悉的服务启动参数，安装、调用与转换通过 `sakuratts` 包进入。训练、WebUI 和数据集制作不在本仓库的产品范围内 |
| [GSV-TTS-Lite `6c049397`](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/pyproject.toml) | `pyproject.toml` 声明可安装的 `gsv-tts-lite` 推理包；核心依赖包含 transformers、语言处理与音频库 | 沿用可安装引擎包的方式，保留运行、服务、语言和离线转换 extras，避免普通日文请求加载转换依赖 |

文件合并优先用于重复说明：环境准备归入开发指南，兼容范围归入兼容矩阵，逐次研究结论收拢到 `research/`。API、模型格式和推理档位分别服务不同使用任务，继续单列。

CUDA 语义生成、ONNX 声学、独立进程协议与离线转换具有不同依赖和生命周期，继续保持模块边界。`api.py` 和 `sakuratts.nvidia` 是已有调用方的薄兼容入口；其实现仍复用包内代码。MLX 和中文模块保留已有回归，不把它们写成公共 Engine 已支持的后端。

## 本地原版整合包与后续扩展

2026-09-22 又检查了本机已有的 `D:/Project/sakura/tts/g50`，包括 `go-webui.bat`、安装脚本、requirements 和 `TTS_infer_pack` 源码。此次没有联网，也没有下载其他发行包。以下结论只对应这份本地副本。

原版 Windows 启动器把自己的 `runtime` 加入 PATH，使用 `runtime/python.exe -I webui.py` 启动。CPU / CUDA / ROCm 主要通过安装不同的 PyTorch 构建来选择；运行时通过 PyTorch 的 device 使用模型。同一 requirements 同时覆盖训练、WebUI、音频处理及多种语言。这种方式便于复用原模型代码，但不同设备发行版仍需要各自的二进制依赖，也会携带 SakuraTTS 日文后台推理不需要的功能。

| 扩展 | 本地原版中的做法 | SakuraTTS 已有基础与剩余工作 |
| --- | --- | --- |
| 中文与其他语言 | `TextPreprocessor` 按语言调用清洗和音素处理；中文执行 BERT，其他语言生成零 BERT 特征 | 已分开语言 profile、处理器与资源装配。中文规则、G2PW 和 BERT 有研究代码，仍需组合前端资源、选择特征执行后端、扩展转换和新参考准备，并验证混合语言及音素对齐 |
| CPU | Windows 安装脚本支持 CPU PyTorch；TTS 配置支持 CPU 并关闭半精度 | ORT 声学已有 CPU 路径，公共 GPT 仍是 CUDA。需实现 CPU GPT、明确执行精度和线程策略，串起转换、参考、取消与整链测试；CPU 准备组件不等于 CPU TTS |
| AMD | Linux 脚本支持 ROCm PyTorch，并检查 `/opt/rocm`；该 Windows 脚本只有 CUDA 和 CPU | 必须先确定目标操作系统与 GPU 范围，再选择语义和声学实现。现有 CuPy CUDA 内核及 ORT CUDA 包不能直接在 AMD 上运行；Linux ROCm 路径也不能当作 Windows AMD 已支持 |
| Apple | README 提及 MPS / CPU；本地安装脚本的 MPS 选项共用 CPU 安装分支，TTS 代码有 MPS 清理路径 | 仓库已有 MLX GPT / SoVITS 研究实现，尚需公共 Runtime 适配、模型转换与前端资源装配、macOS 工作进程和发布构建，以及数值、睡醒和内存验收 |

先完成一个设备与一种语言的整链，再扩大组合。新增中文不应让日文包常驻中文特征模型；新增 AMD 或 Apple 不应让 Windows NVIDIA 包携带另一套设备库。当前 2 GB 压缩目标针对已在本机验证的 Windows 日文完整包，新组合的体积需要单独测量。架构取舍见 [ADR 0004](../../docs/adr/0004-composable-components.md)。
