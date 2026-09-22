# 架构与目录

公共入口是 Engine、Model 和 Audio。CLI 和 HTTP 调用相同 Engine，由 `backends.create_runtime` 选择执行后端。当前可运行的公共组合仍是 Windows / NVIDIA CUDA + V2ProPlus + 日文；目录中保留的中文和 MLX 研究模块不代表已经接通。默认 `direct` 和 FP32 保持不变，GPT、采样、声学和随机数消耗顺序也不变。

| 位置 | 职责 |
| --- | --- |
| `engine.py` / `model.py` | 公共生命周期、输出和模型元数据；不导入设备计算库 |
| `converter.py` | 模型转换与独立参考准备，成功后发布目录 |
| `reference.py` | 将原版音频路径和文本解析成条件，透明复用音频缓存 |
| `frontend/profiles.py` / `processors.py` | 已实现的语言模式、标点规则，以及音素和文本特征处理 |
| `frontend/runtime.py` / `text_frontend.py` | 按资源包装配前端，执行共享分句和语言路由；当前只装配日文 |
| `backends/__init__.py` | 显式选择已实现的后端，声明公共生命周期所需的 Runtime 接口 |
| `backends/cuda/` | CuPy GPT 与 Windows 推理编排 |
| `backends/onnx/` | SoVITS 图、分块验证及独立进程适配 |
| `backends/mlx/` | Mac 实验实现，未接入公共 Engine |
| `_internal/` | 采样、生成、参考、权重、诊断和私有 IPC |
| `_internal/conversion/` | 仅离线转换时使用的导出与参考准备 |
| `packaging/recipes/` / `scripts/build_portable.py` | 选择发行目标、后端、语言、服务和工作进程，再从本地组件装配整合包 |

## 配置的边界

`Model` 负责描述模型包含什么：权重和前端资源位置、声明的语言、建议后端，以及可选参考缓存。读取模型信息不检查当前机器是否装有 CUDA，也不根据当前发行包的能力禁止记录未来语言。真正执行时，后端 factory 和前端装配分别判断能否使用这些资源。把 `backend.preferred` 写成 `cpu`，或把 `languages` 改成 `["zh"]`，不会获得对应的推理能力。

设备选择可由 Python `Engine.load(..., backend=...)`、CLI `--backend` 或 HTTP 服务配置指定；显式参数覆盖配置和模型的建议值。factory 只有源码内的静态实现表，当前仅 `cuda`，不扫描插件、不自动更换设备。CUDA 的实验参数在 CUDA 模块解释，共享 Engine 只负责请求互斥、输出和关闭。`sakuratts capabilities` 列出已实现的后端与语言模式，不探测驱动，也不证明当前机器已经可以出声。

语言处理与设备后端分开。`TextFrontend` 使用语言处理器生成音素和特征；日文处理器保持原版的零 BERT 特征语义。`FrontendRuntime` 负责装配和关闭分段器、G2P 及字典。准备中文时，需要补齐中文处理器、BERT 执行与资源构建，不能只往语言表加一个名字。

安装位置在执行适配器中绑定。`acoustic_python` 和 `frontend_python` 分别表示声学和经典日文前端使用的解释器；旧配置未写 `frontend_python` 时沿用 `acoustic_python`。两种角色可以指向同一个 Python，当前整合包就这样减少重复文件；也可以使用各自满足 ABI 的环境。完整包中的 CPU 准备环境独立运行，普通推理不导入它的 PyTorch。

发行 recipe 选择已经实现的组合。当前 `windows-nvidia-ja.toml` 固定 Windows x64 / CUDA / 日文，HTTP 服务与准备组件可独立省略。各项选择、依赖闭包和工作进程路径写入发行清单，详见[整合包](portable-bundle.md)。新增目标应交付自己的运行组件及验收结果，再加入 recipe；无需把所有厂商运行库和语言模型都装进一个压缩包。

声学和经典前端 worker 通过 `_internal/worker.py` 只绑定指定的 SakuraTTS 包，不向私有 Python 暴露主环境整个 `site-packages`。包顶层使用延迟导入，兼容私有 Python 3.9 的加载过程。

HTTP 以原版 api_v2 的字段和默认值接入，在单一线程中创建、调用、切换和关闭引擎，不维护另一套采样流程。参考条件在请求时解析；模型目录不再要求绑定参考。按句流式在每片完成后发送 PCM，通过有界队列施加背压，客户端断开后在计算边界取消。模型内存策略和实验参数仍由后端负责；没有引入插件注册层或自动后端回退。

默认 `direct` 模式保留上述进程安排。显式选择 `managed` 后，`_internal/managed_runtime.py` 在事件循环中管理唤醒、保活与休眠，`inference_process.py` 通过有界 IPC 调用独立进程中的同一个 `Inference`。`inference_worker.py` 仍串行执行模型操作，`process_tree.py` 管理整棵自有进程树。Windows 使用 Job Object 处理正常退出和异常回收；控制进程通过标准库传输 PCM，不加载 NumPy、GPU 计算库或文本前端。具体取舍见 [可选推理进程控制 ADR](adr/0003-managed-runtime.md)。

`benchmarks/` 提供常用入口。`research/` 保存历史工具、报告和原始证据，仅保留在 Git 仓库，不进入 wheel 或源码包；相关测试由 `research/run_tests.py` 单独运行。保存过的失败项不能因归档而改写为通过。

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

先完成一个设备与一种语言的整链，再扩大组合。新增中文不应让日文包常驻中文特征模型；新增 AMD 或 Apple 不应让 Windows NVIDIA 包携带另一套设备库。当前 2 GB 压缩目标针对已在本机验证的 Windows 日文完整包，新组合的体积需要单独测量。架构取舍见 [ADR 0004](adr/0004-composable-components.md)。
