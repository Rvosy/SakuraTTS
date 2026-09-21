# 架构与目录

公共入口是 Engine、Model 和 Audio。CLI 和 HTTP 调用相同 Engine，Engine 将请求交给现有 CUDA 编排；GPT、采样、声学和随机数消耗顺序没有因目录调整而改变。

| 位置 | 职责 |
| --- | --- |
| `engine.py` / `model.py` | 公共生命周期、输出和模型目录 |
| `converter.py` | 模型转换与独立参考准备，成功后发布目录 |
| `reference.py` | 将原版音频路径和文本解析成条件，透明复用音频缓存 |
| `frontend/` | 文本分段、日文和中文前端；派生代码保留许可边界 |
| `backends/cuda/` | CuPy GPT 与 Windows 推理编排 |
| `backends/onnx/` | SoVITS 图、分块验证及独立进程适配 |
| `backends/mlx/` | Mac 实验实现，未接入公共 Engine |
| `_internal/` | 采样、生成、参考、权重、诊断和私有 IPC |
| `_internal/conversion/` | 仅离线转换时使用的导出与参考准备 |

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
