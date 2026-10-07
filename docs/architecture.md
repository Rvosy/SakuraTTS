# 架构与目录

SakuraTTS 参考 GPT-SoVITS 的职责划分组织代码：根目录放服务入口与核心包，核心包内按推理编排、文本、语义生成和声学模块分开。公共 Python 入口仍是 `Engine`、`Model` 和 `Audio`。当前兼容范围和完整兼容目标见[兼容矩阵](specs/compatibility-matrix.md)。

```text
api.py / api_v2.py / start-server.bat
sakuratts/
  TTS_infer_pack/   推理配置、模型切换、请求编排、文本准备
  text/            语言分段、规范化、G2P 与语言资源
  AR/              自回归语义生成、采样与停止规则
  module/          共用声学执行、参考条件与权重存储
  backends/        CPU / CUDA / DirectML / MLX 设备实现，共用 ORT 装配在 ort.py
  prepare/         离线转换、资源准备与转换缓存
  runtime/         进程、IPC、取消、日志与安装位置
  diagnostics/     环境依赖、设备与模型资源检查
  engine.py / model.py / profiles.py / server.py / cli.py / benchmark.py
tools/             可独立运行的模型和语言资源工具
scripts/           环境安装、构建与整合包验收
packaging/          发行组合及资源来源清单
tests/             产品回归
benchmarks/        常用基准入口
research/          实验代码、历史报告与冻结证据
docs/              使用说明、Spec 与 ADR
```

| 位置 | 职责 |
| --- | --- |
| [engine.py](../sakuratts/engine.py) | 公共 Engine 生命周期、请求互斥、Audio 输出 |
| [TTS_infer_pack/TTS.py](../sakuratts/TTS_infer_pack/TTS.py) | Inference 的模型激活、切换恢复、参考选择与请求执行 |
| [TTS_infer_pack/config.py](../sakuratts/TTS_infer_pack/config.py) | 读取上游 YAML 和 SakuraTTS 执行配置，不加载计算资源 |
| [TTS_infer_pack/TextPreprocessor.py](../sakuratts/TTS_infer_pack/TextPreprocessor.py) | 目标文本准备、语言路由与音素特征组合 |
| [TTS_infer_pack/text_segmentation_method.py](../sakuratts/TTS_infer_pack/text_segmentation_method.py) | 固定上游的 `cut0`–`cut5` 与长文本切分 |
| [TTS_infer_pack/runtime.py](../sakuratts/TTS_infer_pack/runtime.py)、[synthesis.py](../sakuratts/TTS_infer_pack/synthesis.py) | 各设备共用的模型生命周期与逐片合成流程 |
| [model.py](../sakuratts/model.py) | 模型描述、资源路径和参考名称关系 |
| [profiles.py](../sakuratts/profiles.py) | 后端执行预设、显式选项合并与精度包匹配 |
| [prepare/converter.py](../sakuratts/prepare/converter.py)、[cache.py](../sakuratts/prepare/cache.py) | 模型转换、转换缓存身份与产物发布 |
| [prepare/sovits_checkpoint.py](../sakuratts/prepare/sovits_checkpoint.py) | MLX 与 ONNX 导出共用的官方声学权重加载、模型构造及张量校验 |
| [prepare/prepare_resources.py](../sakuratts/prepare/prepare_resources.py) | 各后端共用的日文前端资源准备与参考音频编码 |
| [TTS_infer_pack/reference.py](../sakuratts/TTS_infer_pack/reference.py) | 请求参考解析、音频条件缓存与转写特征 |
| [text/](../sakuratts/text) | 语言模式、语言分段、规范化、音素和文本特征组件 |
| [AR/](../sakuratts/AR) | 语义生成循环、采样、EOS 与长度停止规则 |
| [module/](../sakuratts/module) | ONNX 声学、分块、参考条件及共享权重读取 |
| [backends/](../sakuratts/backends) | 静态后端选择、CUDA / CPU / DirectML / MLX GPT 与声学；CPU 与 DirectML 共用 `ort.py` 的装配 |
| [runtime/](../sakuratts/runtime) | 私有工作进程、IPC、取消、休眠、日志与安装绑定；`ort_process.py` 与 `ort_worker.py` 分别是声学进程的客户端和服务端 |
| [diagnostics/](../sakuratts/diagnostics) | `environment.py` 检查依赖与设备，`resources.py` 检查 ONNX 资源和私有解释器，`mlx.py` 检查原生 Apple 资源 |
| [cli.py](../sakuratts/cli.py)、[benchmark.py](../sakuratts/benchmark.py) | 命令行参数与操作分发、公共 Engine 完整请求测量 |
| [prepare/](../sakuratts/prepare) | 独立准备环境中的导出与参考编码 |
| [server.py](../sakuratts/server.py) | HTTP 请求边界、响应与服务生命周期 |
| [packaging/recipes/](../packaging/recipes)、[build_portable.py](../scripts/build_portable.py) | 发行组合与本地运行组件装配 |

## 如何定位代码

一次 HTTP 请求从 `server.py` 进入 `TTS_infer_pack/TTS.py`，经公共 `Engine` 调用所选后端。各后端继承共享的 `InferenceRuntime`，复用文本准备与逐片合成流程；具体计算实现放在各自设备目录。CPU 与 DirectML 的配置和 ORT 声学装配由 `backends/ort.py` 共用，设备目录中的 `engine.py` 声明各自引擎。

`TTS_infer_pack/runtime.py` 管理模型与合成生命周期；`runtime/` 管理解释器、工作进程及通信。模型和参考音频的转换进入 `prepare/`，环境及资源问题从 `diagnostics/` 查起。发行包如何组合这些组件由 `packaging/recipes/` 描述，构建入口在 `scripts/`。

文件按职责和依赖边界拆分。文本资源装配与文本处理、模型生命周期与合成算法分别保留；PCM 编解码和取消类型的小模块供多个入口使用，控制进程可以在不导入数值库的情况下使用它们。`converter.py` 和 `nvidia.py` 保留已公开的导入方式，实现分别归入 `prepare/` 和 `backends/cuda/`。

## 配置与校验所有权

`Model.load` 检查模型描述和资源目录，读取元数据时不加载计算库。模型可声明语言及建议后端；执行能力由后端选择与前端装配判断。字段、路径和覆盖顺序集中在[模型格式](model-format.md)。

`backends.require_backend` 检查已实现的后端，`create_runtime` 导入对应实现。各后端解释自己的执行选项并校验实际使用的资源，`TTS_infer_pack.runtime.InferenceRuntime` 共享模型生命周期和合成流程。公共 Engine 负责请求互斥、输出与关闭。`sakuratts capabilities` 查询源码中实现的能力，依赖与设备检查使用 `sakuratts doctor`。

模型家族由声学 checkpoint 和转换产物描述，设备后端负责执行方式。V2Pro 与 V2ProPlus 共用官方模型加载入口；MLX 和 ONNX 转换器分别输出各自需要的权重包和计算图。MLX 按产物配置读取层数、通道和卷积参数，参考条件必须匹配声学模型家族。整合包示例只选择后端，不根据操作系统指定模型家族。具体支持范围及验证结果由兼容矩阵维护。

`text.profiles` 定义已实现的语言模式，`TextPreprocessor.TextFrontend` 负责共享请求准备，语言处理器生成音素和特征，`TTS_infer_pack.frontend.FrontendRuntime` 装配资源并管理前端进程。`text.LangSegmenter` 只负责语言分段。日文沿用原版零 BERT 特征；中文研究代码尚未接入公共前端。

`SpeechRequest` 负责 HTTP 字段、默认值及请求能力校验。通过检查后，HTTP 调用同一个 `Inference` 和 Engine。参考缓存复用音频条件，转写与语言特征按请求生成。

## 进程与资源

CUDA 声学和经典日文前端使用私有工作进程。`acoustic_python`、`frontend_python` 可以共享解释器；旧配置省略后者时沿用前者。工作进程通过 `runtime/worker.py` 绑定 SakuraTTS 包，保持主环境与私有 Python 的 ABI 隔离。CPU / DirectML 的 GPT 与声学 Session 在推理主解释器中运行；CPU 使用 ORT INT8 GPT 与 FP32 声学，DirectML 使用 GPU FP16 GPT 与全图 FP16 声学。精度、KV 所在设备和验证边界见 [CPU 与 AMD 推理](cpu-amd.md)。MLX 的 CPU / Metal 分工与模型限制见 [Apple 指南](apple.md)。

完整整合包另带 CPU 准备环境，按需转换权重或编码参考。转换完成后退出，参考编码的短暂复用与回收见 [HTTP 指南](http-api.md#生命周期与输出)。前端与参考准备使用同一套 `prepare_resources.py`，由安装配置选择准备解释器。执行适配器从 `runtime/portable.json` 绑定安装位置；模型文件保存资源描述。发行组合与依赖来源见[整合包](portable-bundle.md)。

HTTP 默认 `direct` 模式在专用线程中创建、调用、切换和关闭 Inference。显式选择 `managed` 后，`ManagedRuntime` 在事件循环中管理唤醒与休眠，`ProcessInference` 经有界 IPC 调用独立进程内的同一个 Inference。`process_tree.py` 回收自有进程树，Windows 使用 Job Object。两种模式的适用场景见[后台运行](background-runtime.md)，所有权与取消要求见[推理契约](specs/inference-contract.md)。

## 产品与研究

`tests/` 验证产品行为；`research/` 保存实验工具、历史报告和原始证据，由 `research/run_tests.py` 单独验证。研究目录不进入 wheel 或源码包。

公开过的 `Engine`、`Model`、`Audio`、`sakuratts.converter`、`sakuratts.nvidia` 与 CLI 保留调用方式。`api.py` 仍是现有 API V2 启动入口，旧版协议的补齐情况见[兼容矩阵](specs/compatibility-matrix.md)。内部模块的迁移对应关系见[开发指南](development.md)。

上游职责映射与目录取舍见 [ADR 0007](adr/0007-upstream-layout.md)；模型、后端、语言与发行组合的边界见 [ADR 0004](adr/0004-composable-components.md)。V2ProPlus 适配涉及的重复实现、平台耦合和同类项目对照保存在[2026-10-04 架构核查记录](../research/notes/mlx-v2proplus-architecture-20261004.md)。扩展顺序见[路线图](roadmap.md)。
