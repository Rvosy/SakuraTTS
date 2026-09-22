# 架构与目录

公共入口是 `Engine`、`Model` 和 `Audio`。CLI 与 HTTP 共用 Engine，后端选择由 `backends.create_runtime` 负责。可执行组合见[兼容矩阵](specs/compatibility-matrix.md)。

| 位置 | 职责 |
| --- | --- |
| [engine.py](../src/sakuratts/engine.py) | Engine 生命周期、请求互斥、Audio 与 HTTP 共用的 Inference |
| [model.py](../src/sakuratts/model.py) | 模型描述、资源路径和参考名称关系 |
| [converter.py](../src/sakuratts/converter.py)、[reference.py](../src/sakuratts/reference.py) | 模型转换、请求参考解析与音频条件缓存 |
| [frontend/](../src/sakuratts/frontend) | 语言模式、分句、规范化、音素和文本特征 |
| [backends/](../src/sakuratts/backends) | 静态后端选择、CUDA 推理、ONNX 声学和 MLX 研究实现 |
| [_internal/](../src/sakuratts/_internal) | 生成、采样、权重、私有 IPC 与运行控制 |
| [_internal/conversion/](../src/sakuratts/_internal/conversion) | 独立准备环境中的导出与参考编码 |
| [server.py](../src/sakuratts/server.py) | HTTP 请求边界、响应与服务生命周期 |
| [packaging/recipes/](../packaging/recipes)、[build_portable.py](../scripts/build_portable.py) | 发行组合与本地运行组件装配 |

## 配置与校验所有权

`Model.load` 检查模型描述和资源目录，读取元数据时不加载计算库。模型可声明语言及建议后端；执行能力由后端选择与前端装配判断。字段、路径和覆盖顺序集中在[模型格式](model-format.md)。

`backends.require_backend` 检查已实现的后端，`create_runtime` 导入对应实现。CUDA 后端解释自己的实验选项并校验实际使用的资源。公共 Engine 负责请求互斥、输出与关闭。`sakuratts capabilities` 查询源码中实现的能力，依赖与设备检查使用 `sakuratts doctor`。

`frontend.profiles` 定义已实现的语言模式，`TextFrontend` 负责共享请求准备，语言处理器生成音素和特征，`FrontendRuntime` 装配资源并管理前端进程。日文沿用原版零 BERT 特征；中文研究代码尚未接入公共前端。

`SpeechRequest` 负责 HTTP 字段、默认值及请求能力校验。通过检查后，HTTP 调用同一个 `Inference` 和 Engine。参考缓存复用音频条件，转写与语言特征按请求生成。

## 进程与资源

声学和经典日文前端各自使用私有工作进程。`acoustic_python`、`frontend_python` 可以共享解释器；旧配置省略后者时沿用前者。工作进程通过 `_internal/worker.py` 绑定 SakuraTTS 包，保持主环境与私有 Python 的 ABI 隔离。

完整整合包另带 CPU 准备环境，按需转换权重或编码参考，完成后退出。执行适配器从 `runtime/portable.json` 绑定安装位置；模型文件保存资源描述。发行组合与依赖来源见[整合包](portable-bundle.md)。

HTTP 默认 `direct` 模式在专用线程中创建、调用、切换和关闭 Inference。显式选择 `managed` 后，`ManagedRuntime` 在事件循环中管理唤醒与休眠，`ProcessInference` 经有界 IPC 调用独立进程内的同一个 Inference。`process_tree.py` 回收自有进程树，Windows 使用 Job Object。两种模式的适用场景见[后台运行](background-runtime.md)，所有权与取消要求见[推理契约](specs/inference-contract.md)。

## 产品与研究

`tests/` 验证产品行为；`research/` 保存实验工具、历史报告和原始证据，由 `research/run_tests.py` 单独验证。研究目录不进入 wheel 或源码包。

公开过的 `api.py`、`sakuratts.nvidia` 与 `synthesize` CLI 保留已有调用兼容，内部实现仍共用包内代码。目录迁移和开发命令见[开发指南](development.md)。

模型、后端、语言与发行组合的拆分依据见 [ADR 0004](adr/0004-composable-components.md)。固定版本上游及本地整合包的源码核查保存在[2026-09-22 核查记录](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/architecture-audit-20260922.md)；扩展顺序见[路线图](roadmap.md)。
