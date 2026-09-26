# ADR 0005：CPU GPT 与可选 DirectML 声学执行

状态：首轮架构已采用，公开执行配置已由 [ADR 0006](0006-device-precision-and-directml-kv.md) 替代。日期：2026-09-27。扩展 [ADR 0004](0004-composable-components.md) 中的执行后端边界。

本文保留首轮 CPU GPT 加可选 GPU 声学的设计原因。当前 CPU 默认且仅提供 INT8 GPT 与 FP32 声学，DirectML 默认且仅提供 GPU FP16 GPT 与全图 FP16 声学；默认线程、容量和使用方式见 [ADR 0006](0006-device-precision-and-directml-kv.md) 与[设备指南](../cpu-amd.md)。下述首轮精度与设备组合不再作为公开配置。

## 原因

原公共推理路径依赖 NVIDIA CUDA。内部 ORT 声学实现已有 CPU 执行能力，但缺少 CPU GPT 和完整的后端装配。Windows 上的 AMD 核显需要可选择的 GPU 计算路径，同时保留全 CPU 运行方式，便于比较速度和资源占用。

现有 GPT 包保存模型原始 FP32 张量，声学包已经导出为 ONNX。复用这两类资源可以保持模型与参考条件身份一致，也能避免把 PyTorch 和训练工程加入普通推理环境。AMD 首轮适配针对 Radeon 780M；硬件选择不改变语言前端、采样与取消契约。

## 首轮决定

在显式后端表中加入 `cpu` 和 `directml`。首轮 CPU GPT 使用 NumPy BLAS，随后加入 ONNX Runtime；两种模式分别选择 ORT CPU 或 DirectML 声学 Session。当时 DirectML 模式的 GPT、文本前端和采样均在 CPU 上执行，合成报告分别记录语义和声学设备。

CPU GPT 保留一份 FP32 权重，KV 按容量分配并原位写入。Prefill 按 query 分块，控制临时得分矩阵的大小；每块保留完整允许历史。Decode 使用矩阵与向量运算和可复用工作区。BLAS 线程限制仅在计算期间生效，调用结束恢复宿主设置。实现与默认值由 [CPUGPT](../../src/sakuratts/backends/cpu/gpt.py) 维护，独立小模型的全历史计算用于验证缓存结果。

后续加入的 [ONNXCPUGPT](../../src/sakuratts/backends/cpu/onnx_gpt.py) 让 Prefill 和 Decode 共用一个 ORT Session，避免重复驻留 Transformer 权重。独立导出的附加资源绑定原 GPT 包身份，运行时不解压原 Transformer 权重。这轮路径保留 CPU FP32 计算和固定容量 KV，但不支持 query 分块。当前精度选择与资源校验边界见 [ADR 0006](0006-device-precision-and-directml-kv.md)。

共享的模型装配、请求编排、随机数消耗、取消、错误回收和报告移入 [InferenceRuntime](../../src/sakuratts/_internal/runtime.py)。后端只负责执行选项和模型加载。原 CUDA 的默认精度、资源策略、独立声学 worker 和公共入口继续保留。

首轮 CPU / DirectML 使用 FP32 GPT 与 FP32 整图声学包，随后验证了 DirectML 声码器混合 FP16。该候选单独转换，通过绑定文件身份、执行参数和 GPU 算子证据的工程筛查；记录保留硬件与运行库版本，不代表其他设备的质量验收。CUDA 的 FP16 筛查不能直接用于 DirectML，chunked 声学执行限 CUDA。这些对照用于后续独立设备路径的选择。

DirectML 使用显式 DXGI 适配器索引，关闭 ORT memory pattern，采用串行 Session 执行；现有单活动请求所有权与此约束一致。缺少 provider 或初始化失败时明确报错，禁用执行失败后的整体 provider 自动回退。图内的 CPU 算子分配仍由 ORT 处理，实际 GPU 执行需通过 profile 或设备计量验证。接口限制见 [DirectML execution provider 文档](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)。

CPU / DirectML 的 ORT 与主解释器使用同一 ABI，声学计算在主进程内执行，不启动 `acoustic_python` worker。日文前端继续遵循 `frontend_python` 及旧配置的兼容规则。转换和参考准备仍由独立环境负责，普通推理不导入 PyTorch。

发行依赖按 ORT 版本分开：`cpu` 使用 CPU ORT，`directml` 使用 DirectML ORT，后者也能运行 `cpu` 后端。`japanese-text` 提供不绑定 ORT 发行包的日文依赖；原 `japanese` extra 保持原来的 CPU ORT 组合。同一解释器不能同时安装两种共用模块目录的 ORT 发行包。依赖定义见 [pyproject.toml](../../pyproject.toml)。

后端通过模型建议、Python 参数、CLI 或服务配置显式选择。服务的 `sakuratts.runtime_options` 提供执行选项，显式 `experimental` 参数覆盖同名配置。`managed` 继续控制完整推理进程的生命周期；后端选择、请求状态释放和空闲休眠分别配置。具体用法集中在 [CPU 与 AMD 推理指南](../cpu-amd.md)。

## 代价与验收

NumPy GPT 每步在 CPU 上读取权重，可能成为 AMD 模式的主要耗时。DirectML 声学执行也可能包含 CPU 算子及设备传输；核显与 CPU 共享内存带宽，不能仅凭 GPU 启用就推断整条请求更快。线程数、请求状态释放和内存 arena 的选择须结合完整请求与空闲占用测量。

整图声学执行暂时保留较大的活动工作区；降低驻留、使用 `managed` 或切换 `staged` 会改变加载等待，需分别报告。主进程内的声学执行减少一个工作进程，也使 ORT 发行包成为主环境的明确依赖。

数值测试、真实模型执行、性能、人工听音和干净环境安装分别验收。CPU / DirectML 的实现不继承 CUDA 既有硬件与音质结论；当前人工音色、自然度和长句完整性仍未完成验收。验证结果由[兼容矩阵](../specs/compatibility-matrix.md)索引，测量遵循[基准协议](../specs/benchmark-protocol.md)。
