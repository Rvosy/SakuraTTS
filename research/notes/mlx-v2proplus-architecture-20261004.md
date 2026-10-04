# V2ProPlus macOS 适配与架构核查

记录日期：2026-10-04。本次核查以 SakuraTTS `da0d6916a34f1cf113307fe8992b7c6184bd804a` 为修改前的源码起点，并在线读取下列固定提交。本文记录源码事实与结构取舍；推理、数值、音质及发行验证由各自的运行记录提供，当前支持范围见[兼容矩阵](../../docs/specs/compatibility-matrix.md)。现行模块职责见[架构说明](../../docs/architecture.md)。

## 结论

项目已经有公共 Engine、模型描述、共享推理生命周期与逐片合成流程。各设备复用文本处理、采样、请求编排和资源回收，独立实现计算与设备资源管理。这套结构符合多后端推理项目的常见组织方式。

仍有几处代码保留了早期平台实验的假设：两条声学转换路径重复加载同一种模型，MLX 多处只接受 V2Pro，跨平台准备工具仍以 Windows 命名，打包脚本又根据操作系统指定模型家族。这些问题让同一项模型支持需要在多个位置分别补齐，也容易造成能力描述与实际代码不一致。此次适配应收敛这些具体边界，继续沿用已有目录和共享推理流程。

## 修改前的具体问题

下表描述上述起点的代码，不代表修改后的限制。

| 位置 | 源码事实 | 影响 |
| --- | --- | --- |
| `prepare/convert_sovits_mlx.py:convert` 与 `prepare/export_sovits_onnx.py:load_official` | 分别导入官方模型、识别版本和 LoRA、构造 `SynthesizerTrn`、加载并检查权重；MLX 只接受 V2Pro，ONNX 接受 Pro/Plus，后者还检查 checkpoint header 与配置版本一致性 | 同一模型加载规则已有两份不同实现，新增家族容易遗漏其中一条 |
| `backends/mlx/sovits_package.py:validate_manifest`、`backends/mlx/diagnostics.py:check_packages`、`module/reference_condition.py:BoundAcousticReference.from_reference` | 各自要求 V2Pro；公共 `TTS_infer_pack/synthesis.py:_validate_model` 已按参考与声学家族是否一致处理 Pro/Plus | 后端限制、参考匹配和诊断信息混用了固定家族名 |
| `prepare/prepare_windows_resources.py` | 由统一转换器和请求参考管理器调用，实际准备日文前端与参考条件 | 名称不能准确说明职责，容易被误当作 Windows 专属路径 |
| `scripts/build_portable.py:assemble` 中的示例配置生成 | macOS 示例写 V2Pro，其他平台写 V2ProPlus | 模型家族被当成了操作系统属性 |

`TTS_infer_pack/runtime.py:InferenceRuntime` 已经共享模型生命周期、逐片生成和失败清理；`backends.create_runtime` 按显式后端导入实现。此次问题不需要复制这些流程或再次整体迁移核心包。

## 官方模型与同类项目

### GPT-SoVITS

核查提交：[`48b1a0169a28582a8984402f82cf438d3bfa6aca`](https://github.com/RVC-Boss/GPT-SoVITS/tree/48b1a0169a28582a8984402f82cf438d3bfa6aca)。

[`module/models.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/GPT_SoVITS/module/models.py#L623) 用 `v2pro_set` 同时表示 V2Pro 与 V2ProPlus，两者使用同一个 `SynthesizerTrn`。其 `is_v2pro` 分支创建相同的 `sv_emb`、`ge_to512` 和 `prelu` 条件模块。模型配置与执行设备分别由 [`TTS_Config`](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/GPT_SoVITS/TTS_infer_pack/TTS.py#L255) 表达。

比较官方 [s2v2Pro.json](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/GPT_SoVITS/configs/s2v2Pro.json) 与 [s2v2ProPlus.json](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/GPT_SoVITS/configs/s2v2ProPlus.json)，`model` 对象有两项差异：

| 配置 | V2Pro | V2ProPlus |
| --- | --- | --- |
| `upsample_initial_channel` | 512 | 768 |
| `upsample_kernel_sizes` | `[16, 16, 8, 2, 2]` | `[20, 16, 8, 2, 2]` |

这两份配置都使用 32 kHz 采样率。具体 checkpoint 仍应按其实际配置构造模型并检查权重，不能仅凭这份默认配置对照接受任意变体。

上述源码支持让 Pro/Plus 共用模型构造与配置驱动的执行路径。它不证明现有 MLX 算子的数值、内存或音质已经满足 Plus 模型要求，这些需要实际运行。

### MLX Audio

核查提交：[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9)。

[`tts/utils.py`](https://github.com/Blaizzy/mlx-audio/blob/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/utils.py#L103) 提供统一的模型加载入口，通过模型类型选择具体实现；[`tts/generate.py`](https://github.com/Blaizzy/mlx-audio/blob/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/generate.py#L151) 处理公共输入和音频输出，再调用 `model.generate`。各模型保留自己的结构、配置和转换代码。

这与 SakuraTTS 保留统一服务和推理入口、由模型与后端处理具体差异的方向一致。MLX Audio 本身以 MLX 为计算框架，这次对照只能说明模型组织方式，不能用于判断跨平台依赖是否应统一。

### whisper.cpp

核查提交：[`60c0be6ac8fa71b1a2ae2dd938a31a34a508e774`](https://github.com/ggml-org/whisper.cpp/tree/60c0be6ac8fa71b1a2ae2dd938a31a34a508e774)。

[`src/whisper.cpp`](https://github.com/ggml-org/whisper.cpp/blob/60c0be6ac8fa71b1a2ae2dd938a31a34a508e774/src/whisper.cpp#L1302) 通过 ggml 后端接口选择设备，同时保留各设备的具体实现。[Core ML 路径](https://github.com/ggml-org/whisper.cpp/blob/60c0be6ac8fa71b1a2ae2dd938a31a34a508e774/README.md#core-ml-support) 还需要独立转换 encoder，并与其余推理部分组合。

统一入口、共享流程与专门的设备实现可以并存。该项目使用自己的张量与后端基础设施，不能据此要求 SakuraTTS 立即增加同等规模的抽象层；本项目已有的静态后端工厂足以处理当前实现。

## 此次结构调整

调整沿用 [ADR 0004](../../docs/adr/0004-composable-components.md) 的模型、后端、语言与发行边界，以及 [ADR 0007](../../docs/adr/0007-upstream-layout.md) 的目录职责：

1. 官方声学 checkpoint 加载归入 `prepare/sovits_checkpoint.py`，由 MLX 与 ONNX 两个导出器共用。模型家族识别、官方模型构造和权重检查只有一份实现；原生权重包与 ONNX 计算图仍由各自的导出器生成。
2. 前端与参考准备工具使用 `prepare_resources.py` 名称。调用方、转换缓存和发行脚本沿用同一条资源准备路径，由安装配置提供解释器。
3. MLX 按声学产物配置执行 Pro/Plus，诊断报告读取实际模型家族，参考条件检查家族匹配。整合包示例只配置设备与权重路径，不再由操作系统指定模型家族。

CUDA 的工作进程与 ABI 隔离、DirectML 的图形状和 KV 分配、MLX 的 CPU/Metal 数值策略、各平台的二进制依赖与启动脚本继续保留。这些差异有实际设备和运行时约束，改目录名称无法消除。其余公开入口、请求编排与资源生命周期继续复用现有实现。

本记录不包含推理测试、试听或重新打包的验收结果，也不据源码相似性推断跨后端数值等价。
