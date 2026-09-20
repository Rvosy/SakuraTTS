# Windows 正式入口的独立声码器分块包

日期：2026-09-20。Windows 11 / RTX 5060 8 GB，Sakura V2ProPlus。已验证的[独立工作进程分块](2026-09-20-windows-vocoder-worker.md)现在可以由公开 `synthesize` / `NVIDIAEngine` 入口加载，不再依赖实验 harness 替换声学加载器。它仍是需要显式选择的精度与分块候选，音色、自然度和内容质量尚未验收。

## 包与执行边界

`tools/package_sovits_chunks.py` 从现有声学证据生成 `sakuratts-sovits-chunked-v1`。打包过程中重新读取 21 组实际输入及原图、拆图整段、256 帧分块的 63 份输出 NPZ，重算完整波形、单次 PCM 转换、接缝及工程门槛。输入覆盖四类普通文本、16 种语义长度和一个零噪声控制。每组后两次执行只有保存的哈希；工具会核对它们，但不声称重新读取不存在的重复数组。原严格容差 `atol=1e-4, rtol=1e-5` 及失败记录保留。

导出包包含 7 个文件：latent / vocoder 各一张 ONNX 图和一份权重、纯 JSON 感受野说明、分块筛查报告、manifest。两部分权重没有重复 initializer。报告绑定完整包身份，准入时核对图、权重、接口、RF、精度、执行设置和已验证块长；运行时只读取包内文件。原整图与实验路径作为来源字符串保留，不是运行依赖。当前产物的身份为 `cbfc9c62626f9a98924f61b5ec0333052527ccbd61d0ca8b0a31c63263fdaac6`。

`ORTChunkedSoVITS` 和旧 harness 共用分块执行实现。完整 encoder / flow 先得到 HALF latent，按真实 11 帧上下文执行局部 vocoder，再按整数样本位置裁剪并重建完整 FP32 波形。HALF latent 保留在独立声学进程的主存，不经过 IPC；所有块成功后才向引擎返回波形。归一化、追加静音和 PCM 转换仍各执行一次。没有缩短文本、裁剪非因果上下文、填充 latent 或交叉淡化。

公开工作进程沿用既有 `ORTProcessSoVITS` / `ort_worker`。父进程核对实际子进程 PID、解释器、包哈希、精度、块长和 arena 设置；返回波形必须为完整长度的有限 FP32 数组。请求失败清除进程和本次传输状态，下一请求重新加载。关闭释放两张 session；构造失败也清除半构造对象的 session 引用。普通推理不需要 Torch、ONNX 导出器或 `harness` / `scripts` 模块。

## 已完成的公开入口运行

低显存配置已通过普通命令行生成，两种配置均由正式 benchmark 路径执行。计时轮为首次短句加四类文本各 3 次热请求，不启用持续内存采样；资源轮另起进程，各类热请求执行 1 次。时间覆盖原始文本提交至完整 PCM 返回，包含前端、GPT、声学通信和波形重建，不含写 WAV。

| GPT 与输入 | 长句完整 PCM | 热长句中位数 | 热短句中位数 |
| --- | ---: | ---: | ---: |
| FP16、baseline attention、自然生成 | 25.46 s | 2373.28 ms | 180.77 ms / 3.46 s PCM |
| FP32、split-KV 256、官方随机输入回放 | 27.30 s | 1580.12 ms | 125.20 ms / 2.02 s PCM |

两组都使用相位重排 FP16 声学、256 帧分块和 arena 回收。它们的 GPT 精度、随机输入与生成工作量不同，不能横向比较后把更低显存与更低耗时拼成同一配置。上一轮私有 harness 的长句分别为 2422.83 / 1567.07 ms；本轮主要验证公开入口保持相同执行，少量耗时变化不能解释为这次接入带来的加速。

自然生成资源轮的全卡首次样本为 2258 MiB、峰值为 3057 MiB，差值 799 MiB；速度配置为 2167→3291 MiB，差值 1124 MiB。采样包含桌面和其他进程，也可能漏掉短峰值；这是本轮完整请求序列的全卡增量，不能称为精确的进程独占峰值或普遍显存上限。计时轮与资源轮分开，资源采样期间的耗时不用于上述表格。

公开 `--acoustic-chunk-frames 0` 的拆图整段控制及原 FP32 整图官方回放也已执行。原 FP32 的 5 个完整回放均通过；候选的官方回放继续以 `replay_validation_failed` 结束，保留声学 PCM 相对官方 FP32 的严格失败。这些失败不因引入新包格式而被重新归类为通过。

独立 CPU 审计读取了本轮 46 个 WAV 和旧私有路径的 49 个 WAV；41 对公开 / 私有同条件请求的契约和 PCM 全部一致，另外 10 对计时 / 资源请求逐位一致。60 处实际块边界的 RF 计划也已重算。fast 计时 / 资源轮共 18 个请求保留官方 PCM 严格失败，非 PCM 契约全部通过。拆图整段控制、候选对原图的波形与接缝检查、官方回放失败和各次源码身份记录在[汇总数据](data/2026-09-20-windows-vocoder-public.json)。CLI 短句另与旧候选的同输入 PCM 核对一致。

`research/tools/windows_chunked_lifecycle.py --public-package` 直接构造 `NVIDIAEngine`，没有替换声学加载器。两种 GPT 配置分别完成 6 个成功长请求、1 个预期工作进程故障和 3 次取消，覆盖空闲 worker 被终止后的报错 / 重试、连续两次卸载，以及 `after_prefill`、`before_acoustic`、`after_acoustic` 取消后重试。12 份公开 NPY 及对应的 12 份旧私有 NPY 已重新读取并核对，各组恢复输出及跨入口输出逐位一致。声学后取消确认已经执行多个 vocoder 块，但调用者未收到部分 PCM。这里检查既有计算完成边界，不测执行中的取消延迟或逐块流式行为。

本地完整单元回归运行 257 项，255 项通过，2 项因 Windows 符号链接权限跳过。测试覆盖包搬迁后准入、损坏或错配的产物与筛查证据、公开加载和两张 session 清理、IPC 完整返回和失败关闭，以及生命周期状态机。单元测试使用合成小包或模拟运行库；实际 GPU 请求与保存音频另行核对。

实测期间源码未变。提交前只清理了运行时 RF 模块末尾多余的空行及末行换行格式；汇总分别保留实测和最终文件哈希，计算实现没有变化。

## 复现

本机已生成 `outputs/windows-vocoder-worker/package-offline-v1`。从原始证据重新导出时选择新目录：

```powershell
.venv/Scripts/python.exe -B tools/package_sovits_chunks.py `
  --source outputs/windows-acoustic/fp16-candidate-polyphase-deterministic `
  --split-package outputs/windows-vocoder-chunks/fp16-split-v3 `
  --rf-spec outputs/windows-vocoder-chunks/rf-spec.json `
  --evidence-root outputs/windows-vocoder-chunks `
  --ordinary-inputs outputs/windows-acoustic/polyphase-deterministic-screen/inputs.json `
  --boundary-inputs outputs/windows-vocoder-chunks/boundary-inputs/inputs.json `
  --output outputs/windows-vocoder-public/new-package
```

这是离线证据校验与文件打包，不创建 GPU session。将 Windows 配置的 `sovits` 指向新包，保留 GPT、前端、参考及 `acoustic_python`。已有目录会被拒绝覆盖。普通 CLI 用法见 [Windows 使用说明](../../docs/setup-windows-nvidia.md#独立声码器分块包)。

本轮低显存配置的计时命令为：

```powershell
.venv-windows-runtime/Scripts/python.exe -B research/tools/windows_nvidia_benchmark.py `
  --config outputs/windows-vocoder-public/runtime-fp16-chunk256.json `
  --gpt-precision fp16 --allow-experimental-acoustic-fp16 `
  --acoustic-arena-shrink --acoustic-chunk-frames 256 `
  --repeats 3 --skip-reference-switch --skip-random --skip-idle-unload `
  --no-memory-sampler --output outputs/windows-vocoder-public/new-natural-timing
```

速度配置将 GPT 参数改为 `--gpt-precision fp32 --gpt-attention split-kv --gpt-attention-chunk-size 256`，再加 `--replay-captures outputs/windows-precision/captures.json`。资源轮用新目录、`--repeats 1`，去掉 `--no-memory-sampler`。固定官方回放只使用配置中的 CUDA 参考条件。

生命周期示例：

```powershell
.venv-windows-runtime/Scripts/python.exe -B research/tools/windows_chunked_lifecycle.py `
  --public-package --config outputs/windows-vocoder-public/runtime-fp16-chunk256.json `
  --gpt-precision fp16 --gpt-attention baseline `
  --allow-experimental-acoustic-fp16 --acoustic-arena-shrink `
  --output outputs/windows-vocoder-public/new-lifecycle-fp16
```

FP32 split-KV 生命周期使用上述 GPT 速度配置参数，但仍是自然生成 seed 1234，不使用官方随机回放。各组输入和输出身份按各自基线核对。

分块包必须显式选 `0` 或 `256`，启用 arena 回收，FP16 包还须允许实验精度。它不支持中间层诊断捕获；原整图包拒绝分块参数。`doctor --nvidia` 目前仍只覆盖原 FP32 整图范围；本轮候选由实际合成和专用验证确认，没有扩大 doctor 的声明范围。干净机器部署、更多模型、人工听音和 ASR 继续待验，完整音频完成时间也不代表流式首包。
