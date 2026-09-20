# Windows 独立工作进程的声码器分块

日期：2026-09-20。Windows 11 / RTX 5060 8 GB，Sakura V2ProPlus。本轮将[已验证的声码器分块](2026-09-20-windows-vocoder-chunks.md)放入现有 Python 3.9 / ORT 1.19.2 独立声学工作进程，检查完整请求的资源、波形与故障恢复。它仍是开发 harness，公开模型格式和默认配置尚未接入分块。

## 执行方式

`research/tools/windows_chunked_worker.py` 在一个声学进程内加载完整 latent 图和局部 vocoder 图。完整 HALF latent 留在该进程的主存中，每块按实际 11 帧上下文取数；两张图均启用 arena 收缩。进程间仍只传既有的 FP32 / int64 输入和重建完成的 FP32 波形，HALF latent 不经过 IPC。主进程继续运行自有 GPT，日文前端保留自身工作进程。

`research/tools/windows_chunked_process.py` 复用既有输入与参考身份检查，只替换实验中的声学加载器。加载时核对源模型、拆图、RF、执行设置、运行库路径和源码哈希，并验证返回的 PID 对应仍在运行的自有子进程。请求失败会关闭并移除该工作进程，下一次请求重新加载；输出必须具有完整长度、正确类型和有限数值，不能返回已生成的部分块。

GPU latent 传递已单独[实现并测量](2026-09-20-windows-vocoder-device.md)，没有显示收益，因此此处沿用主机传递。源模型权重、GPT 采样、停止条件和完整文本均保持各自对照配置，波形只在重建完成后归一化、追加静音和转换 PCM。

## 完整请求的性能

原整图和分块候选分别启动新进程。计时轮为首次请求加四类文本各 3 次热请求，不启用持续资源采样；资源轮单独运行首次请求加四类各 1 次。表中时间均为热长句中位数，覆盖原始文本提交至完整 PCM 返回，含前端、GPT、声学工作进程通信及波形重建，不含 WAV 写盘。

| 配置 | 完整音频 s | 原整图 ms | 256 帧分块 ms | 原整图 / 分块全卡峰值减初值 MiB |
| --- | ---: | ---: | ---: | ---: |
| GPT FP16 baseline attention，自然生成 | 25.46 | 2391.52 | 2422.83 | 1153 / 800 |
| GPT FP32 split-KV 256，官方随机输入回放 | 27.30 | 1590.61 | 1567.07 | 1578 / 1126 |

两组均使用 polyphase FP16 声学权重。自然生成长句约慢 1.3%，固定回放长句约快 1.5%，分块的主要收益仍是显存。短句、多句、标点没有一致提速，不能将长句改善推广到所有请求，也不能把两种不同工作量拼成一个速度 / 显存组合。

自然生成资源轮，原整图全卡初值 / 峰值为 2129 / 3282 MiB，分块为 2124 / 2924 MiB；固定回放分别为 2241 / 3819 MiB 和 2320 / 3446 MiB。以上是 100 ms 的全卡采样，包含桌面与其他进程，且初始桌面用量有变化；峰值减初值不是进程独占显存，也可能漏掉短峰。800 MiB 只描述本轮输入和配置，不能据此宣称普遍达到 Lite 的 0.8 GB。

## 对照与故障验证

不分块的 `--chunk-frames 0` 单独作为拆图控制，和原整图使用相同的独立工作进程。分块的数值检查沿用之前预先规定的波形、频谱、幅度与接缝门槛，并继续报告 `atol=1e-4, rtol=1e-5` 的原严格检查。固定回放中 GPT 与请求契约的通过，不会覆盖 FP16 声学相对官方 FP32 波形的严格失败。

本轮 9 个完整请求实验共保存 85 个 WAV，已逐个读取验证。拆图整段的 13 个请求与原整图 PCM 逐位一致；两组分块各 13 对完整 PCM 和去固定尾静音的语音段均通过工程门槛，各有 9 对未通过原严格容差，保留失败。自然生成与固定回放分别核对 21 和 24 个接缝，全部通过工程检查。20 对计时 / 资源轮 PCM 逐位一致。

与此前共享进程的原图和分块记录另做 52 对比较，其中 26 对为分块候选，PCM 和请求契约全部一致。共享路径使用 Python 3.11，独立声学进程使用 Python 3.9，两者 ORT 均为 1.19.2；该对照验证本轮输出一致，不能将两组环境的耗时差异归因于单一 IPC 成本。固定官方回放的 36 个请求全部保留声学 PCM 严格失败，非 PCM 契约均通过，原始总状态继续为 `replay_validation_failed`。

`research/tools/windows_chunked_lifecycle.py` 使用实际长文本，核对每片波形长度、latent 帧数与分块数，并要求请求至少包含一次多块执行。它覆盖以下行为：

- 在两个请求之间终止本实验拥有的空闲声学进程；下一请求报错且不返回 PCM，清除进程、请求状态与 busy 标记，再次请求启动新进程。
- 连续卸载两次，确认旧声学 PID 退出，再重新加载 GPT 和声学进程。
- staged 模式分别在 `after_prefill`、`before_acoustic`、`after_acoustic` 取消，然后用相同输入重试。最后一项确认已经执行多个声码器块，但取消时尚未向调用者发布 PCM。

这类取消仍发生在既有计算完成边界，测试不涉及执行中强杀、逐块取消延迟或流式首包。生命周期检查、相对同一候选的 PCM 严格检查及逐位一致性分开报告；任何一项都不表示人工听音或 ASR 已通过。

两种 GPT 配置分别跑过一轮上述生命周期，各有 6 个成功长请求、1 个预期失败请求和 3 次取消。12 份保存的 PCM 均独立重算哈希，与各自基线逐位一致，原严格容差也通过。FP16 轮为 1258 个 latent 帧、5 块、25.46 秒；FP32 split-KV 轮为 1280 帧、5 块、25.90 秒。两轮声学完成后的取消均确认已执行 5 块。缓存检查覆盖 KV 与 CUDA Graph 引用释放，不能据此宣称整个进程的 GPU 显存归零。两轮都使用自然生成的 seed 1234；FP32 split-KV 生命周期轮不使用性能表中的官方随机输入回放。

## 运行与证据

完整计时示例：

```powershell
.venv-windows-runtime/Scripts/python.exe -B research/tools/windows_chunked_process.py `
  --config outputs/windows-acoustic/runtime-fp16-polyphase-deterministic.json `
  --split-package outputs/windows-vocoder-chunks/fp16-split-v3 `
  --rf-spec outputs/windows-vocoder-chunks/rf-spec.json --chunk-frames 256 `
  --gpt-precision fp16 --allow-experimental-acoustic-fp16 --acoustic-arena-shrink `
  --repeats 3 --skip-reference-switch --skip-random --skip-idle-unload `
  --no-memory-sampler --output outputs/windows-vocoder-worker/new-timing
```

输出目录必须是新目录。资源轮去掉 `--no-memory-sampler`，不能使用它的耗时作为性能结论。工作进程默认使用 `data/windows-ort-runtime/python.exe` 和同目录下的 CUDA DLL；可用 `--acoustic-python` / `--cuda-dir` 显式指定，加载时会核对实际路径。

原始结果保存在 `outputs/windows-vocoder-worker`；[独立汇总](data/2026-09-20-windows-vocoder-worker.json)包含模型与参考身份、运行库版本、实际输出、严格失败、接缝、原始显存采样和工作进程生命周期。早期计时后补了父进程的 PID / 存活校验，随后补了正常关闭时非零退出码的错误报告；这些修改不改变声学计算。每次运行的源码哈希单独保留，各轮未在执行中改动源码。

最终本地单元回归共 214 项，212 项通过，2 项因 Windows 符号链接权限跳过。新增工作进程测试覆盖 IPC 完整返回、ready 身份、第二块失败、响应截断、关闭超时及清理错误；生命周期测试使用真实引擎状态机和模拟工作进程，另由上述实机长请求验证 GPU 路径。测试不代替样音质量验收。

下一步是将分块产物的验收信息和运行配置接入公开入口，并补充内容、听感和更广的长文检查。速度优化继续针对 GPT 的剩余设备计算；本轮只看到约百分之一的长句耗时变化，不能把分块视为主要的解码加速手段。

本轮另查了两条可用于后续实现的论文思路。[FlashDecoding++ §5](https://arxiv.org/html/2311.01282v4)按矩阵形状选择单行 GEMV 实现，提示 batch=1 时 CUDA Core 实现可能优于 Tensor Core GEMM；这篇是 MLSys 2024 论文，实验模型宽度远大于本项目 512，不能照搬性能比例。[FlashInfer §3.2 与附录 D.2](https://arxiv.org/html/2501.01005v2)讨论单 query 的 CUDA Core 注意力、访存组织，以及短 KV 不经部分结果归并而直接输出；其 v2 为 2025 年，主要验证 A100 / H100，也没有证明 RTX 5060 的收益。

准备评估的范围是本模型五个固定 Decode 线性层形状，以及 `head_dim=32` 的 FP16 注意力内核。保持 FP32 累积、稳定 softmax、完整 KV 和既有固定 CUDA Graph，不引入逐 token 的主机调度。当前较早的 profiler 含主机入队间隙，不能给出最新组合中 GEMV / attention 的独立耗时占比；下一轮先补同配置测量，再用固定历史 logits、采样 token 和完整请求检查筛选实现。这里只记录有来源的候选，尚无新内核性能结论。
