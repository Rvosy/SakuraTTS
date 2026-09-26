# CPU / Radeon 780M：完整请求与执行档位

本轮把单 Session ORT GPT 与 DirectML 声码器混合 FP16 接入公共 Engine，并比较完整请求。FP32 优化后的短、长句 PCM 与 NumPy 基线逐样本一致；混合 FP16 改变波形，但降低了本轮观测到的工作集峰值。FP16 没有在所有输入上缩短完整请求耗时。

原始逐请求数据位于 `results/cpu-amd-performance/`，各进程的计时、CPU 时间、工作集、私有内存、音频长度与文件哈希汇总在[紧凑证据](../experiments/data/2026-09-27-cpu-amd-performance.json)。[GPT 实验](cpu-gpt-ort-20260927.md)与[声学精度实验](directml-mixed-vocoder-780m-20260927.md)分别记录计算变化、失败候选和数值检查，本页只讨论整链结果。

## 条件

Ryzen 7 7840HS、Radeon 780M、约 16 GB RAM、Windows 11，DirectML ORT 1.24.4。N.A.V.I V2ProPlus 使用原始 GPT / SoVITS 检查点、中性参考和保存的相同转写。原权重、参考音频与 Genie 安装没有改写。原始包身份沿用[首轮记录](cpu-directml-780m-20260927.md)，本轮导出资源哈希另存于紧凑证据。

所有原生路径使用 seed=1234、top_k=15、temperature=1、repetition_penalty=1.35、`cut0` 和相同噪声参数。每个配置启动独立进程，先执行一次冷短句，再对每种句长预热一次、测两次热请求。表中是完整 PCM 返回的耗时中位数，包含 GPT、声学和片段处理，不含音频写盘。采样线程在计时期间每 100 ms 读取进程树资源。

配置按顺序执行，未随机轮换，也未控制温度、功耗和其他桌面负载。两次热请求不能估计稳定的延迟分布；小幅差异只保留实测值，不视为确定的普遍提升。

## 2 线程整链对照

这一组 GPT 与声学均设为 2 线程。DirectML 的 GPU 卷积不受 CPU 线程数直接控制。

| 路径 | 热短句 | 热长句 | 进程树峰值工作集 | 进程树峰值私有内存 |
|---|---:|---:|---:|---:|
| Genie CPU | 1.492 s | 22.011 s | 5152.6 MiB | 6577.8 MiB |
| NumPy GPT + CPU FP32 声学 | 3.782 s | 23.603 s | 1374.7 MiB | 1655.1 MiB |
| ORT GPT + CPU FP32 声学 | 3.369 s | 23.006 s | 1346.5 MiB | 1676.3 MiB |
| NumPy GPT + DirectML FP32 声学 | 2.476 s | 14.500 s | 1736.2 MiB | 2308.2 MiB |
| ORT GPT + DirectML FP32 声学 | 1.740 s | 12.118 s | 1786.2 MiB | 2427.9 MiB |
| ORT GPT + DirectML 混合 FP16 声学 | 1.828 s | 12.084 s | 1419.7 MiB | 2055.6 MiB |

原生各配置短、长音频分别为 3.42、20.70 秒，所有生成序列和停止位置相同。每个后端的 NumPy / ORT FP32 对照均为逐样本相同的 PCM。声学模型不变时，DirectML 完整请求短、长句分别减少约 29.7%、16.4%；全 CPU 路径分别减少约 10.9%、2.5%，其声学部分仍占主要时间。

混合 FP16 相对 ORT GPT + FP32 声学的峰值工作集减少约 20.5%，但短句慢约 5.1%，长句只有约 0.3% 的差异。不能把固定声学图的提速直接当成整链提速。混合精度短、长句相对对应 FP32 PCM 的 SNR 分别为 43.54、41.28 dB，最大差异为 221、333 个 PCM16 量化单位；人工听音和 ASR 未验收。

Genie 使用用户已有 CPU 安装与预转换模型。其两次热短句音频为 2.28 / 1.72 秒，热长句为 19.16 / 18.00 秒；前端、采样与停止规则不同，不能按表格计算固定工作量的加速倍数。

工作集与私有内存不是同一口径，不能相加。进程树求和可能重复统计共享页，也可能与核显共享分配重叠；该表不含独立的 WDDM GPU 计量。峰值覆盖进程启动、加载、所有请求和卸载，不只是热请求。

## 线程选择

[GPT 线程实验](cpu-gpt-ort-20260927.md)按正反顺序比较 1、2、4、8 线程，所有 16 条固定历史输出 logits 逐位相同。4 线程兼顾短句和长句；8 线程的长句再快一些，但短句更慢。`gpt_threads` 因此独立于声学 `threads`，可以在不改变 DirectML FP16 筛查设置的情况下调整 GPT。

全局缺省仍由 [CPUEngine](../../src/sakuratts/backends/cpu/engine.py) 定义；[4 线程示例](../experiments/configs/cpu-onnx-4threads.json)是这台机器的显式选择，不是所有低端 CPU 的最佳值。提高线程数可能增加处理器占用，需连同实际完整请求时间比较。

随后以相同输入、预热和两次热请求复测 GPT 4 线程，声学仍为 2 线程：

| 路径 | 热短句 | 热长句 | 进程树峰值工作集 | 进程树峰值私有内存 |
|---|---:|---:|---:|---:|
| ORT GPT + CPU FP32 声学 | 3.492 s | 20.691 s | 1348.3 MiB | 1677.1 MiB |
| ORT GPT + DirectML FP32 声学 | 1.771 s | 10.787 s | 1830.0 MiB | 2469.9 MiB |
| ORT GPT + DirectML 混合 FP16 声学 | 1.518 s | 10.201 s | 1452.4 MiB | 2122.3 MiB |

三种配置与各自 2 线程版本的 PCM 均逐样本一致。与本轮初始 NumPy + DirectML FP32 路径相比，4 线程 ORT GPT + 混合 FP16 的短、长句耗时分别减少约 38.7%、29.6%。同为 4 线程时，混合 FP16 的工作集比 FP32 少约 20.6%。这些仍是依次执行的小样本测量；不同轮次的 FP16 短句结果方向不同，不能用其中最快一次推断所有文本的收益。

全 CPU 的长句有所改善，短句却比 2 线程 ORT 版本稍慢。因此保留线程选项，让短句延迟、长句吞吐和资源占用分别选择。进程 CPU 时间已保存在紧凑证据中：模型优化并不保证 CPU 占用按墙钟耗时同比下降，也未测量后台游戏或其他应用的响应影响。

## 服务与释放

实际 `managed` HTTP 服务分别加载 CPU ONNX、DirectML ONNX FP32、DirectML ONNX 混合 FP16 三份配置，每份执行两轮唤醒、合成与休眠。6 次请求均返回完整的 3.42 秒短句 WAV；每轮休眠后，所有观测到的自有推理进程均退出。记录在 `results/cpu-amd-performance-managed/summary.json`，内存字段按整个进程树保存，包含 Windows Python launcher 与实际模型进程。

另在混合 FP16、GPT 4 线程配置上采集 WDDM 进程 GPU 内存。168 个采样覆盖完整进程树，GPU 实例均匹配 Radeon 780M 的 LUID，没有无效计数、重复身份或未解析实例。长句的 Dedicated / Shared / Committed 峰值分别为 118,116,352 / 482,414,592 / 600,530,944 字节；请求后约 3 秒仍保留这些分配。休眠后观测到的自有进程全部退出，GPU 实例随之消失；实例缺失不记成实测零字节。原始数据保存在 `results/cpu-amd-performance-wddm/result.json`，摘要与哈希已加入紧凑证据。这轮计量包含采样开销，不替代上表延迟，也不能与进程工作集相加。

## 配置与验证边界

`backend` 选择 CPU、CUDA、DirectML 或实验性 MLX，`profile` 选择对应的执行预设，`runtime_options` 可覆盖线程和驻留策略。选择预设不会转换权重；声学 FP16 必须匹配独立筛查包，CPU 不借用 DirectML 的筛查结果。配置方法集中在[推理档位](../../docs/inference-profiles.md)与[CPU / AMD 指南](../../docs/cpu-amd.md)。

Apple 适配器只完成公共 Engine 的预制 V2Pro 接入与 Windows 上的软件边界测试，未在本轮验证 Metal 性能或音质。NVIDIA 路径保留既有实现和精度选项，本轮没有 NVIDIA 硬件复测。INT8 GPT 仍是研究候选，已观察到生成序列分歧，没有进入公开档位。

最终产品回归共 440 项，439 项通过、1 项跳过，日志为 `results/cpu-amd-performance-full-tests-final.txt`；相关实验工具另有 12 项测试通过。CPU 与 DirectML 的实际依赖、模型资源诊断通过，Windows 上的 MLX 诊断按预期报告平台不支持。12 份原始检查点、参考和 Genie 模型资源的 SHA-256 复核保持不变。上述自动检查不包含人工听音、ASR 或其他机器的部署验收。
