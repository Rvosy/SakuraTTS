# CPU 与 AMD 独立精度：完整请求和试听

Genie 与当前配置的同机复测、前后两轮基线和对照试听见 [Genie 对比](genie-comparison-20260927.md)。本文保留此前十种配置的测量记录。

CPU 与 AMD 现在分别提供 FP16 配置。CPU 模式的计算会话全部使用 CPU；AMD 的 GPT Transformer 与声学模型使用 DirectML，KV 在 GPU 上保留。此前 CPU GPT 加 GPU 声码器的组合仍可用，命名为 `hybrid-fp16`。当前设备、精度和资源参数见[推理档位](../../docs/inference-profiles.md)及[设备指南](../../docs/cpu-amd.md)。

全图 FP16 指主干的可转换浮点运算、权重和 GPT KV，不包含整数索引，也不要求文本前端、embedding、采样或公共声学 I/O 改成半精度。CPU 和 GPU 的实际内核行为分别验证；不能把 FP16 文件容量当作真实计算精度或峰值内存。

## 条件与证据

Ryzen 7 7840HS、Radeon 780M、Windows 11、约 16 GB RAM、驱动 31.0.14005.11001、ORT DirectML 1.24.4。N.A.V.I V2ProPlus、中性参考、相同短长日文文本；seed=1234、top_k=15、temperature=1、repetition_penalty=1.35、`cut0`。12 份原始检查点、参考和 Genie 资源的 SHA-256 复核未变。

原始数据在 `results/precision-listening/`，包含 job、进程日志、逐请求 WAV、进程树资源采样和逐请求报告。[紧凑证据](../experiments/data/2026-09-27-cpu-amd-precision-listening.json)保存原始结果哈希、模型 sidecar 身份、参数、音频时长、CPU 时间、内存与试听文件身份。

十种配置分别启动新进程，依次运行，共 68 次完整请求。每进程先生成冷短句，再对各句长预热一次、测两次热请求；512 容量只运行短句，测三次热请求。每 100 ms 采样整个自有进程树，采样在计时期间保持开启。时间从原文到完整 PCM 返回，排除 WAV 写盘。配置没有随机轮换，未控制温度、功耗与其他桌面负载，这些小样本不代表稳定延迟分布。

## 完整请求

表中内存为进程树峰值工作集，覆盖加载、请求和卸载。CPU GPT 使用 4 线程；声学线程单列。AMD 行的线程数为 ORT CPU 部分设置，不代表 GPU 并行度。

| 配置 | 声学线程 | KV 容量 | 热短句 s | 热长句 s | 峰值工作集 MiB |
|---|---:|---:|---:|---:|---:|
| CPU ORT FP32 | 2 | 2048 | 2.908 | 21.689 | 1348.5 |
| CPU ORT FP32 | 8 | 2048 | 2.457 | 17.309 | 1350.4 |
| CPU FP16 | 2 | 2048 | 5.152 | 33.338 | 1502.9 |
| CPU INT8 GPT + FP32 声学 | 2 | 2048 | 2.424 | 17.257 | 1119.3 |
| CPU INT8 GPT + FP32 声学 | 4 | 2048 | 1.778 | 12.520 | 1121.6 |
| CPU INT8 GPT + FP32 声学 | 8 | 2048 | 1.611 | 10.574 | 1121.4 |
| AMD FP32 | 2 | 2048 | 2.458 | 16.276 | 3605.7 |
| AMD FP16 | 2 | 2048 | 2.009 | 11.893 | 1980.3 |
| AMD FP16 | 2 | 1280 | 1.762 | 9.714 | 1760.9 |
| AMD FP16，仅短句 | 2 | 512 | 0.972 | 不适用 | 1191.7 |

短句音频均为 3.42 秒。长句的 CPU FP16 为 20.50 秒，CPU INT8 为 20.26 秒，其他配置为 20.70 秒；生成序列、停顿与音色尚待听音，因此耗时减少不能全部视为相同输出下的加速。512 行没有运行长句，内存峰值不能直接当作长句工作集与其他行相减。

同为 8 线程声学时，CPU INT8 比 CPU FP32 的长句完整请求耗时少约 38.9%，峰值工作集少约 17.0%。CPU INT8 的 4 / 8 线程档保留了速度与处理器并发使用的选择；该长句的主推理进程 CPU 时间中位数分别为 17.81 / 17.94 CPU-s，不能据此推断整机功耗或其他应用的响应。

同为 2048 容量时，AMD FP16 比 AMD FP32 的长句耗时少约 26.9%，峰值工作集少约 45.1%。1280 容量继续降低时间与工作集。固定 GPU KV 图会处理并维护整个容量，容量对性能的影响比 CPU 有效前缀缓存更明显。512 是短句选项：本轮长文本加参考的 Prefill 已有 552 token，明确超出该容量。实际容量拒绝后再次生成短句成功，记录在 `results/precision-listening/capacity-check/result.json`；没有截断文本或自动换设备。

前一轮[混合设备配置](cpu-amd-performance-20260927.md)的最优长句为 10.20 秒、峰值工作集约 1452 MiB。独立 AMD 1280 配置略快，但工作集更大；GPU Prefill / Decode 两个 Session 与两组 KV 有额外驻留成本，不能宣称完整 GPU 路径在所有资源指标上都优于混合设备配置。

## 精度与优化边界

本机 NumPy CPU 特性检测为 `AVX512FP16=false`、`AVX512BF16=true`、`F16C=true`。F16C 提供半精度转换，不代表原生半精度算术。CPU FP16 图、权重与 KV 确实使用半精度，但声学 profile 中主要 Conv / MatMul 实际为 FP32；转换和内核选择没有在本轮带来时间或内存收益。CPU FP16 保留为显式实验配置，不能当作这台机器的默认加速建议。

INT8 仅量化 GPT 的常量矩阵乘法，使用动态 UINT8 激活、逐输出通道 INT8 权重；embedding、attention、归一化、KV 与采样接口保留 FP32。它会改变真实采样序列。`int8-fp16` 组合已提供路由与边界测试，本表没有测量该组合。

AMD 全图声学 FP16 保留了原工程误差筛查失败记录，当前按独立设备的有限输出实验记录准入；没有把失败标为音质通过。GPU profile 确认 FP16 神经计算。CPU / GPU 的具体误差和执行内核见[声学实验](acoustic-fp16-finite-20260927.md)。

GPU GPT 加速来自固定形状与 KV 驻留共同作用，不能单独归因于 FP16。首次原型的 I/O 绑定错误曾导致 NaN，已通过每步重新绑定小型 CPU 输入修复；失败证据保留，正式完整请求没有依靠预热绕过错误。共享采样入口现在拒绝任何原始非有限 logits，错误不会伪装成 token 0 继续生成。细节见[静态 GPT 实验](directml-gpt-static-20260927.md)和 [ADR 0006](../../docs/adr/0006-device-precision-and-directml-kv.md)。

## 试听与使用

六个未处理 WAV 保存在 `outputs/precision-listening-20260927/`，清单为同目录的 `试听说明.md`。五份长句分别对应 CPU FP32、CPU FP16、CPU INT8、AMD FP32 和 AMD FP16 1280；另有 AMD FP16 512 的短句。复制后的文件与基准原 WAV 哈希一致，未调音量、裁剪或后期处理。人工听音和 ASR 尚未验收，重点比较漏字、尾句、语速、齿音、音色和底噪。

本机可直接加载 `configs/tts_infer.cpu-fp16.yaml`、`configs/tts_infer.cpu-int8-threads8.yaml`、`configs/tts_infer.amd-fp16.yaml`、`configs/tts_infer.amd-fp16-cap1280.yaml` 或 `configs/tts_infer.amd-fp16-cap512.yaml`。它们包含本机资源绝对路径，其他安装按[设备指南](../../docs/cpu-amd.md)准备资源。CPU / GPU 选择在 `sakuratts.backend`，精度在 `sakuratts.profile`，线程和容量在 `sakuratts.runtime_options`；选择配置不会转换权重。

```powershell
.venv-amd/Scripts/python.exe -m sakuratts serve -c configs/tts_infer.amd-fp16-cap1280.yaml --runtime-mode managed
```

CPU FP16、CPU INT8 8 线程、AMD FP32、AMD FP16 1280 各完成两轮真实 HTTP 唤醒、合成、休眠。8 次均返回有效 3.42 秒 PCM WAV；每轮所有观测到的自有推理进程均退出。记录在 `results/precision-listening/managed/summary.json`。

另以 AMD FP16 1280 采集 151 个 WDDM 样本，全部匹配 780M LUID，没有无效、重复或未解析实例。长句 Dedicated / Shared / Committed 峰值分别为 91,496,448 / 1,386,618,880 / 1,477,705,728 字节；请求结束约 3 秒仍有分配。休眠后自有进程退出，实例消失，缺失实例不记为测量零。WDDM 属于独立资源探针，不替代上表时间，也不能加到进程工作集上。

最终产品回归 469 项：468 项通过、1 项跳过；相关实验工具 10 项通过。CPU / AMD FP16 的实际依赖与资源诊断通过，`pip check` 通过。完整日志为 `results/precision-listening/full-tests.txt`。其他硬件、驱动、音色与共存负载尚未验证。
