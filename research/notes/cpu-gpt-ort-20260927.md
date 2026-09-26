# CPU GPT：统一 ONNX 图与 INT8 实验

2026-09-27，在 Ryzen 7 7840HS 上，将 GPT 的 Prefill 和 Decode 放进同一张 ONNX 图后，两条日语输入的 FP32 实际语义生成耗时分别从 **2.218 秒降到 1.622 秒**、**14.476 秒降到 10.973 秒**。与 NumPy 路径相比，短句减少约 26.9%，长句减少约 24.2%；固定种子下，完整采样序列与停止条件相同。

这是 GPT 阶段的测量，不包含文本前端、声学模型、PCM 转换或播放。正式实现保留 NumPy 路径，ONNX FP32 需要显式选择。INT8 更快，但采样序列已经变化，只保留为研究候选。完整数值、输入与图身份、逐请求 token 摘要和原始报告哈希见[紧凑证据](../experiments/data/2026-09-27-cpu-gpt-ort.json)。

## 测量条件

硬件为 Ryzen 7 7840HS，8 核 16 线程；Windows 11 10.0.22631，Python 3.12.14、NumPy 2.4.6、onnxruntime-directml 1.24.4，实际执行 provider 为 CPU。NumPy 和 ORT 均使用 2 个推理线程。

模型使用 `models/navi-cpu-amd/gpt` 中的原生 FP32 权重：24 层、hidden_dim=512、16 个 attention heads、词表大小 1025。原 checkpoint SHA-256 为 `250400bfebe8547ef9167dc0225239eb2cac57e8648afbaa0834ac3ff95d4fcf`。短、长句沿用[CPU / DirectML 实测](cpu-directml-780m-20260927.md)中的参考条件和音素，输入来自已经保存的真实请求结果。本实验直接重放前端输出，不重新运行前端。

固定历史测试使用同一组保存的采样 token 驱动各候选，每一步都比较 logits，避免不同生成长度影响计时。另一次真实采样让每个候选沿自己的历史继续生成，使用 seed=1234、top_k=15、top_p=1、temperature=1、repetition_penalty=1.35 与相同停止规则。

统一图每个候选、每条输入先预热一次，随后计时一次固定历史回放，再运行一次真实采样。样本量不足以估计稳定分布；执行顺序没有轮换，电源、温度和后台负载没有独立量化。

## FP32 结果

| 输入与路径 | Prefill | 固定历史 Decode | 真实采样全过程 | 含 EOS 的采样步数 |
| --- | ---: | ---: | ---: | ---: |
| 短句 NumPy | 0.566 s | 1.633 s | 2.218 s | 79 |
| 短句 ORT FP32 | 0.336 s | 1.217 s | 1.622 s | 79 |
| 长句 NumPy | 1.405 s | 12.848 s | 14.476 s | 511 |
| 长句 ORT FP32 | 0.929 s | 10.328 s | 10.973 s | 511 |

真实采样时间来自另一轮运行，包含 Prefill、Decode、采样和停止判断，不能直接与前两列相加核对。两条输入的完整采样序列都相同，停止记录均为 `argmax_eos` 与 `sample_eos`。固定历史下没有 argmax 分歧；短句最大 logits 绝对误差为 3.05e-5，长句为 3.48e-5，输出均为有限值。

较早的 Decode-only 原型使用 NumPy Prefill 加 ORT Decode。其短句 78 步 Decode 中位数从 1.898 秒降到 1.256 秒，但需要同时保留两种执行器。统一图把 Prefill 也交给 ORT，避免为了 Prefill 再加载整套 NumPy Transformer 权重。早期原型的两次回放数据另存于证据的 `studies.decode_only_short`，没有与统一图合并。

## 缓存与内存

运行时按 `[layer, capacity, head, head_dim]` 预分配 KV 缓存。各层有效前缀是连续数组，直接作为 ORT 输入；图只返回当前输入对应的新 K/V 行，随后写入原缓冲区。Decode 每步运行一次完整 Transformer 图，减少 Python 层算子调用，并允许 ORT 优化线性层。

ORT 的 CPU GEMM 实现可以为常量矩阵预打包权重，见 [ORT 1.24.4 GEMM 源码](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.24.4/onnxruntime/core/providers/cpu/math/gemm.cc)。本实验测的是整张图的优化效果，没有单独隔离预打包、算子融合与 Python 调用减少各自的收益。

Prefill 与 Decode 共用一张图，模型文件约 304.8 MB，另有约 13.9 MB 的 embedding、位置编码和 BERT 投影数组。运行时不导入 Torch 或 ONNX Python 包。ONNX 仅在准备 sidecar 时使用。

Windows CPU provider 的初版实验在零长度历史轴上触发了原生整数除零。运行时现在固定保留一个全零、永久遮罩的缓存槽，让矩阵维度保持非零；这不占用用户配置的语义容量。小模型独立 FP64 对照以及本次真实模型采样验证了遮罩和写入位置。

独立产品加载器只加载 GPT，依次生成短、长句，按 50 ms 采样当前模型进程，取得以下资源数据：

| 指标 | 实测 |
| --- | ---: |
| 模型加载后工作集 | 373.1 MiB |
| 采样工作集峰值 | 501.9 MiB |
| 采样私有内存峰值 | 1151.8 MiB |
| 关闭模型并回收后工作集 | 74.0 MiB |

这次资源检查没有加载声学模型，也不是完整 TTS 的占用。工作集和私有内存是不同口径，不能相加。其短、长句均得到与保存的 NumPy 结果相同的 token，logits 和采样概率均为有限值，进程未导入 Torch 或 ONNX Python 包。

统一图目前使用完整 Prefill 注意力矩阵，明确要求 `gpt_prefill_query_chunk_size=0`。它不实现 NumPy 路径的查询分块，传入非零值会报错。更长输入的 Prefill 内存仍需单独评估，不能用这两条输入的峰值推断容量上限附近的占用。

## GPT 线程数

使用正式 FP32 加载器，在同一进程中分别按 `1 → 2 → 4 → 8` 与 `8 → 4 → 2 → 1` 的顺序测试两轮，每轮也反转短、长句顺序。每个配置先预热两种输入形状的 Prefill 和前 8 个 Decode token，再重放完整固定历史。模型加载和权重哈希检查不计入时间。

| GPT 线程数 | 短句总耗时中位数 | 长句总耗时中位数 | 短句 CPU 时间中位数 | 长句 CPU 时间中位数 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.947 s | 14.519 s | 1.633 CPU-s | 11.711 CPU-s |
| 2 | 1.562 s | 12.119 s | 1.281 CPU-s | 10.836 CPU-s |
| 4 | 1.444 s | 9.495 s | 1.070 CPU-s | 8.992 CPU-s |
| 8 | 1.523 s | 9.032 s | 1.367 CPU-s | 6.875 CPU-s |

4 线程相对 2 线程，短句耗时减少约 7.5%，长句减少约 21.6%。8 线程的长句比 4 线程再快约 4.9%，但短句慢约 5.5%。这组数据支持将 GPT 与声学模型的线程数分开配置；本机 4 线程适合兼顾短、长句，8 线程的收益主要出现在长句。

16 条记录的 logits 均为有限值，跨线程与跨轮次逐位一致，没有 argmax 分歧。各阶段墙钟时间、CPU 时间、采样工作集和私有内存见证据的 `thread_sweep`。这是同一进程内轮换配置，分配器保留及先前输入会影响工作集，不能把各配置的 RSS 差异全部归因于线程数。CPU 时间为整个模型进程的累计 CPU 时间，也不能当作整机功耗。

该轮固定历史总耗时不包含抽样，且发生在另一组运行中，不与前表的真实采样耗时合并。每种配置每条输入只有两次计时，仍不足以判断长时间负载下的稳定性。

## INT8 候选

量化只处理常量权重的 MatMul；embedding、attention 与 LayerNorm 保留 FP32。实验使用 ORT 动态 UINT8 activation / INT8 weight 量化，分别比较逐张量和逐输出通道权重。[ORT 量化文档](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)说明了动态量化接口及其适用范围；质量结果以本实验为准。

| 输入与候选 | Prefill | 固定历史 Decode | 真实采样全过程 | 采样步数 | 固定历史 argmax 分歧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短句逐张量 INT8 | 0.157 s | 0.693 s | 0.933 s | 80 | 2 |
| 短句逐通道 INT8 | 0.150 s | 0.682 s | 0.850 s | 79 | 2 |
| 长句逐通道 INT8 | 0.532 s | 6.930 s | 7.320 s | 500 | 14 |

短句两种 INT8 的真实采样都在第 27 次抽样首次偏离 FP32。长句的逐通道 INT8 最终生成 500 步，FP32 为 511 步。固定历史 logits 最大误差达到约 0.94–1.66；输出虽然为有限值，也正常遇到 EOS，但这些事实不能证明文本、韵律和音色合格。

INT8 的图约 77 MB，解码耗时明显减少；真实采样速度还混入了生成长度变化，不能全部视为相同输出下的提速。本次没有人工试听或 ASR 验收，也没有将 INT8 加入正式配置或默认路径。研究脚本保留完整采样结果和图哈希，便于后续复现。

## 实现与验证入口

正式加载器为 `sakuratts.backends.cpu.onnx_gpt.ONNXCPUGPT`。它读取原 GPT 包下的 `onnx/` sidecar，核对原 manifest、原权重文件、checkpoint 身份，以及图和 embedding 文件哈希；仅加载 sidecar 推理数据，不解压原 NumPy Transformer 权重。加载失败会释放 ORT session，推理失败会丢弃当前 KV 状态。

准备 sidecar：

```powershell
python -m sakuratts._internal.conversion.export_gpt_onnx --gpt models/navi-cpu-amd/gpt
```

正式转换器只导出 FP32，默认写入新的 `gpt/onnx/` 目录，已有目录会拒绝覆盖。原模型和许可证保留，sidecar 同时携带许可证副本。

重现统一图与 INT8 研究：

```powershell
python research/tools/cpu_gpt_ort.py export --package models/navi-cpu-amd/gpt --output .cache/gpt-ort-study --unified --int8
python research/tools/cpu_gpt_ort.py profile --model models/navi-cpu-amd --result results/cpu-amd-final/cpu/result.json --graphs .cache/gpt-ort-study --output results/gpt-ort-study-short --case short --repeats 2 --sampling
```

`tests/test_cpu_onnx_gpt.py` 的 6 项测试已通过，使用真实小 ONNX 图对照独立完整前缀 FP64 计算，覆盖 Prefill、Decode、跨请求重置、固定容量、推理失败回收、源权重与 sidecar 校验，以及不支持的分块配置。微基准、实际加载器资源检查和小模型回归属于不同验证层；它们不能替代整链 TTS 的耗时、内存和听感检查。
