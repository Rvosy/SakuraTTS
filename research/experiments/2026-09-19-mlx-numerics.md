# MLX 日文长句的数值误差定位

日期：2026-09-19。Apple M4，FP32，固定“朱雀院红叶”V2Pro GPT 权重。所有步骤沿用官方采样历史，不做自由采样，也不生成音频。

## 当前结论

原 MLX FP32 路径在扩展语料的日文长句上未通过原有容差。误差可以追溯到 Prefill 的一个文本位置：微小的 FP32 运算差异经过多层计算放大，留在 K/V 中；第 329 步对该位置的注意力较高，使误差传到输出。现有证据没有发现 token、位置、掩码或缓存索引错位。

通用 CPU FP64 Prefill 接续原 MLX FP32 Decode 的独立候选，已在 10 条回归、1805 步上通过原容差。默认 FP32 行为保持不变，没有放宽 `1e-4 + 1e-5 * abs(official)`，也没有按特定句子或位置加入修补。该结果只覆盖固定输入和历史的 GPT 计算，完整 TTS 与其他模型仍待验证。

## 失败范围与对照可信度

`ja-long` 共 374 步、每步 1025 个 logits。只有零起始第 329 步的 11 个值越界，最大绝对误差 `0.000293731689453125`，最大容差比 `1.448657`。11 个词表索引是 `9, 74, 83, 176, 329, 584, 607, 654, 772, 1007, 1024`。第 328、330 步的最大绝对差分别约 `2.96e-5`、`2.67e-5`。

第 329 步的有效 KV 长度为 756，其中 314 个文本位置、113 个参考语义位置；当前输入的音频位置是 441。该步 top-1 都是 token 522，官方前两名分差约 0.61659。全部 374 步 top-1 一致，但这不能替代数值验收或语音质量检查。

`research/tools/diagnose_mlx_step.py` 增加了独立诊断图：

- 标准 MLX 诊断保存中间值后，374 步输出与原 MLX 实验逐元素相同。
- PyTorch 诊断按官方 `F.linear → SDPA → 残差/LayerNorm → ReLU FFN → 残差/LayerNorm` 顺序运行，使用同一转换包。MPS 的 374 步 logits 与原始官方 trace 逐元素相同。
- 截取 Prefill 和 Decode 的层间数组，实验只替换指定算子；这一诊断阶段没有修改 `src/sakuratts/backends/mlx/gpt.py`。

## 误差如何传递

第 329 步，第 0–16 层的 attention 输出差异均在数个 `1e-6` 内。第 17 层第 2 个头的差异升至 `1.462698e-4`，随后传到后续层与输出投影。层号、头号和位置均从 0 开始。

这个头给文本位置 308 的注意力概率约为 22.727%。该位置在 Prefill 后已有较大的 K/V 差异：第 13 层约 `1e-4`，第 17 层的 K、V 最大差分别约 `1.824e-4`、`1.836e-4`。Prefill 位置 308 的 hidden state 在第 6–12 层逐步放大；第 12 层输出差为约 `1.14e-3`。因此，仅检查 Prefill 最后一个位置的 logits 会漏掉这些留在文本 K/V 里的差异。

`research/tools/analyze_gpt_capture.py` 用 CPU NumPy FP64 对保存的同一组输入计算各算子。它只辅助定位，不改变验收容差。第 329 步第 17 层的结果为：

| 检查 | attention 输出最大绝对差 |
|---|---:|
| 官方 SDPA 相对其自身输入的 FP64 计算 | `4.38e-7` |
| MLX SDPA 相对其自身输入的 FP64 计算 | `6.29e-7` |
| 两组输入各自以 FP64 计算后的差异 | `1.46536e-4` |
| 在离线计算中，仅把 MLX 的 KV 位置 308 换成官方值 | `1.77978e-6` |

这组反事实计算说明，Decode 此处的主要误差来自输入 K/V，而不是本次 SDPA 的局部舍入。替换单个位置只用于定位，没有进入候选运行时。

Prefill 中 LayerNorm 也会放大输入差异。例如第 12 层，位置 308 的残差差异约 `2.60e-4`，norm1 输出差异约 `1.65e-3`；两边 LayerNorm 相对各自输入的 FP64 局部误差都只有约 `1e-6`。因此，看到归一化后误差增大，并不能直接认定 LayerNorm 内核错误。

## 算子替换结果

以下每项都从 Prefill 开始完整重放 374 步，保持同一权重、输入和历史。

| MLX 诊断路径 | 越界元素数 | 最大绝对差 | 最大容差比 | 处理 |
|---|---:|---:|---:|---|
| 原路径 | 11 | `2.93732e-4` | 1.44866 | 保留为失败对照 |
| 显式 QK、softmax、V | 1 | `1.92642e-4` | 1.06749 | 未通过，不采用 |
| 显式均值、方差与 LayerNorm | 15 | `3.16620e-4` | 1.64387 | 退化，不采用 |
| `addmm` 融合线性层 bias | 11 | `2.93732e-4` | 1.44866 | 与原路径逐元素相同，无数值收益 |

这些时间包含逐步 CPU 复制与中间值保留，不能与正常生成计时比较。本轮只定位正确性，没有资源或速度收益结论。

## 高精度 Prefill 候选

完整 CPU FP64 Prefill 的末位置 logits 仍在原容差内，最大绝对差约 `2.05e-5`。它对官方第 17 层 K/V 的最大差约 `2e-5`，比原 MLX 的约 `2e-4` 小。位置 308 的第 12 层输出相对 FP64 的差异，官方约 `1.09e-4`，MLX 约 `1.25e-3`。

PyTorch CPU FP64 Prefill 的本次诊断计算约 0.328 秒，不含权重加载，仅作为高精度参照。

随后新增 `research/tools/fp64_prefill_replay.py`，用 NumPy 独立执行相同 Prefill；完成验证后，计算部分提取到了 `src/sakuratts/backends/mlx/gpt_prefill.py`。每层从同一 FP32 转换包读取权重并临时转为 FP64，完成全部 Prefill 后把 K/V 和首步 logits 一次舍入为 FP32，再交给未修改的 MLX Decode。进程没有导入 PyTorch，也不常驻完整 FP64 权重副本。NumPy 与 PyTorch FP64 Prefill 的 K/V 舍入为 FP32 后，最大差仅 `7.28e-12`。

先验证日文长句，再运行全部 10 条：

| 样例 | 固定历史步数 | 最大 logits 绝对差 |
|---|---:|---:|
| 日文用户报告样例 | 121 | `3.43e-5` |
| 中文用户报告样例 | 147 | `3.05e-5` |
| 日文短句 | 24 | `2.34e-5` |
| 中文短句 | 42 | `3.15e-5` |
| 日文长句 | 374 | `5.34e-5` |
| 中文长句 | 461 | `4.15e-5` |
| 日文标点 | 212 | `8.49e-5` |
| 中文标点 | 175 | `4.05e-5` |
| 中英混合 | 132 | `4.58e-5` |
| 日英混合 | 117 | `3.03e-5` |

1805 步全部满足原容差，top-1 也全部一致。保存后重新读取每份 NPZ 独立核验，并检查运行清单中的文件哈希。

本轮 CPU Prefill 为 0.183–0.522 秒，包含每请求约 0.053–0.064 秒的权重读取与转换；CPU K/V 交给 MLX 并等待完成为 2.49–6.99 毫秒。先跑的单条日文长句中，这一步耗时 0.160 秒；新进程和首次分配的开销尚未通过重复测量拆开，不能只引用后续较小的数值。

整个 10 条运行的 MLX 分配器 peak 为约 440.39 MiB，进程 lifetime peak RSS 为约 717.34 MiB。MLX 计数不包含 NumPy CPU 临时数组，两个数不能相加；它们也不是 NVIDIA 显存。释放模型并清理 MLX 缓存后 active/cache 均为 0，进程 RSS 仍约 390.88 MiB。

上述阶段时间与资源值来自诊断运行，Decode 每步复制 logits 到 CPU，不能直接作为正常速度结论。后续正常计时如下；本实验仍未生成音频。

## 正常计算成本

`research/tools/gpt_benchmark.py --backend mlx-fp64-prefill` 增加独立选项。以 Lite 固定历史路径为控制，双方用同一份 10 条官方历史、FP32 源权重、1024 槽 KV，每例预热 2 次、正式测量 5 次。每条样例先在计时外验证所有 logits；失败则停止该例测量。Lite 的 1805 步与官方逐元素相同，高精度候选的 1805 步全部满足原容差。

计时内包含 CPU 输入处理、每次请求逐层读取与转换权重、完整 CPU Prefill、K/V 交接和全部 Decode 投影；边界等待设备完成。Decode 不逐步复制 logits 到 CPU，权重读取的诊断计时也在正式测量中关闭。每次结束后才复制最后一个输出，核对它与计时外验证一致。

| 样例 | Lite 中位数 | 高精度候选中位数 | 请求时间变化 |
|---|---:|---:|---:|
| 日文用户报告样例 | 1.135 s | 0.909 s | -19.9% |
| 中文用户报告样例 | 1.376 s | 1.029 s | -25.2% |
| 日文短句 | 0.237 s | 0.330 s | +39.2% |
| 中文短句 | 0.403 s | 0.467 s | +16.0% |
| 日文长句 | 3.556 s | 2.671 s | -24.9% |
| 中文长句 | 4.423 s | 3.013 s | -31.9% |
| 日文标点 | 2.029 s | 1.422 s | -29.9% |
| 中文标点 | 1.663 s | 1.112 s | -33.1% |
| 中英混合 | 1.265 s | 0.897 s | -29.1% |
| 日英混合 | 1.121 s | 0.833 s | -25.6% |

高精度 Prefill 有固定 CPU 成本，短句没有普遍加速。两组在不同新进程里依次运行，已串行安排项目内其他重 CPU/GPU 任务，但系统后台负载和温度未完全控制。完整原始范围保存在结果中，不用这五次样本推断稳定 p95。原纯 FP32 MLX 的长句失败保留，不把它作为已通过的十例性能对照。

| 同轮资源口径 | Lite | 高精度候选 |
|---|---:|---:|
| 新进程加载时间，单次 | 2.898 s | 0.224 s |
| 模型加载后 RSS 边界 | 802.61 MiB | 402.56 MiB |
| 进程 lifetime peak RSS | 904.05 MiB | 742.02 MiB |
| 请求完成后的框架 active/allocated | MPS 407.69 MiB | MLX 399.89 MiB |
| 两边实际 KV 字节数 | 96 MiB | 96 MiB |
| 释放并清缓存后的框架 active/allocated | 0 | 0 |
| 释放后进程 RSS | 738.48 MiB | 316.14 MiB |

候选的 MLX 分配器 peak 最大约 440.40 MiB；MPS 只有边界计数，未测同口径分配器峰值。RSS、MPS 与 MLX 计数不能相加或当作同一种显存指标。两者都处于 Apple 统一内存环境。新进程没有清空 OS 文件缓存，加载时间也不代表首次安装或下载时间。

当前环境 distribution 文件分别为 1,001,659,727 与 270,360,955 字节，不含 Python 解释器和符号链接目标；该差值仍不是最终 TTS 安装包收益。转换权重、文本资源与声学模块另计。

## 显式精度接口与移植复核

`MLXGPT.load(..., prefill_precision="fp64")` 选择 CPU FP64 Prefill，省略时仍为原 FP32 路径。`prefill(..., precision="fp32")` 可以只覆盖本次请求，不改变加载时的选择。两种模式都使用 FP32 转换包与 FP32 Decode，没有按语言、句长或失败样例自动切换。

高精度计算位于纯 NumPy 模块，K/V 交接集中在 `MLXGPT` 内，Harness 复用该实现。正式调用默认关闭分阶段插桩；诊断可以用 `profile=True` 读取 `prefill_profile`。原始权重与转换包均未修改。

移入 `src` 后重新运行十例对照，1805 步输出与移植前逐元素相同；另对两条短句各做 2 次预热、5 次正常测量，得到日文 0.338 秒、中文 0.405 秒。它们与前轮的波动一起保留，不能把一次中文短句改善泛化成所有短句获益。默认 FP32 日文长句重新运行仍以 `numerical_mismatch` 和退出码 1 报告同一失败，全部 logits 与原失败记录逐元素相同，没有被精度选项掩盖。

## 原始证据

以下路径均相对于 `SakuraTTS-References`。各运行目录保存命令、输入哈希、源码副本、logits、所选步骤的层间数组和清单，不覆盖历史实验。

- `runs/20260919T111015.587245Z-gpt-step-analyze-standard/`：11 个失败坐标、阈值和相邻步骤。
- `runs/20260919T111015.724010Z-gpt-step-mlx-standard/`：标准 MLX，第 329 步各层。
- `runs/20260919T111054.306104Z-gpt-step-torch-standard/`：MPS 官方算子顺序，全部 logits 与官方相同。
- `runs/20260919T111124.956382Z-gpt-step-mlx-eager-attention/`：显式 attention。
- `runs/20260919T111220.803083Z-gpt-step-mlx-eager-norm/`：显式 LayerNorm。
- `runs/20260919T111413.580258Z-gpt-step-mlx-fused-linear/`：融合线性层 bias。
- `runs/20260919T111310.941574Z-gpt-step-mlx-standard/`、`runs/20260919T111321.569473Z-gpt-step-torch-standard/`：Prefill 位置 308 的层间输出及全部 K/V。
- `runs/20260919T111626.531645Z-gpt-capture-fp64-cpu/`、`runs/20260919T111644.716181Z-gpt-capture-fp64-cpu/`：Decode 与 Prefill 的离线 FP64 算子参照。
- `runs/20260919T111720.012742Z-gpt-step-torch-standard/`：CPU FP64 Prefill；具体 dtype 以 `result.json` 为准。
- `runs/20260919T112325.650335Z-fp64-prefill-mlx-decode/`：NumPy 高精度 Prefill 接续 MLX，首次日文长句验证，含完整 Prefill K/V。
- `runs/20260919T112408.005521Z-fp64-prefill-mlx-decode/`：10 条、1805 步全部通过；分阶段时间、资源、源码快照及原始 logits。
- `runs/20260919T112527.343325Z-gpt-numerics-verification/`：保存数组的独立复核与清单验证。
- `runs/20260919T113357.532977Z-gpt-benchmark-lite-gpu-kv1024/`、`runs/20260919T113651.752062Z-gpt-benchmark-mlx-fp64-prefill-gpu-kv1024/`：十例正常时间与资源对照。
- `runs/20260919T114000.454585Z-gpt-fp64-benchmark-comparison/`：双方输入、源码、输出哈希及原容差的独立复核。
- `runs/20260919T114238.467547Z-mlx-gpt-self-test-cpu/`：小模型完整前缀 oracle、两种 Prefill 与单次精度覆盖。
- `runs/20260919T114359.220915Z-fp64-prefill-mlx-decode/`：移入 `src` 后的十例重放。
- `runs/20260919T114443.795328Z-gpt-benchmark-mlx-fp64-prefill-gpu-kv1024/`：移植后的两条短句正常计时。
- `runs/20260919T114508.944024Z-mlx-gpt-gpu/`：默认 FP32 路径仍保留原失败。
- `runs/20260919T114800.487285Z-gpt-src-precision-verification/`、`runs/20260919T114800.616347Z-mlx-gpt-self-test-cpu/`：移植前后逐元素复核及最终 CPU 小图检查。

## 复现入口

在项目根目录运行，GPU 实验与其他性能测量串行安排。默认模式仅读取数组：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
OFFICIAL="$REF/runs/20260919T104548.448908Z-official-mps"
PACKAGE="$REF/models/converted/20260919T104221.749176Z-010197bfc30b-gpt-fp32"
"$REF/.venv-mlx-macos/bin/python" research/tools/diagnose_mlx_step.py --references "$REF" --official-run "$OFFICIAL" --mlx-run "$REF/runs/20260919T110150.875057Z-mlx-gpt-gpu"
"$REF/.venv-mlx-macos/bin/python" research/tools/diagnose_mlx_step.py --references "$REF" --official-run "$OFFICIAL" --package "$PACKAGE" --mode mlx --device gpu
"$REF/.venv-official-macos/bin/python" research/tools/diagnose_mlx_step.py --references "$REF" --official-run "$OFFICIAL" --package "$PACKAGE" --mode torch --device cpu --torch-dtype fp64 --step 0 --stop-after-step --capture-position 308
OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 "$REF/.venv-mlx-macos/bin/python" research/tools/fp64_prefill_replay.py --references "$REF" --package "$PACKAGE" --official-runs "$OFFICIAL" "$REF/runs/20260919T110311.350332Z-official-mps" --cases ja-reported-intro zh-reported-greeting ja-short zh-short ja-long zh-long ja-punctuation zh-punctuation mixed-zh-en mixed-ja-en
```

`--variant eager-attention`、`eager-norm`、`fused-linear` 用于独立对照。`--prefill-override` 只在诊断中接续同一 trace、同一模型的完整 Prefill K/V；其时间包含被丢弃的原 MLX Prefill 和归档读取，不是高精度候选的性能数据。

诊断 Harness 现在也用 `numerical_mismatch` 和退出码 1 表示超差，避免把调查执行完毕误读为数值通过。离线复核 `runs/20260919T112919.163735Z-gpt-step-analyze-standard/` 保留原 11 个超差坐标；历史结果未回写。
