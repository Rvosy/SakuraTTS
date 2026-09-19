# MLX 日文长句的数值误差定位

日期：2026-09-19。Apple M4，FP32，固定“朱雀院红叶”V2Pro GPT 权重。所有步骤沿用官方采样历史，不做自由采样，也不生成音频。

## 当前结论

原 MLX FP32 路径在扩展语料的日文长句上未通过原有容差。误差可以追溯到 Prefill 的一个文本位置：微小的 FP32 运算差异经过多层计算放大，留在 K/V 中；第 329 步对该位置的注意力较高，使误差传到输出。现有证据没有发现 token、位置、掩码或缓存索引错位。

通用 CPU FP64 Prefill 接续原 MLX FP32 Decode 的独立候选，已在 10 条回归、1805 步上通过原容差。没有修改默认运行时，没有放宽 `1e-4 + 1e-5 * abs(official)`，也没有按特定句子或位置加入修补。该结果只覆盖固定输入和历史的 GPT 计算，完整 TTS 与其他模型仍待验证。

## 失败范围与对照可信度

`ja-long` 共 374 步、每步 1025 个 logits。只有零起始第 329 步的 11 个值越界，最大绝对误差 `0.000293731689453125`，最大容差比 `1.448657`。11 个词表索引是 `9, 74, 83, 176, 329, 584, 607, 654, 772, 1007, 1024`。第 328、330 步的最大绝对差分别约 `2.96e-5`、`2.67e-5`。

第 329 步的有效 KV 长度为 756，其中 314 个文本位置、113 个参考语义位置；当前输入的音频位置是 441。该步 top-1 都是 token 522，官方前两名分差约 0.61659。全部 374 步 top-1 一致，但这不能替代数值验收或语音质量检查。

`harness/diagnose_mlx_step.py` 增加了独立诊断图：

- 标准 MLX 诊断保存中间值后，374 步输出与原 MLX 实验逐元素相同。
- PyTorch 诊断按官方 `F.linear → SDPA → 残差/LayerNorm → ReLU FFN → 残差/LayerNorm` 顺序运行，使用同一转换包。MPS 的 374 步 logits 与原始官方 trace 逐元素相同。
- 截取 Prefill 和 Decode 的层间数组，实验只替换指定算子；`src/sakuratts/mlx_gpt.py` 保持不变。

## 误差如何传递

第 329 步，第 0–16 层的 attention 输出差异均在数个 `1e-6` 内。第 17 层第 2 个头的差异升至 `1.462698e-4`，随后传到后续层与输出投影。层号、头号和位置均从 0 开始。

这个头给文本位置 308 的注意力概率约为 22.727%。该位置在 Prefill 后已有较大的 K/V 差异：第 13 层约 `1e-4`，第 17 层的 K、V 最大差分别约 `1.824e-4`、`1.836e-4`。Prefill 位置 308 的 hidden state 在第 6–12 层逐步放大；第 12 层输出差为约 `1.14e-3`。因此，仅检查 Prefill 最后一个位置的 logits 会漏掉这些留在文本 K/V 里的差异。

`harness/analyze_gpt_capture.py` 用 CPU NumPy FP64 对保存的同一组输入计算各算子。它只辅助定位，不改变验收容差。第 329 步第 17 层的结果为：

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

随后新增 `harness/fp64_prefill_replay.py`，用 NumPy 独立执行相同 Prefill。每层从同一 FP32 转换包读取权重并临时转为 FP64，完成全部 Prefill 后把 K/V 和首步 logits 一次舍入为 FP32，再交给未修改的 MLX Decode。进程没有导入 PyTorch，也不常驻完整 FP64 权重副本。NumPy 与 PyTorch FP64 Prefill 的 K/V 舍入为 FP32 后，最大差仅 `7.28e-12`。

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

这些阶段时间与资源值来自诊断运行，Decode 每步复制 logits 到 CPU。尚未做同条件正常计时、自由采样或音频验证，因此只保留为显式精度候选，不据此宣称更快或更省内存。

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

## 复现入口

在项目根目录运行，GPU 实验与其他性能测量串行安排。默认模式仅读取数组：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
OFFICIAL="$REF/runs/20260919T104548.448908Z-official-mps"
PACKAGE="$REF/models/converted/20260919T104221.749176Z-010197bfc30b-gpt-fp32"
"$REF/.venv-mlx-macos/bin/python" harness/diagnose_mlx_step.py --references "$REF" --official-run "$OFFICIAL" --mlx-run "$REF/runs/20260919T110150.875057Z-mlx-gpt-gpu"
"$REF/.venv-mlx-macos/bin/python" harness/diagnose_mlx_step.py --references "$REF" --official-run "$OFFICIAL" --package "$PACKAGE" --mode mlx --device gpu
"$REF/.venv-official-macos/bin/python" harness/diagnose_mlx_step.py --references "$REF" --official-run "$OFFICIAL" --package "$PACKAGE" --mode torch --device cpu --torch-dtype fp64 --step 0 --stop-after-step --capture-position 308
OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 "$REF/.venv-mlx-macos/bin/python" harness/fp64_prefill_replay.py --references "$REF" --package "$PACKAGE" --official-runs "$OFFICIAL" "$REF/runs/20260919T110311.350332Z-official-mps" --cases ja-reported-intro zh-reported-greeting ja-short zh-short ja-long zh-long ja-punctuation zh-punctuation mixed-zh-en mixed-ja-en
```

`--variant eager-attention`、`eager-norm`、`fused-linear` 用于独立对照。`--prefill-override` 只在诊断中接续同一 trace、同一模型的完整 Prefill K/V；其时间包含被丢弃的原 MLX Prefill 和归档读取，不是高精度候选的性能数据。

诊断 Harness 现在也用 `numerical_mismatch` 和退出码 1 表示超差，避免把调查执行完毕误读为数值通过。离线复核 `runs/20260919T112919.163735Z-gpt-step-analyze-standard/` 保留原 11 个超差坐标；历史结果未回写。
