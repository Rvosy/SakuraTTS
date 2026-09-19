# 自有声学模块组合与累计误差

日期：2026-09-19。已接成独立 MLX 声学链。全 GPU 路径首次出现中文单点超差；经定位后，显式 CPU 编码器配合 GPU flow / 声码器的两例、12 阶段全部通过。正常声学计算比官方控制慢约 29%–41%，MLX 缓存限额可以减少请求后的缓存占用，但会进一步增加延迟。

## 组合方式

`src/sakuratts/mlx_sovits.py` 直接衔接自有模块，外部布局保持 NCT。编码器输出 `mean/log_scale/mask` 后，按官方顺序计算 `mean + noise * exp(log_scale) * 0.5`，再执行 reverse flow、乘 mask 和完整声码器。中间值不经 CPU 转换；参考准备、原始文本处理与 GPT 生成不在本次声学检查中。

输入取自已保存的官方固定声学条件：相同 codes、phones、ge、ge512 和显式正态噪声。噪声来自该固定条件实验，不能当作原始完整请求中未捕获的随机数。所有比较沿用 `atol=1e-4, rtol=1e-5`，没有对某条句子或某个位置修改容差。

## 首次结果

| 指标 | 日文 | 中文 |
|---|---:|---:|
| 波形样本数 | 153,600 | 186,880 |
| decoder 输入最大绝对差 | `1.62125e-5` | `1.33514e-5` |
| 波形最大绝对差 | `7.17342e-5` | `1.09270e-4` |
| 波形 RMS 误差 | `1.65079e-6` | `2.47429e-6` |
| 超差波形元素 | 0 | 1 |

两例的码本、SSL 编码、文本编码、MRTE、第二段编码、mean、log_scale、mask、flow 输入/输出及 decoder 输入均通过；中文最终波形仍判 `numerical_mismatch`，进程返回 1。单独给声码器官方输入时两例均通过，这次失败需要检查误差如何随组合输入进入波形。

诊断 Harness 另按官方单片段后处理导出 WAV：振幅超过 1 才归一化，末尾追加 0.3 秒静音，再乘 32768 转 int16。相对同样处理的固定官方波形，日文最大差 2 个整数刻度，中文 4 个；该观察不替代浮点判据、ASR 或试听。本轮尚未进行新的内容和音色验收。

运行只导入 MLX、NumPy 与标准库，没有 PyTorch 或上游推理代码。释放全部模块和缓存后，MLX active/cache 均为 0。阶段捕获保留中间数组，其资源数据属于诊断，不能当作正常请求峰值或 NVIDIA 显存。

## 数值定位与显式 CPU 编码器

逐阶段反事实检查表明，组合差异主要随编码器输出进入后续计算。给投影层相同的官方 hidden state，可以精确复现 mean、log_scale 和 flow 输入；把后续 flow / 声码器输入固定为官方值时，两条完整波形通过。尝试更高精度的 weight normalization 未解决超差，未采用。详细矩阵及未采用的实验见[声学数值定位](2026-09-19-mlx-sovits-numerics.md)。

`MLXSoVITS.load(package, encoder_device="cpu")` 现在显式选择 CPU 编码器，后续 latent、flow 和声码器仍在加载时指定的 GPU stream 执行。设备选择对所有输入相同，不按句子或位置切换，也没有改变权重精度、容差和噪声。

| CPU 编码器组合结果 | 日文 | 中文 |
|---|---:|---:|
| 完整波形最大绝对差 | `7.40141e-5` | `4.92409e-5` |
| 完整波形 RMS 误差 | `1.74400e-6` | `1.68282e-6` |
| 12 阶段超差元素总数 | 0 | 0 |

从诊断代码移入源模块后重新运行，两条全部 12 阶段数组与独立诊断逐位相同。原全 GPU 模式与失败证据继续保留，尚未作为通过路径。这里仍只覆盖两条固定语义和参考条件；长句、其他语义历史和音质需要继续验证。

## 正常声学成本与缓存代价

`harness/sovits_benchmark.py` 在新进程中分别加载同一解码包的 650 个张量。官方控制只构造 decode 所需的官方模块，删除 codebook 训练缓冲，没有加载完整 TTS 的参考准备或文本模型。两端都接收相同 codes、phones、ge、ge512 和 noise，因此这些结果是纯声学成本，不是完整请求或首包延迟。

每个后端先在计时外验证全部波形，再做 2 次预热和 5 次测量。计时包含输入转换与传输、计算和边界同步；不捕获阶段、不复制最终波形到 CPU、不查询内存。项目内其他重计算串行安排，普通桌面负载没有控制。

| 声学请求中位数 | 官方 MPS | MLX 默认缓存 | MLX 256 MiB 缓存限额 |
|---|---:|---:|---:|
| 日文 4.80 秒波形 | 0.16070 s | 0.22718 s | 0.26764 s |
| 中文 5.84 秒波形 | 0.19176 s | 0.24775 s | 0.30667 s |
| 日文 RTF | 0.0335 | 0.0473 | 0.0558 |
| 中文 RTF | 0.0328 | 0.0424 | 0.0525 |

MLX 默认缓存的声学路径慢约 41.4% / 29.2%，没有加速收益。限制缓存后，又比默认慢约 17.8% / 23.8%。这是为了减少请求后保留的空闲缓存而付出的实际代价，是否适合完整引擎还需连同 GPT、前端和空闲策略测量。

| 资源口径 | 官方 MPS | MLX 默认缓存 | MLX 256 MiB 缓存限额 |
|---|---:|---:|---:|
| 请求后框架 active / allocated | 293.49 MiB | 165.15 MiB | 165.15 MiB |
| 请求后框架 cache | 未单列 | 2654.96 MiB | 中文 258.26 MiB |
| 请求后 MPS driver | 1458.70 MiB | 不适用 | 不适用 |
| MLX allocator peak | 未提供同口径值 | 1052.52 MiB | 1052.45 MiB |
| OS 进程全生命周期 peak RSS | 745.25 MiB | 370.70 MiB | 297.42 MiB |

这些列不能横向当成同一种显存：MPS driver、MLX cache、allocator peak 和 RSS 的覆盖不同，不能相加。特别是默认 MLX 虽然 RSS 较低，却保留了约 2.59 GiB 分配器缓存，不能仅凭 RSS 宣称整体占用更小。

缓存限额前后的两条浮点波形逐位相同，原数值对照仍通过。`mx.set_cache_limit(268435456)` 将原值 16,320,875,724 字节改为 256 MiB；实际请求后 cache 为日文 265.36 MiB、中文 258.26 MiB，因此它不是硬性的“绝不超过 256 MiB”。相对默认，中文请求后缓存少约 90.3%，而 allocator peak 几乎不变，说明该设置主要改变释放后缓存保留，没有解决计算时的临时空间。

## 原始证据

- 官方条件：`SakuraTTS-References/runs/20260919T111545.846735Z-sovits-fixed-official-mps/`。
- 初次组合：`SakuraTTS-References/runs/20260919T120834.391918Z-mlx-sovits-complete-gpu/`，含 12 阶段数组、完整浮点波形、WAV、输入/源码哈希与命令。
- 所用包：`models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32/`。
- 显式 CPU 编码器源模块复跑：`runs/20260919T121811.736035Z-mlx-sovits-complete-gpu/`。
- 源模块移植逐值核对：`runs/20260919T122138.464428Z-native-source-migration-check/`。
- 官方正常声学控制：`runs/20260919T121910.618148Z-sovits-benchmark-official-gpu/`。
- MLX 正常声学对照：`runs/20260919T121926.065779Z-sovits-benchmark-mlx-gpu/`。
- MLX 缓存限额：`runs/20260919T122129.358241Z-sovits-benchmark-mlx-gpu/`。

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/mlx_sovits_replay.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T111545.846735Z-sovits-fixed-official-mps" \
  --device gpu --encoder-device cpu
```

完整基准命令保存在对应 `result.json` 的 argv 中。移除 `--encoder-device cpu` 可重现原全 GPU 路径。下一项扩展固定语料，并将已验证的语义生成与声学链衔接；当前没有新 ASR 或人工试听结论。
