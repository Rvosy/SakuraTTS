# 语义生成后提前释放 GPT

日期：2026-09-19。两条原始用户回归输入在相同模型、条件和随机数下，把 GPT 改为按请求加载、语义完成后立即释放，MLX allocator peak 从 1452.41 MiB 降到 1052.45 MiB，减少约 400 MiB（27.54%）。每次请求增加约 0.25–0.32 秒。保留为 Harness 的显式实验选项，默认仍常驻。

## 改动与计时

`harness/native_prepared_speech.py` 新增 `--gpt-lifecycle resident|request`，默认 `resident`。`request` 模式只让 SoVITS 常驻：

1. 请求计时开始后加载 GPT，仍使用容量 1024 的 KV、CPU FP64 Prefill 和 GPU FP32 Decode。
2. 生成本次语义后，同步、释放 GPT 和 KV，执行 GC、`mx.clear_cache()` 并同步。
3. 把本次语义传给原有声学路径，完成波形复制和 PCM 转换。

加载和释放分别记录为 `gpt_load_seconds`、`gpt_release_seconds`，都计入 `prepared_request_seconds`。常驻模式这两个字段为零。声学计时从 GPT 释放结束后开始；清理缓存也可能影响声学耗时，因此该选项评估的是整个生命周期策略，不能把差异只归给权重驻留。

另加 `--cases`，默认仍为 `ja zh`，可对已有官方轨迹和声学条件按名称选择样例。来源哈希检查和默认样例不变。

计时没有 observer、逐层 capture 或内存轮询。每例依次执行首次请求、2 次热身和 5 次测量，验证在每次请求计时之外进行。包括按请求 GPT 权重读盘、验证、展开及释放，不包括文本前端、参考准备、SoVITS 初次加载、文件输出或 Python 启动。两轮均独占本机计算窗口。

## 正确性

使用无损紧凑模型包，与常驻基线 `20260919T124134.295402Z-native-prepared-speech` 对照。两个新运行中，全部 32 次请求的 token、history、返回索引、停止原因和 acoustic semantic 均通过原官方对照；同一输入重复波形一致。

离线进一步按 dtype、shape 和数组原始字节检查两个运行的最终 sampled tokens、history、semantic、完整浮点 waveform，均与常驻基线逐位相同。两种语言的 native 和 official-fixed WAV 文件 SHA-256 也分别相同。模型包、来源数据和所有运行时源码哈希一致，只有 Harness 增加生命周期参数。

完整波形仍使用原 `atol=1e-4, rtol=1e-5`，没有放宽容差。日文、中文最大绝对误差分别为 `7.40141e-5`、`4.92409e-5`，超差元素均为零。每轮仍未导入 PyTorch 或上游推理模块。

本轮没有新做 ASR 或试听。由于 WAV 文件逐字节相同，生成的是[已由用户确认的同一份样音](2026-09-19-native-prepared-speech.md)，不据此扩展到其他文本或声学阶段。现有扩展语料的 MRTE 超差仍由独立实验跟踪。

## 资源与速度

表中内存均来自 Apple 统一内存上的 MLX allocator 计数；active 和 cache 是执行边界快照，peak 是加载至全部请求结束期间的分配器高水位。它们不等于完整进程内存或 NVIDIA 独立显存。

| 配置 | 加载后 active | 中文结束后 active / cache | MLX allocator peak | 进程生命周期 RSS 最大值 |
|---|---:|---:|---:|---:|
| GPT 常驻，默认缓存 | 469.05 MiB | 565.05 / 2674.53 MiB | 1452.41 MiB | 760.78 MiB |
| GPT 常驻，cache limit 256 MiB | 469.05 MiB | 565.05 / 258.26 MiB | 1452.34 MiB | 707.31 MiB |
| 按请求加载并提前释放，默认缓存 | 165.15 MiB | 165.15 / 1584.20 MiB | 1052.45 MiB | 701.00 MiB |
| 按请求加载并提前释放，cache limit 256 MiB | 165.15 MiB | 165.15 / 257.16 MiB | 1052.45 MiB | 661.92 MiB |

GPT 常驻基线请求后 active 比加载后多约 96 MiB，这是实际生成时求值的 KV。新模式语义结束后同时释放 GPT 权重和 KV，请求后只剩 165.15 MiB 声学模型。所有进程最终释放模型、GC、清缓存后，active/cache 均为零。

缓存限制进一步降低请求后的 cache，峰值基本不变。日文请求后的 cache 可达 265.36 MiB，因此 256 MiB 是配置值，不能描述成严格的总内存上限。

| 配置 | 日文请求中位 | 中文请求中位 | 日文 / 中文 RTF |
|---|---:|---:|---:|
| GPT 常驻，默认缓存 | 1.053 秒 | 1.288 秒 | 0.219 / 0.220 |
| GPT 常驻，cache limit 256 MiB | 1.100 秒 | 1.329 秒 | 0.229 / 0.228 |
| 按请求加载并提前释放，默认缓存 | 1.371 秒 | 1.542 秒 | 0.286 / 0.264 |
| 按请求加载并提前释放，cache limit 256 MiB | 1.344 秒 | 1.512 秒 | 0.280 / 0.259 |

按请求模式在默认缓存下比常驻基线分别慢 30.15% 和 19.77%。日文、中文 GPT 加载中位分别为 0.2695、0.2683 秒，释放为 0.0206、0.0198 秒；声学为 0.2367、0.2778 秒。RTF 分母是无尾静音音频的 4.80 秒、5.84 秒，尚不包含文本或参考准备。

限制缓存后的 GPT 加载中位为 0.2641、0.2623 秒，释放为 0.0086、0.0085 秒，声学为 0.2466、0.2987 秒。这轮完整请求略快，但声学部分变慢且轮次只有各 5 次，不能据此宣称缓存限制提高了速度。内存收益和每次重载的延迟成本已有直接证据，默认策略暂不改变。

两个新运行只加载 SoVITS 的初始耗时分别为 0.2210、0.2195 秒。默认缓存下日文首请求为 1.4837 秒，中文在日文之后的首请求为 1.5677 秒；不作为完整冷启动指标。后续可以调查保持 SoVITS 常驻时，减少 GPT 权重重复验证和加载的成本，但必须保留完整模型能力与同条件输出对照。

## 证据与复现

路径相对于 `/Users/beyondpower/Documents/Projects/SakuraTTS-References/`：

- 常驻默认缓存：`runs/20260919T124134.295402Z-native-prepared-speech/`。
- 常驻限制缓存：`runs/20260919T124221.025101Z-native-prepared-speech/`。
- 按请求默认缓存：`runs/20260919T124916.761363Z-native-prepared-speech/`。
- 按请求限制缓存：`runs/20260919T125018.071612Z-native-prepared-speech/`，另存 `compare-lifecycle.py` 和 `lifecycle-equivalence.json`，记录比较命令、来源哈希与每项逐位核验。

每个运行保留完整源码副本、输入和模型包哈希、逐次耗时、资源计数、生成数组与 WAV。

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/native_prepared_speech.py \
  --references "$REF" \
  --official-run "$REF/runs/20260919T114052.042551Z-official-mps" \
  --official-conditions "$REF/runs/20260919T111545.846735Z-sovits-fixed-official-mps" \
  --gpt-package "$REF/models/converted/20260919T123226.764557Z-20260919T104221.749176Z-010197bfc30b-gpt-fp32-lossless-storage" \
  --sovits-package "$REF/models/converted/20260919T123227.630582Z-20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32-lossless-storage" \
  --gpt-lifecycle request
```

第二轮追加 `--mlx-cache-limit-mib 256`。离线逐位比较：

```sh
"$REF/.venv-mlx-macos/bin/python" \
  "$REF/runs/20260919T125018.071612Z-native-prepared-speech/compare-lifecycle.py"
```
