# 自有语义生成接完整声学计算

日期：2026-09-19。两条原始用户回归输入，在已有官方文本和参考条件下，已通过自有 GPT、采样、停止、语义切片、SoVITS 到 PCM 的整条计算路径。运行没有导入 PyTorch 或上游推理代码，GPT 未使用参考目标 token 推进生成，SoVITS 接收的是本次生成器的实际输出。

这次验证没有重新执行原始文本处理或参考音频准备。中文 BERT、参考语义、音色条件来自已保存的官方结果；GPT 使用官方实际采样的随机数，声学使用另一组固定显式噪声。它证明模块已接通，但不代表独立 RNG、原始输入端到端或其他版本已兼容。

## 输入和执行边界

Harness 为 `harness/native_prepared_speech.py`。加载时检查 GPT、SoVITS 的原模型哈希与固定官方提交；对比两个来源轨迹的 semantic、phones、参考频谱和 speaker embedding，避免把不同参考条件的声学金标拼进来。原始输入句子没有修改。

- GPT：`MLXGPT.load(..., prefill_precision="fp64")`，CPU FP64 Prefill、GPU FP32 Decode，KV 容量 1024。
- 生成：`generate_semantic()` 不传 observer，逐步使用自己的生成历史；Top-p 1，其他采样参数与来源相同。
- 声学：CPU encoder、GPU flow/decoder，`speed=1, noise_scale=0.5`；不传 capture。
- 输出：复制完整波形到 CPU，沿用固定官方单片段的幅度归一化、尾部 0.3 秒静音和 int16 转换。不裁剪开头。

官方声学金标的噪声是在固定条件实验中另存的，不是用户此前听过的原始官方自由生成波形。这里的 WAV 因此单独命名为 `official-fixed` 和 `native`，没有沿用此前试听结论。

## 原 FP32 包的验证结果

运行目录：`runs/20260919T123730.996242Z-native-prepared-speech/`。

两例各执行首次请求、2 次热身和 5 次计时。每次 token、返回历史、停止条件、索引、声学 semantic 都与官方一致，同一输入的重复浮点波形逐位相同。完整波形按原 `atol=1e-4, rtol=1e-5` 检查：

| 样例 | 采样 token / 声学 token | 无尾静音波形时长 | 波形最大绝对误差 | 超差元素 |
|---|---:|---:|---:|---:|
| 日文 | 121 / 120 | 4.80 秒 | 7.40141e-5 | 0 |
| 中文 | 147 / 146 | 5.84 秒 | 4.92409e-5 | 0 |

最终波形与已单独验证的自有声学链数值结果相同。这里没有再次保存逐层诊断，扩展语料仍受[日文标点 MRTE 超差](2026-09-19-expanded-acoustic.md)限制，不能把两例整链结果推广为十例全阶段通过。

## 正常路径耗时与资源

计时期间没有逐层捕获、logits 比较或观察回调。包括语义循环中的 CPU/GPU 同步、读取已载入的随机数、完整声学求值、CPU 波形复制和 PCM 转换；不包括输入准备、模型加载、结果比较或写文件。不是完整 TTS 延迟或流式首包时间。GPU 没有同时运行其他实验。

| 样例 | 语义生成中位 | 声学生成中位 | 准备条件请求中位 | RTF |
|---|---:|---:|---:|---:|
| 日文 | 0.832 秒 | 0.206 秒 | 1.039 秒 | 0.216 |
| 中文 | 0.966 秒 | 0.256 秒 | 1.229 秒 | 0.210 |

各列独立取中位数，因此不要求列值精确相加。RTF 使用无尾静音的波形时长。加载两个模型耗时 0.404 秒；日文第一次请求 1.161 秒，中文在日文运行后第一次请求 1.268 秒。它们没有覆盖 Python 启动或冷文件系统缓存，不能标为完整冷启动。

加载后 MLX active 为 469.05 MiB，请求后为 565.05 MiB。整个模型加载和请求阶段的 MLX allocator peak 为 1452.41 MiB；中文请求后 cache 为 2674.44 MiB。进程生命周期 RSS 最大值为 767.94 MiB，包含 Python、输入和对照数组。这些计数的涵盖范围不同，不能互相代替、相加或当作 NVIDIA 显存。

释放两个模型、GC 并执行 `mx.clear_cache()` 后，MLX active 和 cache 均为零。这是本轮单生命周期结果，多轮切换或长期泄漏还需验证。

## 质量与复现

用户对本轮两条 native 与两条 official-fixed 样音回复：“四条都正常，未听出明显差异”。提问明确包含日文开头和 wa、中文“你好”、全文和音色；这四份文件的对应检查记为用户通过。原回复和文件 SHA-256 保存于 `runs/20260919T124243.798790Z-native-user-listening-review/`。这不是 MOS 评分，其他样例与模型不继承结果。

随后使用固定的本地 Whisper small 对四条文件进行无文本提示转写，官方与自有版本在每种语言上得到相同文字。日文完整识别出开头；中文均识别为“您好,歡迎使用音樂語音。現在就在測試蘋果電腦上的語音合成。”，与目标文本仍有差异。保留 ASR 原文，不用用户反馈改写自动识别结果。证据为 `runs/20260919T124306.646465Z-asr-review/`，详见 [ASR 记录](2026-09-19-asr-review.md)。

路径均相对于 `/Users/beyondpower/Documents/Projects/SakuraTTS-References/`。运行目录保存完整源码副本、来源和模型包哈希、每次耗时、生成 token/semantic/波形数组、PCM WAV 及退出状态。

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/native_prepared_speech.py \
  --references "$REF" \
  --official-run "$REF/runs/20260919T114052.042551Z-official-mps" \
  --official-conditions "$REF/runs/20260919T111545.846735Z-sovits-fixed-official-mps" \
  --gpt-package "$REF/models/converted/20260919T104221.749176Z-010197bfc30b-gpt-fp32" \
  --sovits-package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32"
```

默认热身 2 次、计时 5 次，可显式配置 MLX cache limit。

## 紧凑包、权重复用与缓存限制

后续把两个模型改为无损紧凑包，并使用已验证的 [Prefill 权重复用](2026-09-19-gpt-prefill-weight-reuse.md)。运行 `20260919T124134.295402Z-native-prepared-speech` 的 token、history、semantic、完整浮点波形和两条 WAV 均与原 FP32 包路径逐位相同，核验保存在该目录的 `source-and-storage-equivalence.json`。新包中附件和原始权重来源均保留。

再显式设置 `--mlx-cache-limit-mib 256`，运行 `20260919T124221.025101Z-native-prepared-speech`；同样的十项数组/文件比较全部通过，见 `cache-policy-equivalence.json`。三轮输出的实际文件一致，因此对应的是用户刚确认过的同一份音频字节；没有把不同输出推断为已试听。

| 配置 | 日文请求中位 | 中文请求中位 | 中文结束后 active / cache | MLX allocator peak | 进程生命周期 RSS 最大值 |
|---|---:|---:|---:|---:|---:|
| 原 FP32 包，旧 Prefill 读取 | 1.039 秒 | 1.229 秒 | 565.05 / 2674.44 MiB | 1452.41 MiB | 767.94 MiB |
| 紧凑包，复用已加载权重 | 1.053 秒 | 1.288 秒 | 565.05 / 2674.54 MiB | 1452.41 MiB | 760.78 MiB |
| 同上，cache limit 256 MiB | 1.100 秒 | 1.329 秒 | 565.05 / 258.26 MiB | 1452.34 MiB | 707.31 MiB |

缓存限制把中文结束后的 cache 减少约 90.3%，两条完整准备条件请求分别慢约 4.5% 和 3.2%；active 和 allocator peak 基本不变。观察到的 cache 仍大于 256 MiB，配置不是严格峰值上限。三个进程释放后 active/cache 均为零。

GPT 子模块的重复读盘优化已经单独测得收益，但这两次整链测量没有看到相对原 FP32 包的总耗时改善。这里不把子模块约 17% 的收益扩大到完整请求，也不把更低 RSS 解释为活动声学工作区缩小。下一项是测量语义结束后提前释放 GPT/KV 能否降低声学期间的峰值，以及每次重载所需的延迟。
