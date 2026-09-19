# 十例准备条件下的完整生成

日期：2026-09-19。把十条回归输入接到自有 GPT、采样、SoVITS 和 PCM 输出，全部生成 token、历史、停止条件及语义切片与官方相同，最终波形通过原定 `atol=1e-4, rtol=1e-5`。这些结果尚未解决独立诊断中的三个采样概率值和一个 MRTE 中间值超差，完整数值验收仍未通过。

## 本轮验证范围

本轮使用既有官方文本特征、参考条件、实际语义采样噪声和另一组固定声学噪声。GPT 使用原参考语义前缀，并自行生成目标 token；声学计算只接收本次生成的目标语义，没有输入官方目标 token。CPU FP64 GPT Prefill、GPU FP32 Decode、CPU 声学 encoder、GPU flow/decoder 均沿用已记录路径，声码器使用已提交的逐残差对求值方式。

固定对照仍为官方提交 `48b1a0169a28582a8984402f82cf438d3bfa6aca`、朱雀院红叶 V2Pro、相同日文参考和 Top-p 1。两个无损存储模型包恢复后按原 FP32 计算。运行进程没有导入 Torch 或上游推理代码。

十例各执行首次请求和一次后续请求，共 20 次。逐次检查均通过，同一输入的两次浮点波形逐位相同。全部 1,805 个采样 token 与独立自有生成轮相同；十例最终浮点波形也逐位复现优化前的独立完整声学轮。声码器调度改变没有在这批整链结果中引入额外数值差异。

原始中日文对应的四个 WAV 与用户试听的 `20260919T123730.996242Z-native-prepared-speech` 文件逐字节相同。用户回复“四条都正常，未听出明显差异”，涵盖日文开头和 wa、中文“你好”、全文及音色。其余八例没有新增人工试听或 ASR 证据。

## 执行时间和资源

本轮用于扩展正确性，未预热，每例仅记录一次后续请求。以下时间是实际观察值，不是稳定性能基准，也不用于宣称优化加速。

| 样例 | 采样 token 数 | 不含尾静音的时长 | 后续请求 | RTF |
|---|---:|---:|---:|---:|
| 日文原始回归 | 121 | 4.80 s | 1.076 s | 0.224 |
| 中文原始回归 | 147 | 5.84 s | 1.293 s | 0.221 |
| 日文短句 | 24 | 0.92 s | 0.358 s | 0.389 |
| 中文短句 | 42 | 1.64 s | 0.506 s | 0.309 |
| 日文长句 | 374 | 14.92 s | 3.526 s | 0.236 |
| 中文长句 | 461 | 18.40 s | 4.038 s | 0.219 |
| 日文标点 | 212 | 8.44 s | 1.898 s | 0.225 |
| 中文标点 | 175 | 6.96 s | 1.513 s | 0.217 |
| 中英混合 | 132 | 5.24 s | 1.182 s | 0.226 |
| 日英混合 | 117 | 4.64 s | 1.063 s | 0.229 |

计时无诊断 observer 和中间捕获，包含共享噪声查找、最终 CPU 复制和 PCM 处理；不含文本前端、参考准备、常驻模型加载、文件写入或验证。首次请求也不等于应用冷启动。这里没有流式首包延迟测量。

本轮 GPT 常驻，MLX 缓存设置为 256 MiB。十例期间 allocator 高水位为 1,989.12 MiB，OS 进程生命周期最大 RSS 为 969.47 MiB；两种统计口径不同，不能相加或互相替代。模型释放后 MLX active/cache 都为 0。高水位从模型加载前开始累计，没有逐例重置，因此各请求后的该字段不是各例独立峰值。它是 Apple 统一内存的 MLX 分配器统计，不是 NVIDIA 显存或完整进程物理占用。

RSS 还包含预载的十例条件、采样随机数和官方对照数组，不能与只载两例的 Harness 直接比较产品内存收益。MLX 释放后，这些 CPU 输入及最后一次 PCM 等对象仍在进程内；active/cache 为零不表示进程内存清空。

此前两条短请求在按请求释放 GPT 等条件下测到的 622.05 MiB，不能推广到本轮长句或完整前端。具体资源与延迟取舍见 [声码器工作区实验](2026-09-19-decoder-workspace.md)。

## 保留的失败与证据

- [十例自有历史诊断](2026-09-19-expanded-native-generation.md)：token 和 logits 通过，但两例三个采样概率值超差。这轮没有概率 observer，最终输出通过不覆盖原失败。
- [十例声学诊断](2026-09-19-expanded-acoustic.md)：120 项中有一项 MRTE 中间结果超差，最终波形通过。其首层差异已定位到 [softmax](2026-09-19-attention-softmax-numerics.md)，候选仍需全链验证。
- 原始文本到音频、独立 RNG、其他 seed、其他模型版本和完整质量验收均未完成。

证据目录为 `SakuraTTS-References/runs/20260919T131148.832058Z-native-prepared-speech/`，包含源码快照及哈希、模型和输入身份、每次请求时间、全部目标 token / 历史 / 语义 / 波形和二十个 WAV，实际进程退出码为 0。

`verify-composed-v2.py` 与 `composition-verification-v2.json` 保存独立生成、完整声学和四个试听文件的比对。首版附加比较错误地要求本次输出等于“官方 decoder 输入”产生的独立 decoder 输出；两者输入不同，该要求不成立。原脚本及 mismatch 结果保留，新脚本逐例确认输入不同，只使用相同生成链的输出做逐位保持判断。

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/native_prepared_speech.py \
  --references "$REF" \
  --official-run "$REF/runs/20260919T125743.942661Z-official-mps" \
  --official-conditions "$REF/runs/20260919T122232.246358Z-sovits-fixed-official-mps" \
  --gpt-package "$REF/models/converted/20260919T123226.764557Z-20260919T104221.749176Z-010197bfc30b-gpt-fp32-lossless-storage" \
  --sovits-package "$REF/models/converted/20260919T123227.630582Z-20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32-lossless-storage" \
  --cases ja-reported-intro zh-reported-greeting ja-short zh-short ja-long zh-long \
          ja-punctuation zh-punctuation mixed-zh-en mixed-ja-en \
  --warmup 0 --repeat 1 --mlx-cache-limit-mib 256
```

运行后修正了 Harness 文档字符串，把含糊的“不使用参考语义”改为“不使用官方目标语义；保留参考语义条件”。执行代码未改，原运行快照保留。
