# 十条语料的自有历史生成对照

日期：2026-09-19。中日文用户样例、短句、长句、标点和中日英混合文本共十条，全部 1,805 步自行采样 token 与官方相同；返回历史、停止条件、索引和声学语义切片也相同。生成器每一步回填自己抽到的 token，没有用官方目标 token 推进计算。

整轮结果仍为 `generation_mismatch`。日文长句和中文标点共有 3 步、3 个概率元素超过既定容差；其他检查通过。本轮保留失败状态与原容差，不能写成十条全部通过。

## 官方随机数与输入身份

以固定官方提交 `48b1a0169a28582a8984402f82cf438d3bfa6aca`、朱雀院红叶 V2Pro、原参考音频和 seed 1234，重新运行全部十条语料。使用官方 MPS / FP32，中文 BERT 保留，不启用参考预计算、BERT 裁剪或声学资源释放。`ORT_DISABLE_TELEMETRY=1` 在导入 ORT 前设置，进程正常退出，退出码为 0。

官方环境为 PyTorch 2.7.1、NumPy 1.26.4、ONNX Runtime 1.30.0。仓库已有的 `polyphonic.md5`、日语 userdict 生成物状态与历史运行一致，没有改动上游源码。十条原始输入和 `cut0` 分句参数不变。

捕获官方每步实际生成的指数随机数、过滤后概率、原始 logits 和完整调用轨迹。捕获使用官方的 `empty_like(probs).exponential_(1)` 和 `argmax(probs / noise)` 表达式，增加的同步与 CPU 复制属于诊断开销。

新的十份 WAV 与之前 `104548.448908Z` 的八份和 `110311.350332Z` 的两份逐字节相同。额外对照现有十例声学金标的原 trace，以下内容也逐位相同：

- GPT 的 phones、lengths、参考 prompt、BERT、每步原始 logits、采样 token、返回历史；索引和采样参数相同。
- 声学的 semantic、phones、参考频谱、speaker embedding、原始浮点 waveform。
- 每例停止原因相同，全部 1,805 步的实际随机数和概率均已保存。

这使新捕获可以接到已有十例声学金标。WAV 字节核验没有新增试听结论；它确认捕获插桩没有改变这次固定运行的音频。

## 自有生成结果

自有路径使用当前无损紧凑 GPT 包、CPU FP64 Prefill、MLX FP32 Decode、NumPy FP32 采样，KV 容量 1024。Top-k 15、Top-p 1、temperature 1、repetition penalty 1.35。每步读取对应的官方实际随机数，随后使用自己的概率、采样结果和历史继续生成。

| 样例 | 采样步数 | 声学语义 token | 停止条件 | 概率检查 |
|---|---:|---:|---|---|
| `ja-reported-intro` | 121 | 120 | sampled / argmax EOS | 通过 |
| `zh-reported-greeting` | 147 | 146 | sampled / argmax EOS | 通过 |
| `ja-short` | 24 | 23 | sampled / argmax EOS | 通过 |
| `zh-short` | 42 | 41 | sampled / argmax EOS | 通过 |
| `ja-long` | 374 | 373 | sampled / argmax EOS | 第 274 步超差 |
| `zh-long` | 461 | 460 | sampled / argmax EOS | 通过 |
| `ja-punctuation` | 212 | 211 | sampled / argmax EOS | 通过 |
| `zh-punctuation` | 175 | 174 | sampled / argmax EOS | 第 95、97 步超差 |
| `mixed-zh-en` | 132 | 131 | 仅 sampled EOS | 通过 |
| `mixed-ja-en` | 117 | 116 | sampled / argmax EOS | 通过 |

步骤使用从零开始的索引。十例均未出现首次 token 分歧，未耗尽已保存的随机数；全部 1,805 步都在相同历史下比较，没有跳过任何不同历史步骤。每例撤回 EOS 后的历史、返回 idx、`[-idx:]` 语义切片和过滤集合完全一致。混合中英文本只有 sampled EOS 停止，另一项 EOS 条件也与官方一致。

全部 logits 满足原 `atol=1e-4, rtol=1e-5`，最大绝对误差为 `8.48770e-5`，出现在日文标点样例。概率采用更严格的 `atol=1e-6, rtol=1e-5`；logits 通过不能代替概率通过。

## 三处概率超差的来源

用保存的 logits 做离线对照：一条路径将官方原始 logits 输入当前 NumPy 采样器，另一条路径输入自有 GPT 的原始 logits；两者使用相同历史和参数。前一条路径全部 1,805 步概率通过原容差，最大绝对误差 `2.38419e-7`。后一条复现原运行的三处失败。

| 样例、步骤、词表索引 | 自有概率与官方之差 | 该元素允许误差 | 同一官方 logits 下的 NumPy 采样误差 |
|---|---:|---:|---:|
| `ja-long`，274，857 | `3.59118e-6` | `2.45403e-6` | `2.98023e-8` |
| `zh-punctuation`，95，327 | `1.780689e-6` | `1.780488e-6` | `0` |
| `zh-punctuation`，97，801 | `2.01911e-6` | `1.82865e-6` | `1.49012e-8` |

这说明当前三处差异主要来自 GPT logits 的误差传播；相同 logits 下，NumPy 采样计算没有重现这些失败。中文标点第 95 步非常接近阈值，仍按失败记录。没有针对样例补丁、改输入或放宽容差。

这项归因尚未确定 GPT 中具体哪个算子造成这些差异，也不能据此承诺其他随机数下 token 一定不分歧。后续应沿失败步骤追踪 Prefill/KV/Decode 的误差来源，保留当前高精度对照；不需要用修改停止规则或采样阈值来消除失败记录。

## 验证范围与证据

自有进程没有导入 PyTorch，释放后 MLX active/cache 均为零。本轮是逐步诊断，没有测量正常请求速度或分配器峰值，也没有执行自有声学生成、ASR 或新试听。它使用共享的官方实际随机数，不能证明独立 NumPy RNG、其他 seed、Top-p 小于 1、无参考或流式功能的兼容性。

路径均相对于 `/Users/beyondpower/Documents/Projects/SakuraTTS-References/`：

- `runs/20260919T125743.942661Z-official-mps/`：十例实际 draws、概率、官方 trace 和 WAV，完整原始输出与 `process-result.json`；`capture-condition-equivalence.json` 及 `compare-capture-conditions.py` 保存输入和中间结果逐位检查。
- `runs/20260919T130013.344711Z-audio-run-comparison/`：十份新旧 WAV 字节与请求身份核验。
- `runs/20260919T130029.229727Z-native-gpt-generation/`：自有历史、token、语义切片、逐步 logits 和全部检查；`probability-attribution.json` 及 `diagnose-probabilities.py` 保存三处差异的离线归因。

各运行保存实际命令、来源和模型包哈希、源码副本。复现时先产生新的官方目录，再替换第二条命令的 `--official-run`：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
ORT_DISABLE_TELEMETRY=1 "$REF/.venv-official-macos/bin/python" -u research/tools/run_reference_process.py \
  --references "$REF" --backend official --device mps \
  --diagnostic --capture-sampling-noise --repeat 1 \
  --case-ids ja-reported-intro zh-reported-greeting ja-short zh-short \
  ja-long zh-long ja-punctuation zh-punctuation mixed-zh-en mixed-ja-en

"$REF/.venv-mlx-macos/bin/python" research/tools/native_gpt_generation.py \
  --references "$REF" \
  --official-run "$REF/runs/20260919T125743.942661Z-official-mps" \
  --package "$REF/models/converted/20260919T123226.764557Z-20260919T104221.749176Z-010197bfc30b-gpt-fp32-lossless-storage" \
  --cases ja-reported-intro zh-reported-greeting ja-short zh-short \
  ja-long zh-long ja-punctuation zh-punctuation mixed-zh-en mixed-ja-en
```

第二条在当前实现下以 `generation_mismatch`、退出码 1 结束，完整保存十例结果，不隐藏上述概率失败。
