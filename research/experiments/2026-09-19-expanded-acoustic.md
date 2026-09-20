# 十条固定条件的声学回归

日期：2026-09-19。此前完整 MLX 声学链路只比较两条用户报告样例。这里扩大到短句、长句、带标点的多句文本和中英/日英混合文本，观察误差是否随输入和序列长度变化。仍使用“朱雀院红叶”V2Pro、FP32、`speed=1`、`noise_scale=0.5`。

## 输入与官方条件

复用已经保存的官方轨迹：

- `runs/20260919T104548.448908Z-official-mps/`：两条报告样例、两条短句、两条长句、两条标点样例。
- `runs/20260919T110311.350332Z-official-mps/`：中英混合、日英混合。

第一批来源 run 的总状态是 `failed`，原因是后续执行英文前端时缺少 `wordsegment`。前八条已有完整的单条完成记录、轨迹和音频。本次只读取这八条完成记录，保留来源状态，没有把原 run 改成成功；混合文本使用补齐依赖后另存的两条完成轨迹。

`research/tools/sovits_fixed_conditions.py` 现在支持重复传入 `--official-trace-run`，用 `--cases` 指定样例 ID。旧 `--languages` 参数和默认 ja/zh 入口仍可用。执行前核对固定官方提交、模型版本、原模型/参考文件哈希及采样参数，要求每个 ID 恰好匹配一个已完成的轨迹行，再读取原始 semantic、phones、参考频谱和 speaker embedding。十例参考频谱和 speaker embedding 也逐位一致。

这个入口复用原来的模型加载、诊断插桩和执行逻辑，没有复制另一套声学生成实现。输入文本和原轨迹中的 codes/phones 均未改写，没有重新执行文本前端或 GPT。每例用私有 CPU FP32 随机数发生器，按该例 latent 形状重新以 seed `20260919` 产生噪声，并把实际噪声数组保存下来；它与原始自由生成轨迹中未捕获的噪声不同。

## 官方基准结果

新官方条件位于 `runs/20260919T122232.246358Z-sovits-fixed-official-mps/`。进程退出码 0，十例全部完成。每例保存 14 个中间阶段及完整 waveform，共 15 项，另有 semantic/phones 输入、来源波形和 32 kHz Float32 WAV。没有裁剪、响度归一化或额外拼接静音。

| 样例 ID | 语义 token | phones 数 | 完整波形时长 |
|---|---:|---:|---:|
| ja-reported-intro | 120 | 58 | 4.80 秒 |
| zh-reported-greeting | 146 | 56 | 5.84 秒 |
| ja-short | 23 | 11 | 0.92 秒 |
| zh-short | 41 | 6 | 1.64 秒 |
| ja-long | 373 | 267 | 14.92 秒 |
| zh-long | 460 | 200 | 18.40 秒 |
| ja-punctuation | 211 | 117 | 8.44 秒 |
| zh-punctuation | 174 | 57 | 6.96 秒 |
| mixed-zh-en | 131 | 58 | 5.24 秒 |
| mixed-ja-en | 116 | 63 | 4.64 秒 |

离线复核确认所有数组有限、15 项阶段均存在，源码快照、NPZ 和 WAV 的 SHA-256 均正确。十例的 `ge/ge_projected` 逐位相同。两条报告样例的全部 15 项与原官方固定条件 `runs/20260919T111545.846735Z-sovits-fixed-official-mps/` 逐位一致，验证了入口扩展没有改变原有执行结果。复核记录保存为新 run 内的 `verification.json`。

这些证据说明官方固定条件已准备完整。生成成功、波形长度和阶段数不代表内容、发音或音色已经验收；本轮没有 ASR 或人工试听。

## MLX 扩展回归保留一次阶段失败

使用原始 FP32 转换包 `models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32/`，明确选择 CPU encoder 和 GPU flow/decoder，在 `runs/20260919T122652.881973Z-mlx-sovits-complete-gpu/` 中比较十例各 12 个阶段。预设容差仍为 `atol=1e-4, rtol=1e-5`。

最终波形十例全部通过，最大绝对误差如下：

| 样例 ID | 波形最大绝对误差 | 波形超差元素 |
|---|---:|---:|
| ja-reported-intro | 7.40141e-5 | 0 |
| zh-reported-greeting | 4.92409e-5 | 0 |
| ja-short | 7.85775e-5 | 0 |
| zh-short | 9.52184e-6 | 0 |
| ja-long | 9.73158e-5 | 0 |
| zh-long | 6.43805e-5 | 0 |
| ja-punctuation | 6.94394e-5 | 0 |
| zh-punctuation | 3.75807e-5 | 0 |
| mixed-zh-en | 6.61463e-5 | 0 |
| mixed-ja-en | 4.75785e-5 | 0 |

但全部 120 个阶段比较中只有 119 个通过。`ja-punctuation` 的 MRTE 数组在 NCT 坐标 `[0,21,125]` 有一个元素超差：官方为 `2.363581895828247`，MLX 为 `2.363433837890625`，误差为 `-1.48057938e-4`，该点上限为 `1.23635819e-4`。MRTE 整体 RMS 误差为 `8.03898e-6`，其后的隐藏层、latent 和最终波形均通过。

因此整轮保留 `numerical_mismatch` 和退出码 1。不能用最终波形通过来覆盖中间阶段失败，也不能把此前两例通过扩大为十例全阶段通过。下一轮需要进一步区分 MRTE 输入误差和交叉注意力/投影本身的累积误差，不调整阈值或替换某个坐标。

## 无损存储独立复核

随后仅把包替换为 `models/converted/20260919T122406.407310Z-20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32-lossless-storage/`，继续使用相同执行路径和十例官方条件。该包在加载时恢复原 FP32 权重。结果位于 `runs/20260919T122739.465235Z-mlx-sovits-complete-gpu/`。

新旧两包生成的 120 个 native 阶段数组全部逐位相同，十个 PCM WAV 的 SHA-256 也相同。MRTE 的同一元素仍然超差，run 同样保留 `numerical_mismatch` 和退出码 1。这证明本轮存储格式没有引入额外计算差异，不能把它解释为算法失败已经消失。逐项比较保存为新 run 中的 `storage-equivalence.json`。

两轮运行的源码快照和数组哈希均已核验。两条原报告样例的数值结果也与前一轮显式 CPU encoder 候选保持一致。运行计数用于诊断；本轮没有正常延迟、资源收益、ASR 或人工试听结论。

复现官方固定条件：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-official-macos/bin/python" research/tools/sovits_fixed_conditions.py \
  --references "$REF" \
  --official-trace-run "$REF/runs/20260919T104548.448908Z-official-mps" \
  --official-trace-run "$REF/runs/20260919T110311.350332Z-official-mps" \
  --backend official --device mps --threads 4 \
  --cases ja-reported-intro zh-reported-greeting ja-short zh-short \
    ja-long zh-long ja-punctuation zh-punctuation mixed-zh-en mixed-ja-en
```

追加 `--check-inputs` 可只验证轨迹输入，不加载 PyTorch 或模型。MPS 运行中的各阶段复制会增加同步和分配开销，本实验不用于正常性能比较，统一内存结果也不能作为 NVIDIA 显存指标。

复现 MLX 十例对照：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" research/tools/mlx_sovits_replay.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T122232.246358Z-sovits-fixed-official-mps" \
  --device gpu --encoder-device cpu
```

该命令预期退出码为 1，因为包含上述 MRTE 阶段失败。将 `--package` 改为无损存储包可复现第二轮等价结果。
