# 自有采样规则的 CPU 对照

日期：2026-09-19。状态：真实历史的采样对照通过；3 项 Top-p 边界仍不兼容。随后已完成两条 Top-p 1 请求的[自有历史生成对照](2026-09-19-native-gpt-generation.md)，使用官方实际随机数，尚未验证独立 RNG 或完整语音。

新增 [NumPy FP32 采样模块](../../src/sakuratts/_internal/sampling.py)，其运行依赖只有 NumPy 和 Python 标准库。它用于核对官方概率处理、指数噪声抽样和非流式停止规则，不是 GPU 采样优化。单独导入模块已确认没有加载 PyTorch；开发 Harness 使用 PyTorch 执行官方对照。

## 已验证的范围

固定官方 GPT-SoVITS 提交为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`。对照来自 `AR/models/utils.py` 的 `logits_to_probs`、`sample`，以及 `t2s_model.py` 的 `infer_panel_naive` 非流式路径。

使用“朱雀院红叶”V2Pro 的中日文真实 trace，共 268 个生成步骤；另取其中 8 个步骤改变 Top-p、Top-k 和 temperature，检查规则组合。两组均保持各步骤之前的官方历史不变。概率的最大绝对差分别为 `2.384185791015625e-7` 和 `2.9802322387695312e-8`；过滤集合、同噪声抽样 token、重复惩罚后的原 logits、停止状态与历史均一致。两条真实 trace 的最终 history 和返回 idx 也与原始保存结果一致。

概率比较预先使用 `rtol=1e-5`、`atol=1e-6`。过滤集合、抽样 token、停止后的历史和语义切片要求完全相同；概率容差不能用于放过这些离散差异。

5 项回归测试已通过，覆盖重复 token 的惩罚次数、惩罚对 EOS argmax 的影响、Top-p 与 temperature 的顺序、Top-k 边界并列、前 11 步 EOS 排除和 early-stop 切片。另有 16 个合成边界进入官方对照，其中 3 个暴露出未解决的兼容缺口。

## 必须保留的官方行为

重复惩罚对负值乘 penalty，对非负值除 penalty。同一历史 token 出现多次时，只根据原值处罚一次。这个步骤原地修改传入 logits；后面的 Top-p、temperature 和 Top-k 不继续原地修改它。因此停止检查读取的是“重复惩罚后、过滤前”的 argmax，而不是最终概率的 argmax。

Top-p 先于 temperature 和 Top-k。官方这里移除累计概率超过阈值的 token，仅强制保留排序后的第一个 token；它没有采用另一个常见 Top-p 实现中“保留第一个跨过阈值 token”的右移步骤。Top-k 按第 k 大值作为阈值，等于阈值的 token 都保留，实际数量可以多于 k。

`infer_panel_naive` 在 idx 0 至 10 共 11 步移除最后的 EOS 列。它先追加 sampled token，再检查生成长度是否严格大于 `early_stop_num`；若 sampled token 或惩罚后的 argmax 是 EOS，会撤回刚追加的 token，即使被抽到的 token 本身不是 EOS。idx 1499 也触发停止。

本轮保留了这些容易出现边界偏差的行为，没有替官方纠正：

- 仅由长度上限停止时，生成长度可能是 `early_stop_num + 1`，返回的 idx 仍是从 0 开始的循环索引。
- 调用者使用 `[-idx:]`；仅因长度停止时，它可以丢掉本轮生成的第一个 token。
- idx 为 0 时，`[-0:]` 会选择全部历史，包括已有参考前缀。
- 无参考语义的非流式路径第一次 yield 返回 idx 0；本原型只表达调用者取第一个结果的行为，没有扩展为无参考推理已兼容。

Harness 从固定官方源码的 AST 提取了追加 token、early stop、EOS 撤回、循环上限和空输出回填的实际代码，保存为 `official_stop_oracle.py`。只移除了不可执行到的流式分支，使其能够作为单步 CPU 对照调用；没有另外手写一份“应当如此”的停止规则作为唯一标准。

## 仍然存在的 3 项兼容缺口

| 合成样例 | NumPy 结果 | 官方 CPU 结果 | 影响 |
|---|---|---|---|
| 33 个分数按 `i % 3` 交错并列，Top-p 0.3 | 保留 `[2, 5, 8, 11]` | 保留 `[2, 11, 26, 29]` | 同一份指数噪声抽样为 2 与 26，后续历史随之不同 |
| 1025 个分数全为 0，Top-p 0.3 | 保留 0 至 306 | 保留 1 至 306，以及 512 | 本次同噪声 token 恰好相同，但候选集合和概率不同 |
| 分数 `[4, 3, 2, 1, 0]`，Top-p `0.87053025` | 保留 `[0]` | 保留 `[0, 1]` | FP32 softmax / 累计求和的舍入差异改变了离散过滤决定 |

前两项来自排序策略：本原型使用 NumPy 稳定排序，固定按 token 顺序处理并列；官方 `torch.sort` 没有要求稳定排序。少量全零分数恰好排序相同，不能证明更大词表或交错并列也兼容。

第三项即使没有并列分数也会发生。此处最大的概率差约为 0.26894，已经是候选集合改变，不能通过放宽数值容差解释为通过。最终结果状态为 `completed_with_compatibility_gaps`，没有把 268 个真实步骤通过扩大为全部采样参数兼容。

对照程序在存在兼容缺口时现在返回退出码 1。复跑结果 `runs/20260919T111611.543276Z-sampling-equivalence-cpu/` 仍是上述 3 项缺口、268 个真实步骤和 8 个参数变体通过，避免自动化把执行完成误读为兼容通过。

## 随机数与验证边界

Harness 用 `numpy.random.default_rng(20260919)` 生成正指数噪声，把同一份数组分别交给两端的 `argmax(probabilities / noise)`。真实 trace 的原始 sampled token 另外用于停止回放，和本次共享噪声下的新抽样 token 分开记录。

这验证的是相同 logits、历史、参数与噪声下的计算。没有声称 NumPy 和 PyTorch 使用相同 seed 会得到相同随机数流，也没有验证自由生成轨迹、波形、ASR、人工听感或音色。所有对照在 CPU 上执行，没有新增 GPU 性能结论。

## 原始证据与复现

本机结果目录：`SakuraTTS-References/runs/20260919T110945.813163Z-sampling-equivalence-cpu/`。

目录中的 `result.json` 保存每项检查、阈值、版本、源码与 trace 哈希和实际命令；各 NPZ 保留原 logits、历史、指数噪声、两端概率、过滤集合、惩罚后 logits、抽样 token、停止后历史及官方后缀切片。执行代码与提取的官方停止代码也已保存。首轮较小测试集的结果位于 `20260919T110806.732183Z-sampling-equivalence-cpu/`，没有覆盖后来发现差异的 3 个样例，保留它用于说明测试覆盖的变化。

从项目根目录执行：

```sh
TTS_REF_DIR="/Users/beyondpower/Documents/Projects/SakuraTTS-References"
"$TTS_REF_DIR/.venv-official-macos/bin/python" -u research/tools/sampling_equivalence.py \
  --references "$TTS_REF_DIR" \
  --trace "$TTS_REF_DIR/runs/20260919T102853.549769Z-official-mps/ja-1-trace.json" \
  --trace "$TTS_REF_DIR/runs/20260919T102853.549769Z-official-mps/zh-1-trace.json"

"$TTS_REF_DIR/.venv-official-macos/bin/python" -m unittest discover \
  -s tests -p 'test_sampling.py' -v
```

下一轮接入完整解码前，需要对明确支持的采样参数范围处理上述 Top-p 边界，并继续保留原有高精度对照。当前固定语音样例使用 Top-p 1.0，跳过核采样，因此这些边界没有在本轮真实历史中触发。
