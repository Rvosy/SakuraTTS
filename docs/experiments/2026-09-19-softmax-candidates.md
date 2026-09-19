# softmax 精度候选与十例回归

日期：2026-09-19。承接 [首层 attention 分解](2026-09-19-attention-softmax-numerics.md)，本轮对保存的同一份 `masked_scores` 比较四条 softmax 路径，再把有局部改善的候选统一应用到完整声学编码器。MLX FP64 累积通过完整回归与成本检查后，已作为显式 CPU 选项迁入；默认 FP32 对照保留，容差仍为 `atol=1e-4, rtol=1e-5`。

## 先核对实现含义

当前官方环境使用 PyTorch 2.7.1，构建对应提交 `e2d141dbde55c2a4370fac5165b0561b6af4798b`。其 [MPS softmax 实现](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/mps/operations/SoftMax.mm) 直接调用 `MPSGraph softMaxWithTensor`，该文件没有给出内部指数算法或归约顺序，不能用 PyTorch CPU 实现推断 MPS 的计算细节。

MLX 0.32.2 的文档及 [CPU softmax 源码](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/cpu/softmax.cpp) 表明，`precise=True` 只将 FP16/BF16 的累积提高到 FP32。对当前 FP32 输入，开关两侧走同一路径。

另一个需要区分的细节是：把 MLX softmax 输入转成 FP64，并不代表其中每项计算都使用 FP64。当前安装包的 `mlx/backend/cpu/simd/math.h` 将非复数向量化 `exp` 的输入先转为 `Simd<float,N>`，再执行 FP32 多项式近似；softmax 的归约和缩放可以使用 FP64，尾部标量则使用 `std::exp`。下文将其称为 **MLX FP64 累积候选**。NumPy 候选才在 FP64 中完成减最大值、指数、求和和相除，随后一次转回 FP32。

这些源码与安装包头文件的只读副本、SHA-256 清单保存在参考目录的 `research/20260919-softmax-source/`。

## 相同分数上的局部结果

`harness/softmax_candidates.py` 不加载模型，直接读取官方首层的 `[1,2,422,422]` 分数。四条路径均使用相同输入，没有改变 mask 或筛选元素。

| 路径 | 与官方概率的最大绝对误差 | RMS 误差 | 概率行和偏离 1 的最大值 |
|---|---:|---:|---:|
| 原 MLX FP32，precise=True | 2.98023224e-7 | 2.67503134e-9 | 4.42858436e-7 |
| MLX FP32，precise=False | 2.98023224e-7 | 2.67503134e-9 | 4.42858436e-7 |
| MLX FP64 累积 | 1.19209290e-7 | 1.26243985e-9 | 4.53899816e-8 |
| NumPy FP64 | 1.19209290e-7 | 1.29245461e-9 | 3.37410029e-8 |

`precise` 开关两侧逐位相同，符合源码。两种提高精度的候选都使本例最大误差下降 60%，RMS 大约减半，但两者有 248955 / 356168 个元素不同，彼此最大差异为 `5.96046448e-8`。输入数据类型和最终归约精度相同，不意味着内部指数计算或输出完全相同。

正式证据为 `runs/20260919T132134.749440Z-softmax-candidates/`。更早的 `131850.273229Z` 运行使用相同数值代码，但对 MLX FP64 的文字说明不够完整；保留该记录，并以补充精度说明后的正式运行作为引用。两轮结果 NPZ 的 SHA-256 相同。

## NumPy FP64 完整回归失败

`harness/encoder_softmax_candidate.py` 在独立进程里替换声学 attention 函数，只改变 softmax。候选统一覆盖 SSL、text、MRTE 和 encoder2，不按层、语言、句子或坐标选择路径。LayerNorm、线性运算、flow、decoder、模型权重、固定语义 token、音素、参考条件和噪声保持原样。

运行前，Harness 在首例上检查本地 attention 展开与原生产编码器：8 个保存阶段全部逐位相同。随后以 NumPy FP64 softmax 检查十例的 120 个阶段。结果为 `numerical_mismatch`、退出码 1，118 个阶段通过。原 `ja-punctuation` MRTE 最大误差从 `1.48058e-4` 降为 `8.67844e-5`，全部中间阶段通过，但两条原本通过的日文波形新增失败：

| 样例 | 原波形最大误差 | 候选波形最大误差 | 候选超差点数 |
|---|---:|---:|---:|
| ja-short | 7.85775483e-5 | 1.14217401e-4 | 4 |
| ja-long | 9.73157585e-5 | 1.22867525e-4 | 10 |

短句最大差异位于 `[0,0,5644]`：误差 `+1.14217401e-4`，允许误差 `1.01804503e-4`。长句最大差异位于 `[0,0,216416]`：误差 `-1.22867525e-4`，允许误差 `1.01065962e-4`。全部 14 个坐标均已保存。

因此否决 NumPy FP64 候选，不写入默认运行时。局部 softmax 更接近官方、概率行和更接近 1，并不足以保证整条计算链更接近官方 FP32 结果。

证据目录为 `runs/20260919T132218.724091Z-encoder-softmax-numpy-fp64/`，含十例完整阶段数组、比较结果、`failed-waveform-points.json` 和哈希复核结果。源码与十个 NPZ 均通过 SHA-256 检查，进程没有导入 PyTorch，释放后 MLX active/cache 均为 0。这些诊断含中间结果捕获，未作正常运行速度或内存收益判断，也未增加 ASR、试听或语音质量验收。

## MLX FP64 累积候选通过十例

第二个候选继续使用 MLX CPU 的指数实现，只将 softmax 输入及累积改为 FP64，最后转回 FP32。它的计算路径与 NumPy 候选不同，不能从前者失败推断后者结果。

`runs/20260919T132504.193131Z-encoder-softmax-mlx-fp64/` 完成十例 120 个阶段对照，全部通过原容差，状态为 `completed`、退出码 0。原 `ja-punctuation` MRTE 最大误差降为 `7.23600388e-5`。`ja-short` 波形最大误差为 `7.03772530e-5`，`ja-long` 为 `1.00553036e-4`，均没有超差元素。

长句波形并未在每项指标上改善：原路径最大误差为 `9.73157585e-5`，候选略高；最接近边界的元素距离该点允许误差仅 `5.12926504e-7`。通过使用的是原有绝对与相对误差共同构成的逐元素边界，不是把所有元素与单独的 `1e-4` 比较。当前结果支持继续测量成本，不能据此扩大模型、语言或功能兼容范围。

8 个 installer parity 阶段、全部源码和十个 NPZ 的 SHA-256 均已核验，进程没有导入 PyTorch，释放后 MLX active/cache 为 0。候选在这一阶段仍只存在于诊断进程中。

## 正常请求的成本

基线和候选分别在独立、无其他模型任务的进程中运行，包、十例条件和顺序相同。每例分别测 CPU 编码器与完整声学请求，各热身两次、计时五次；关闭 capture，没有 observer。同步在计时内，输出复制、数值检查、哈希、RSS 轮询和文件写入在计时外。编码器计时不含 flow/decoder，完整声学计时包含 CPU encoder 与 GPU flow/decoder，不包含文本前端、参考准备或 GPT。

| 样例 | 原编码器 / 候选（毫秒） | 编码器变化 | 原完整声学 / 候选（毫秒） |
|---|---:|---:|---:|
| 日文报告样例 | 17.562 / 19.142 | +9.0% | 229.163 / 217.499 |
| 中文报告样例 | 21.575 / 23.028 | +6.7% | 265.840 / 259.953 |
| 日文长句 | 76.241 / 84.523 | +10.9% | 660.482 / 653.516 |
| 中文长句 | 93.270 / 101.839 | +9.2% | 850.785 / 800.909 |

十例编码器中位耗时增加 2.5%–10.9%。本轮完整声学请求多数略快，混合中英文例慢约 0.4%；两个进程按顺序运行，不能据此认定候选带来稳定速度收益。重点四例候选完整声学 RTF 为 0.0435–0.0453，速度仍实际可用。

内存下表使用十进制 MB。MLX active 峰值在每组热身后重置，覆盖五次测量请求；不是边界快照，也不是 NVIDIA 显存。

| 样例 | 原编码器 active 峰值 / 候选（MB） | 原完整声学 active 峰值 / 候选（MB） |
|---|---:|---:|
| 日文报告样例 | 190.105 / 191.038 | 568.285 / 568.285 |
| 中文报告样例 | 193.745 / 195.120 | 652.286 / 652.286 |
| 日文长句 | 253.135 / 266.488 | 1385.406 / 1385.406 |
| 中文长句 | 288.771 / 309.103 | 1666.409 / 1666.409 |

FP64 累积增加了编码器的工作区，重点四例分别增加约 0.93、1.37、13.35、20.33 MB；完整声学峰值仍由后续阶段主导，基本不变。请求完成后 active 均为 173.176 MB。进程终身 RSS 高水位从 924.96 MB 增至 991.12 MB，增加 66.16 MB；它不能归到某一次请求，也不是分配器峰值。十例持续运行后，边界观察到的最大 cache 从 11.3727 GB 增至 11.4520 GB，增加约 79.24 MB。这里使用默认缓存策略，没有清除不同形状请求积累的 cache。两组释放后 active/cache 均为 0，当前 RSS 分别为 113.52、134.97 MB。

因此这项改动的价值是通过当前正确性对照，代价是编码器耗时及工作区略增；它不是内存优化。不能把完整声学峰值基本不变写成没有任何资源成本。

两组各有 140 次请求，全部在组内输出固定，并通过对应编码器末端或波形的官方容差。候选正常路径的 20 组首份输出还与此前带 capture 的诊断数组逐位相同。全部源码和 40 个输出 NPZ 哈希已核验，进程均未导入 PyTorch。

证据为 `runs/20260919T132857.669384Z-softmax-benchmark-native/` 与 `runs/20260919T132952.107911Z-softmax-benchmark-mlx-fp64/`，逐例成本及捕获等价性汇总保存在后者的 `comparison.json`。逐次时间、请求哈希、内存边界及 RSS 原始值都保留在各自 `result.json` 中。

## 以显式 CPU 选项迁入源码

成本确认后，`MLXSoVITS.load()` 增加 `encoder_softmax="fp32" | "fp64-accumulation"`。默认仍是原 FP32 路径；后者只允许 CPU encoder，并仅将 softmax 分数及累积提高到 FP64，输出恢复 FP32。GPU 与该选项组合会在加载模型前报错，不发生隐式 CPU 搬运。运行时未增加 NumPy softmax 或新的依赖。

`harness/native_prepared_speech.py` 增加同名 `--encoder-softmax` 选项，并将配置写进结果清单，供准备条件整链验证选择。此次源码迁移先用固定声学条件复测：

- `runs/20260919T133516.751269Z-encoder-softmax-mlx-fp64-runtime/`：直接调用生产选项，十例 120 个阶段通过官方原阈值，全部 120 个数组逐位等同 `132504` 的独立候选。
- `runs/20260919T133545.356824Z-encoder-softmax-native-runtime/`：原 FP32 路径的 120 个数组逐位等同 `122652` 历史基线，只保留原有 `ja-punctuation` MRTE 单点失败，状态仍为 `numerical_mismatch`、退出码 1。

非法模式及 GPU 组合的拒绝已实际检查。语法检查与 `git diff --check` 通过。此次迁移没有新增 ASR 或人工试听证据；数值范围仍限于当前十例、所用模型与执行配置。

## 复现与后续范围

离线 softmax 对照：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/softmax_candidates.py \
  --references "$REF" \
  --official-attention-run "$REF/runs/20260919T131028.986460Z-attention-nodes-official"
```

NumPy FP64 十例回归：

```sh
"$REF/.venv-mlx-macos/bin/python" harness/encoder_softmax_candidate.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T122232.246358Z-sovits-fixed-official-mps" \
  --candidate numpy-fp64
```

将最后一项改为 `--candidate mlx-fp64`，即可复现通过十例的 MLX FP64 累积候选。其普通请求成本由独立的 `harness/encoder_softmax_benchmark.py` 测量：两次热身、五次计时，编码器与完整声学请求分别运行，关闭 capture，并将数组复制、比较、哈希和内存轮询放在计时之外。

使用 `encoder_softmax_benchmark.py`、相同包及条件参数，分别指定 `--candidate native` 和 `--candidate mlx-fp64` 可复现成本对照。验证生产迁移时，在 `encoder_softmax_candidate.py` 的 MLX 候选命令上增加：

```sh
--runtime \
--equivalence-reference "$REF/runs/20260919T132504.193131Z-encoder-softmax-mlx-fp64"
```
