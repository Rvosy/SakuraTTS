# SSL 和文本编码器的逐层数值对照

日期：2026-09-19。继续定位 `ja-punctuation` 的 MRTE 超差。前一轮已经确认：把 MLX 编码器输出交给官方 MRTE，也会在同一坐标超差，见 [MRTE 来源分解](2026-09-19-mrte-error-attribution.md)。本轮保存上游 SSL 三层、text 六层的输出，并用相同输入单独检查 attention、两处 LayerNorm 和 FFN。

## 第一处差异与放大位置

`harness/encoder_layer_diagnosis.py` 在固定官方模块上加 hooks，保存每层入口、attention 输出、两次归一化输入/输出和 FFN 边界，共 76 个数组。官方执行结果与先前保存的 SSL/text 输出逐位相同。MLX 诊断展开也逐位复现原 CPU 编码器的最终输出，没有改变原失败。

SSL 入口的 1×1 投影和文本 embedding 与官方逐位相同，首个非零差异出现在第 0 层 attention。SSL 该处最大绝对误差为 `9.53674e-6`、RMS 为 `7.42821e-7`。随后的第一处归一化将 RMS 降到 `1.51385e-8`，所以不能只按第一处差异的大小判断它对最后输出的影响。

| SSL 阶段 | 累计最大绝对误差 | 累计 RMS 误差 |
|---|---:|---:|
| 入口投影 | 0 | 0 |
| 第 0 层 attention | 9.53674e-6 | 7.42821e-7 |
| 第 0 层输出 | 5.96046e-7 | 7.798e-8 |
| 第 1 层输出 | 1.72853e-6 | 2.178e-7 |
| 第 2 层 norm2 输入 | 7.15256e-7 | 9.58143e-8 |
| 第 2 层 norm2 输出 | 6.07967e-6 | 8.86650e-7 |

末层 norm2 将该输入中的累计 RMS 差异放大约 9.25 倍。同样的官方输入单独送入 MLX norm2 时，输出 RMS 差异为 `1.26104e-7`，说明该阶段既传播已有差异，也包含自身的 FP32 归约差异。

九层 FFN 在官方同输入下全部逐位一致；attention 和 LayerNorm 的 36 项同输入组件对照均通过原容差，76 项累计阶段对照也均通过。这里没有发现公式或权重错误，也没有哪一层单独超过当前阈值。失败发生在这些小差异继续经过 MRTE 之后。

证据目录：

- 官方逐层捕获：`runs/20260919T124415.513327Z-encoder-layer-diagnosis-official-none/`。
- MLX CPU 基线：`runs/20260919T124435.554050Z-encoder-layer-diagnosis-mlx-none/`。

## 统一 FP64 LayerNorm 候选

依据末层归一化的敏感性，本轮只测一个通用候选：所有 LayerNorm 使用 CPU NumPy FP64 计算均值、中心化方差和仿射，再一次转为 FP32。epsilon、gamma、beta 和运算含义不变。候选通过诊断进程内替换 `norm` 方法实现，生产源码和默认路径均未修改；它没有按层、句子或坐标设置特例。

目标例的 76 项累计阶段与 36 项同输入组件比较全部通过。MRTE 最大绝对误差由 `1.48058e-4` 变为 `7.34329e-5`，原单个超差元素消失。证据为 `runs/20260919T124513.269301Z-encoder-layer-diagnosis-mlx-fp64-layernorm/`。

这不表示每项误差都改善。SSL 输出 RMS 只从 `8.86650e-7` 变为 `8.84017e-7`；text 输出最大误差从 `7.30157e-7` 增至 `8.04663e-7`，但 RMS 从 `1.66422e-7` 降至 `1.54160e-7`。单条 MRTE 通过可能包含不同舍入误差之间的抵消，不能据此采纳候选或宣布完整兼容。

三轮诊断的源码快照和数组哈希均已核验，原失败仍保留。诊断捕获和同输入组件重放会改变同步与工作量，不能用这里的耗时判断正常运行成本。

完整回归使用 `harness/encoder_norm_candidate.py`，把候选统一应用到声学编码器的所有 LayerNorm，包括 MRTE 后的 encoder2。`verify` 模式比较十条固定条件下的全部 12 个阶段；`benchmark` 模式关闭捕获和 observer，同步完整声学请求，在计时外复制波形，分别运行 native 与 fp64 两个独立进程。

## 十例回归否决候选

`runs/20260919T125220.037462Z-encoder-norm-verify-fp64/` 比较了相同十例的 120 个阶段。原先的 `ja-punctuation` MRTE 超差消失，全部中间阶段通过，但 `ja-long` 最终波形新增两个超差点，整轮为 `numerical_mismatch`、退出码 1：

| 从 0 开始的波形坐标 | 误差 | 该点允许误差 |
|---|---:|---:|
| `[0,0,216416]` | -1.05775893e-4 | 1.01065962e-4 |
| `[0,0,216504]` | +1.04011036e-4 | 1.00083881e-4 |

原 native 路径的这条长句波形最大误差为 `9.73158e-5`，没有超差点。候选只是将一处中间阶段失败换成了长句末端波形失败，没有达到回归要求，因此不采纳。没有把单例改善写入默认运行时，也没有针对长句或坐标增加例外。候选仅保留在独立诊断 Harness 中，用于复现实验。

## 正常运行成本

两种归一化路径在独立安静进程中，对同一原 FP32 包和两条报告样例运行一次首次请求、五次热请求。CPU encoder、GPU flow/decoder、默认缓存策略保持一致，计时包含同步和完整声学执行，不含阶段捕获、observer 或输出 CPU 复制。

| 样例 | native 热运行中位数 | FP64 候选热运行中位数 | 本轮差异 |
|---|---:|---:|---:|
| 日文报告样例 | 0.226306 秒 | 0.218519 秒 | -3.44% |
| 中文报告样例 | 0.249989 秒 | 0.249572 秒 | -0.17% |

中文基本持平，日文在本轮中位数略低；单次顺序实验不证明稳定速度收益。两组各六次波形在组内逐位相同，并通过这两条样例的官方波形容差。这个局部性能结果不会覆盖十例正确性失败。

证据为 `runs/20260919T125251.823617Z-encoder-norm-benchmark-native/` 与 `runs/20260919T125313.315415Z-encoder-norm-benchmark-fp64/`。十例回归和两轮正常计时的源码、数组哈希均已复核，进程没有导入 PyTorch，释放后 MLX active/cache 都为 0。MLX allocator 不覆盖 NumPy 的全部临时内存，本轮没有据此声称资源下降。

复现逐层基线：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-official-macos/bin/python" harness/encoder_layer_diagnosis.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T122232.246358Z-sovits-fixed-official-mps" \
  --native-acoustic "$REF/runs/20260919T122652.881973Z-mlx-sovits-complete-gpu" \
  --backend official
```

在 MLX 环境用相同参数，把 `--backend` 改为 `mlx` 并增加 `--official-layer-run` 指向新生成的官方逐层目录，即可复现基线。再增加 `--candidate fp64-layernorm` 可复现候选。本轮没有新增 ASR、人工试听或质量验收。

复现候选十例失败：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/encoder_norm_candidate.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T122232.246358Z-sovits-fixed-official-mps" \
  --mode verify --normalization fp64
```

正常成本使用相同参数，将模式改为 `--mode benchmark`，增加 `--cases ja-reported-intro zh-reported-greeting --repeats 5`，分别以 `--normalization native` 和 `--normalization fp64` 运行。下一步应回到首次 attention 舍入差异继续定位，保留原运行路径与已知 MRTE 失败，不能用另一个输入上的退化换取目标样例通过。
