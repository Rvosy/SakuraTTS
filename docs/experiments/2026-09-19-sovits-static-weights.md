# SoVITS 权重归一化加载时折叠

日期：2026-09-19。本轮将重复的权重归一化移到加载阶段，保留默认关闭的对照开关。十例的 131 份有效权重哈希和 330 个保存阶段全部逐位一致。Flow 的耗时和阶段峰值下降，但完整声学速度有波动，加载时间、加载峰值和加载后缓存增加，因此暂不改默认。这一优化思路可以用于后续 CUDA 运行时；当前 MLX 验证不能证明 NVIDIA 显存或性能收益。

## 已确认的重复计算

当前 V2Pro 包保留官方 g/v 参数，运行时每次卷积都计算 `v * (g / sqrt(sum(v * v)))`。归约轴仍由包中的原始 `dim` 决定。

| 模块 | 每次完整声学请求的归一化次数 | v 参数字节数 | g 参数字节数 |
|---|---:|---:|---:|
| Flow | 36 | 52,887,552 | 70,656 |
| Decoder | 95 | 54,765,568 | 39,680 |
| 合计 | 131 | 107,653,120 | 110,336 |

这些数值来自 `20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32/manifest.json` 及实际调用链。Flow 的四个 coupling 各有一个条件投影、四个输入卷积和四个 residual/skip 卷积。Decoder 有五个上采样卷积，以及十五个残差块各三对卷积。

固定提交的 Lite `Loader.py` 已在两条加载路径中移除 decoder 的 WeightNorm，普通 checkpoint 路径在 CPU 上折叠后再搬到执行设备。它的 flow 仍保留归一化，`initialize_runtime()` 只做预热和 CUDA Graph 捕获。官方普通 V2Pro TTS/WebUI 路径未调用这项移除操作；其 TorchScript 导出路径移除了 decoder 的 WeightNorm。不能把某一导出路径的行为当成全部官方推理入口的行为。

## 候选的范围

`MLXSoVITSFlow.load/from_package` 和 `MLXSoVITSDecoder.load/from_package` 增加 `fold_weight_norm=False`。组合入口 `MLXSoVITS.load` 也提供同名选项。默认仍保留原 g/v 路径。显式启用后，在当前执行 stream 上调用原有 FP32 归一化函数，逐个求值并以 `prefix.weight` 保存结果，然后删除对应 g/v。原 manifest、模型文件和部署包不变，通用包读取器不承担后端计算。

所有原始 `dim` 当前都是 0。Decoder 的五个 ConvTranspose 权重为 IOK 布局，因此其 dim 0 表示输入通道。候选在原布局中归一化，卷积时才沿用已有的 OKI 转置，避免错误更换归约轴。布局缓存、预先打包和参考条件投影缓存均未混入本轮。

Decoder 的 residual pair 和上采样阶段求值位置保持原样。候选不会同时长期持有 g/v 和折叠结果；否则会额外常驻约 107.65 MB，违背本轮目的。静态权重总量理论上只减少 g 的 110,336 字节；更值得测量的是运行期间重复归一化的耗时和临时分配，不能把全部 v 大小写成常驻内存节省。

## 对照方法

新 Harness 为 `harness/sovits_static_weights.py`，每个配置独立进程运行，支持 CPU/GPU Flow 与 Decoder，声学 Encoder 使用 CPU，并显式记录已有 softmax 选项。

- `diagnostic` 保存 131 份有效权重的哈希，逐例保存 Flow 的 flip/coupling、Decoder 的条件/上采样/残差阶段，以及完整声学链的十二个阶段。Flow 和 Decoder 单独检查时使用同一份官方阶段输入，避免前级误差混入；完整链另行检查。
- `benchmark` 不捕获中间结果，不计算诊断权重哈希。分别测 Flow、Decoder 和完整声学请求，每组预热两次、计时五次。同步包含在计时内，输出复制、数值检查、哈希、RSS 查询和写文件在计时外。
- 折叠配置必须引用同包、同条件、同设备、同 softmax、同样例顺序的未折叠记录。候选要求权重哈希及保存阶段逐位一致。官方对照仍使用 `atol=1e-4, rtol=1e-5`，已知默认 FP32 中间量失败不会被等价性检查掩盖。
- 加载计时包含包校验、读取、组件加载及候选折叠，不包含 Python 模块导入；不清除操作系统文件缓存。单次新进程加载时间不能代表稳定的冷启动分布。
- MLX allocator 峰值在加载前和每组预热后分别重置；RSS 只在边界采样，终身 RSS 高水位另列。它们都不是 NVIDIA 独立显存。最后释放模型并清除缓存，记录卸载后占用。

结果还保存执行命令、源码快照、输入/包/结果哈希、权重所有权检查及 PyTorch 是否被意外导入。自动数值对照不替代 ASR 或试听，本轮不会增加这两项验收声明。

## 正确性结果

主对照使用 CPU Encoder 的既有 `fp64-accumulation` 选项，以及 GPU FP32 Flow/Decoder。十例覆盖原始报告样例、短句、长句、标点和已有混合样例。未折叠与折叠诊断都实际退出 0；131 份有效权重哈希相同，330 个阶段逐位相同，140 项官方检查全部通过。后者包括完整声学链的 120 项，以及同一官方阶段输入上的十项 Flow 输出、十项 Decoder 波形。没有放宽容差或更换输入。

另补两条原始报告样例的默认 FP32 Encoder 路径，作为通用改动的既有能力回归：两侧均退出 0，131 份权重哈希、66 个阶段逐位相同，28 项官方检查通过。这不改变之前十例默认 FP32 中 `ja-punctuation` MRTE 的已知失败状态；本轮没有重新验证该失败。CPU Flow/Decoder 尚未运行，不能由 GPU 结果宣称其通过。

折叠后 Flow 的 72 份、Decoder 的 190 份 g/v 均已删除，分别留下 36 份、95 份有效权重。权重仍为 FP32，布局转置仍在原位置。所有模型进程释放后，MLX active/cache 均为 0，没有导入 PyTorch。

## 正常运行成本

正常基线和候选在独占计算窗口内按顺序运行，各用一个新进程、同样的十例顺序和默认 MLX 缓存策略。每例三个阶段分别预热两次、测量五次，每个进程共 210 次阶段请求，其中 70 次为完整声学请求。全部输出在组内保持一致，候选的 30 组首份输出与基线逐位一致。两侧各 30 组正常输出还分别与诊断捕获结果逐位相同，共 60 项捕获等价性检查。

下表为中位耗时，单位毫秒。数字只代表本轮两个顺序执行的进程。

| 样例 | Flow：未折叠 → 折叠 | 完整声学：未折叠 → 折叠 | 完整声学变化 |
|---|---:|---:|---:|
| 日文报告样例 | 10.713 → 3.707 | 239.512 → 240.663 | +0.48% |
| 中文报告样例 | 6.829 → 4.453 | 284.845 → 293.926 | +3.19% |
| 日文短句 | 5.797 → 2.918 | 73.712 → 67.289 | -8.71% |
| 中文短句 | 5.034 → 2.974 | 98.905 → 105.677 | +6.85% |
| 日文长句 | 10.489 → 8.342 | 743.444 → 727.080 | -2.20% |
| 中文长句 | 13.237 → 8.979 | 910.901 → 872.239 | -4.24% |
| 日文标点 | 8.736 → 5.121 | 416.386 → 406.526 | -2.37% |
| 中文标点 | 7.912 → 4.791 | 353.045 → 324.511 | -8.08% |
| 中英文混合 | 6.953 → 4.198 | 276.341 → 252.989 | -8.45% |
| 日英文混合 | 6.600 → 3.956 | 233.182 → 212.434 | -8.90% |

Flow 十例中位耗时均下降，幅度为 20.5%–65.4%。Decoder 单独计时的变化为 -7.37% 到 +9.93%，完整声学为 -8.90% 到 +6.85%；没有证据支持稳定的完整请求加速。阶段分别测量，不能把各阶段表中差值相加，推算完整请求改善。候选日文报告样例、短句、长句的完整声学 RTF 分别为 0.0501、0.0731、0.0487；范围仍不包含文本、参考准备和 GPT。

内存使用十进制 MB。各阶段 MLX allocator active 峰值在该组预热后重置，覆盖五次正常请求。

| 样例 | Flow 峰值：未折叠 → 折叠 | 完整声学峰值：未折叠 → 折叠 |
|---|---:|---:|
| 日文报告样例 | 326.074 → 220.347 | 568.285 → 563.288 |
| 中文报告样例 | 330.925 → 225.099 | 652.270 → 647.273 |
| 日文短句 | 308.788 → 203.291 | 309.001 → 250.599 |
| 日文长句 | 372.393 → 266.533 | 1385.406 → 1380.407 |
| 中文长句 | 396.348 → 290.488 | 1666.392 → 1661.394 |

Flow 的十例峰值减少约 105–107 MB，完整声学多数减少约 5 MB；日文短句从 Flow 主导峰值变为 Decoder 主导，减少 58.402 MB。较长波形仍由 Decoder 的临时分配主导，折叠全部 WeightNorm 并没有等比例降低完整请求峰值。

加载代价需要单列：未折叠与折叠的加载耗时为 0.1651 / 0.2158 秒，本轮多约 50.7 ms；这是各进程的一次观测，不能代表稳定的冷启动分布。加载峰值从 173.176 MB 增至 189.903 MB，增加 16.727 MB。加载完成的 active 从 173.176 MB 变为 173.161 MB，但 cache 从 0 增至 59.593 MB。虽然 g 已被删除，折叠所需的临时分配仍被默认 allocator 缓存策略保留；因此不能宣称空闲总占用改善。逻辑张量字节数和 allocator 统计并非同一口径，不能用减少的 110,336 字节 g 参数直接推算 active 差值。

十种形状连续运行后，边界观察到的最大 cache 为 11.593 / 11.505 GB。这里没有应用先前独立研究的缓存上限，不是推荐的产品缓存配置。进程终身 RSS 高水位为 1002.324 / 947.519 MB；它包含预读条件、NumPy 检查和其他内存，不能归到某一声学请求。最后模型释放并清缓存后，两侧 active/cache 都为 0，当前 RSS 为 153.797 / 143.131 MB，仍包含 Harness 中保留的官方数组。

## 证据与复现

参考目录 `runs/` 下新增六轮记录：

| 配置 | 未折叠 | 折叠 |
|---|---|---|
| 十例诊断，FP64 累积 Encoder | `20260919T140221.860678Z-sovits-static-weights-diagnostic-gpu-unfused` | `20260919T140246.931328Z-sovits-static-weights-diagnostic-gpu-folded` |
| 十例正常 benchmark | `20260919T140308.464819Z-sovits-static-weights-benchmark-gpu-unfused` | `20260919T140451.620477Z-sovits-static-weights-benchmark-gpu-folded` |
| 两原始例诊断，默认 FP32 Encoder | `20260919T140609.431378Z-sovits-static-weights-diagnostic-gpu-unfused` | `20260919T140738.080794Z-sovits-static-weights-diagnostic-gpu-folded` |

六个模型进程的真实退出码均为 0。`140451.620477Z` 目录还保存 `analyze_evidence.py`、`benchmark-comparison.json`、`evidence-verification.json`、`process-exits.json` 和 `evidence-file-sha256.json`。离线审计已复核 72 份源码快照、84 个结果数组文件、包和官方结果的哈希；每对配置的源码快照相同。原始逐次计时、内存观测、执行命令和结果哈希均保留。没有增加 ASR 或人工试听，也没有改动原模型或历史结果。

下面先用已通过声学阶段对照的 CPU FP64 累积 softmax 固定 Encoder，隔离 WeightNorm 变化。`SOVITS_UNFUSED_RUN` 应设为对应第一条命令输出的目录；benchmark 与 CPU 对照必须使用各自的新基线，不能复用 GPU 诊断记录。

```sh
SOVITS_REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
SOVITS_PY="$SOVITS_REF/.venv-mlx-macos/bin/python"
SOVITS_PACKAGE="$SOVITS_REF/models/converted/20260919T123227.630582Z-20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32-lossless-storage"
SOVITS_CONDITIONS="$SOVITS_REF/runs/20260919T122232.246358Z-sovits-fixed-official-mps"

"$SOVITS_PY" harness/sovits_static_weights.py --references "$SOVITS_REF" --package "$SOVITS_PACKAGE" --official-conditions "$SOVITS_CONDITIONS" --mode diagnostic --device gpu --encoder-softmax fp64-accumulation --no-fold-weight-norm

"$SOVITS_PY" harness/sovits_static_weights.py --references "$SOVITS_REF" --package "$SOVITS_PACKAGE" --official-conditions "$SOVITS_CONDITIONS" --mode diagnostic --device gpu --encoder-softmax fp64-accumulation --fold-weight-norm --equivalence-reference "$SOVITS_UNFUSED_RUN"
```

通过后，将两条命令的 `--mode` 同时改为 `benchmark`，以新生成的未折叠 benchmark 目录作对照。CPU 边界验证将两侧 `--device` 同时改为 `cpu`；原始默认路径验证将两侧 `--encoder-softmax` 同时改为 `fp32`。默认 FP32 的既有官方超差应保留为 `numerical_mismatch`，并单独检查折叠等价性，不能改成通过。

## 当前状态

候选的价值已落实到当前 GPU Flow 的重复计算和阶段峰值，完整声学也有较小的峰值改善。但加载后缓存增加，完整速度并非各例都改善，继续默认关闭。下一步如接入实际运行入口，应保留显式开关并沿用经过验证的缓存生命周期策略，再做日文完整请求回归；不把本轮结果扩展为 CUDA、CPU Flow/Decoder 或其他模型版本的结论。
