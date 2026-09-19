# 绑定参考条件后省去声学条件权重

本轮在 Apple M4 上验证了一个可保留的内存优化：先用指定参考条件计算五个投影，再装载其余声学权重。声学模型空闲时的 MLX active 从 165.154 MiB 降到 139.130 MiB，减少 26.023 MiB；原始模型包完整保留。四个日文样例的阶段结果、波形和 PCM 与完整权重路径逐字节一致。

证据来自独立 Harness，尚不等于产品入口已经通过验证。原始权重路径仍是对照。热声学耗时变化较小，长句没有加速；本轮保留该方向的理由是实际内存收益。

## 实现与范围

模型为现有朱雀院红叶 V2Pro，官方行为参考提交为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`。声学编码器运行在 MLX CPU，Flow 和 Decoder 运行在 MLX GPU，均为 FP32，未启用 WeightNorm 折叠。完整 prepared 请求使用 CPU FP64 GPT Prefill、GPU FP32 Decode。

`harness/sovits_bound_projection.py` 只在实验类中实现绑定。一次装载只打开并校验一次完整 SoVITS 包，依次完成：

1. 读取四个 Flow `cond_layer` 的 `g/v/bias`，以及 `dec.cond` 的普通 `weight/bias`，共 14 个张量。
2. 用原基类 FP32 卷积计算五个投影并求值，释放临时权重、组件和分配器缓存。
3. 在读取前从名单中排除这 14 个权重，装载其余 636 个张量。

14 个 FP32 权重共 27,314,176 bytes，五个投影为 26,624 bytes。净差为 27,287,552 bytes，正好对应实测常驻差。每次独立投影准备结束后，MLX active 为 26,624 bytes、cache 为零，未留下条件权重。剩余权重的读取记录与 14 个准备权重没有交集。

绑定保存包身份及 `ge/ge512` 的 dtype、shape、内容散列，不保存调用方可变数组引用。每次 decode 在编码器执行前核对条件。更换参考需先销毁旧实例，再重新准备和装载；失败后 slot 保持为空。该设计增加了更换参考的装载成本，适合反复使用同一角色参考的请求。

原模型文件未改写，末次重新核验的 manifest 和权重 SHA-256 与输入一致。本轮没有减小安装包，也没有删除离线转换所需的原始权重。

## 正确性与错误恢复

四个固定样例为 `ja-reported-intro`、`ja-short`、`ja-long`、`ja-punctuation`。沿用 portable validation bundle 的原始输入、采样参数和显式声学 noise。

诊断顺序为 A→B→A，每次装载后执行全部四例。12 次诊断均保存 `quantized`、`ssl_encoded`、`text_encoded`、`mrte`、`encoder_hidden`、`mean`、`log_scale`、`mask`、`flow_input`、`flow_output`、`decoder_input`、`waveform` 和 PCM，共 156 个数组，与上一轮完整权重基线逐字节一致。B 只替换 `ge/ge512`，沿用 A 的语义、音素和 noise，是声学切换探针；没有据此宣称参考 B 的完整 TTS 已重新通过官方对照。

每次装载分别试验错误 `ge`、错误 `ge512` 和错误包身份，共九次，均在编码器执行前拒绝。另在编码器已经装载、Flow 尚未装载时注入异常，确认旧实例已销毁、slot 为空、MLX active/cache 均为零，再显式装载 A 并复现相同波形和 PCM。

官方数值检查仍用 `atol=1e-4, rtol=1e-5`。`ja-punctuation.mrte` 保留原超差：最大绝对误差 `0.0001480579376220703`，平均绝对误差 `5.7309517050404715e-06`。本轮证明绑定与完整权重路径等价，没有修复或放宽这个既有差异。

完整 prepared 补验实际执行 GPT 生成，再把这次生成的语义送入声学模型。两种路径的四例共 731 个采样步骤、727 个语义 token，token、history、返回索引、停止原因集合、波形和 PCM 均一致。没有把目标 gold token 当作自由生成输入；记录中的真实官方随机抽样仍作为受控实验输入。

## 装载与常驻

资源对照按完整权重、绑定、绑定、完整权重的顺序运行四个独立进程。以下是两次装载的原始耗时，单位毫秒。总装载包含校验、投影准备与释放、剩余模型装载和同步；投影后的边界读数只读取分配器计数，不调用 `ps`。

| 阶段 | 完整权重，两次 | 绑定，两次 |
| --- | ---: | ---: |
| 包校验及身份检查 | 33.140 / 35.180 | 33.232 / 33.663 |
| 投影准备及释放 | 0 / 0 | 25.855 / 25.250 |
| 其余模型装载 | 117.613 / 124.064 | 100.539 / 103.100 |
| 总装载 | 151.095 / 159.631 | 160.000 / 162.393 |

独立声学实例的常驻及装载全过程分配器峰值，均为完整权重 173,176,320 bytes、绑定 145,888,768 bytes。绑定准备阶段最高 77,701,120 bytes，未超过最终模型常驻。完整 prepared worker 同时保留 GPT 时，模型空闲 active 从 469.047 MiB 降到 443.023 MiB。

这些是新进程装载，操作系统文件缓存未清空；不能当作磁盘冷读性能。各 worker 卸载后模型弱引用失效，MLX active/cache 均为零。

## 正常声学请求

每个资源 worker 对每例先热身五次、再测七次。四个 worker 共执行 192 次无 capture 声学调用，每种策略每例有 14 次正式测量。所有输出与完整权重基线逐字节一致。

表中耗时为独立 ABBA 批次的中位数，不是逐请求配对差。计时包含绑定检查和完成同步，输出复制、PCM、散列、内存读取及分阶段诊断在计时外。峰值在每次调用前重置，读取的是完整无 capture 声学调用的 MLX allocator peak。

| 样例 | 完整权重耗时 ms | 绑定耗时 ms | 完整权重峰值 MiB | 绑定峰值中位数 MiB | 峰值减少 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| reported-intro | 215.741 | 214.337 | 541.959 | 515.935 | 26.023 |
| short | 64.114 | 62.689 | 294.969 | 220.852 | 74.117 |
| long | 638.719 | 638.820 | 1321.242 | 1295.219 | 26.023 |
| punctuation | 363.518 | 362.696 | 822.226 | 796.203 | 26.023 |

短句的绑定峰值在 217.732–220.945 MiB 之间。它的额外收益与 Flow 工作区有关，不能推广到所有长度。单独的同步分阶段诊断中，短句 Flow 峰值由约 294.922 MiB 降至 220.750 MiB；长句 Flow 也下降，但长句总峰值由 Decoder 主导，所以整个声学调用只减少常驻的 26.023 MiB。

分阶段计时同样支持把速度收益放在次要位置：短句 Flow 约 5.230→3.617 ms、Decoder 约 51.224→52.073 ms；长句 Flow 约 9.886→8.494 ms、Decoder 约 557.936→556.300 ms。这些带阶段同步的诊断不能与上表的完整调用计时混用。

## 完整 prepared 请求

补验分别在完整权重和绑定模型进程执行四个实际 GPT 请求，每例每种策略只运行一次。范围是已准备好的文本和参考特征→自身 GPT 历史→声学→PCM，包含输出复制和 PCM；不含文本前端、参考音频处理、模型装载和文件写入。

| 样例 | 采样步骤 / 语义 token | 完整权重总耗时 ms | 绑定总耗时 ms | 完整权重峰值 MiB | 绑定峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| reported-intro | 121 / 120 | 1034.052 | 1087.098 | 845.851 | 819.827 |
| short | 24 / 23 | 325.731 | 329.320 | 599.078 | 555.965 |
| long | 374 / 373 | 3127.767 | 3126.056 | 1625.119 | 1599.095 |
| punctuation | 212 / 211 | 1783.124 | 1730.851 | 1126.118 | 1100.094 |

这组测量复核了完整 prepared 链的输出和资源收益。单次耗时有上下浮动，不能证明热端到端速度提升。四例 RTF 分别约为完整权重 0.215 / 0.354 / 0.210 / 0.211，绑定 0.226 / 0.358 / 0.210 / 0.205；原始结果还分别保存 semantic、acoustic 和输出转换耗时。

本轮没有新增 ASR、人工试听或 CUDA 验证。M4 的 MLX 分配器指标属于 Apple 统一内存；RSS 边界快照与进程 lifetime maxrss 单独记录，不冒充单次峰值或 NVIDIA 独立显存。中文、其他模型家族以及未覆盖的精度和功能，不因本轮结果而扩大兼容声明。

## 失败记录、证据与复现

首轮完整 prepared 基线在第一例停止原因比较处退出 1：实际顺序为 `['argmax_eos', 'sample_eos']`，bundle 顺序为 `['sample_eos', 'argmax_eos']`，两者表示相同停止条件。其他 token、语义、波形、PCM 检查均通过。修复 Harness 后按集合比较，并同时保存 `actual_stop` 与 `expected_stop`；运行时停止逻辑未改动。

原失败记录、异常栈和退出码保留。补验目录复制首轮的全部冻结源码，只改上述比较和结果字段，重新运行两个 prepared worker，退出码均为零。五个已成功的诊断/资源 worker 不重复测量，也不将失败的首轮总进程标成成功。

- 首轮诊断、四个资源 worker 及失败记录：`SakuraTTS-References/runs/20260919T165443.130499Z-sovits-bound-projection`
- 完整 prepared 补验、`run_prepared.py`、合并 `audit.py/audit.json`：`SakuraTTS-References/runs/20260919T165829.865520Z-sovits-bound-projection-prepared-recheck`
- 完整权重阶段基线：`SakuraTTS-References/runs/20260919T163620.552175Z-sovits-reference-projection/diagnostic-baseline`

各目录保存源文件 SHA-256、模型和输入身份、实际命令、退出码、stdout/stderr、数组及其散列。独立只读审计复算了阶段字节比较、权重读取名单、正常资源结果、完整生成历史和卸载状态，未发现剩余问题。合并审计为 `passed`，原模型包散列未变。

当前源码还给实验 `SelectedSource.tensors` 增加了 `exclude=()` 参数，与随后新增的产品读取接口兼容；内部排除项和调用方排除项共同生效。该变化没有改写冻结实验。已用不加载模型的接口 mock 验证先筛选再读取，结果保存在补验目录的 `source-adapter-compatibility.json`。本文数值始终以冻结源码为准。

完整实验可用以下命令重建，输出目录必须不存在：

```sh
/Users/beyondpower/Documents/Projects/SakuraTTS-References/.venv-japanese-macos/bin/python \
  /Users/beyondpower/Documents/Projects/SakuraTTS/harness/sovits_bound_projection.py run \
  --baseline-run /Users/beyondpower/Documents/Projects/SakuraTTS-References/runs/20260919T163620.552175Z-sovits-reference-projection \
  --gpt-package /Users/beyondpower/Documents/Projects/SakuraTTS-References/models/converted/20260919T123226.764557Z-20260919T104221.749176Z-010197bfc30b-gpt-fp32-lossless-storage \
  --output /Users/beyondpower/Documents/Projects/SakuraTTS-References/runs/new-sovits-bound-projection
```

复核既有证据无需模型执行：

```sh
/Users/beyondpower/Documents/Projects/SakuraTTS-References/.venv-japanese-macos/bin/python \
  /Users/beyondpower/Documents/Projects/SakuraTTS-References/runs/20260919T165829.865520Z-sovits-bound-projection-prepared-recheck/audit.py
```
