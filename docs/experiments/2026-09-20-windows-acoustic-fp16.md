# Windows / RTX 5060：声学 FP16 候选

本轮把已验证的 FP32 声学包转换成独立 FP16 候选，并解决了 ORT 1.19.2 转置卷积带来的重复执行波动。最终四类固定条件通过工程筛查；同一候选的诊断图、生产图和带 profile 的生产图输出逐位一致，每次重复执行也一致。原 FP32 数值容差仍然失败，人工听音和 ASR 尚未验收，因此候选需要显式开启，FP32 继续保留。

短句、长句、多句、标点的热声学耗时分别从 67.42 / 650.80 / 277.96 / 207.22 ms 降到 26.39 / 235.06 / 89.31 / 66.71 ms。这里只测声学计算，不含前端、GPT、跨进程传输、尾部静音和完整请求开销。

## 输入、精度和运行边界

模型是已验证的 Sakura V2ProPlus，来源为 `models/windows-sakura/sovits-onnx-v1`。转换器创建新目录，没有覆盖 FP32 包。运行使用原独立 Python 3.9.13 / NumPy 1.23.4 / ONNX Runtime 1.19.2 CUDA 环境；转换使用开发环境的 ONNX 1.23.0 和 ORT 1.30.0，不导入 PyTorch。

四类输入来自原 FP32 捕获：43 / 675 / 258 / 202 个语义 token。直接读取捕获中的目标音素、`ge512`、声学噪声，以及同目录参考条件中的 `ge`，不重新采样或替换参考。FP32 基线的编码器隐藏值、均值、log scale 和 mask 均通过原捕获容差检查。

公共接口保持 int64 的语义和音素、FP32 的参考条件和显式噪声、FP32 的返回波形。图内权重和主要中间张量实际使用 FP16；生产 profile 观察到 223 个 Conv、50 个 MatMul、13 个 Softmax、24 个 LayerNormalization 对应的 CUDA FP16 输出，没有神经计算回退 CPU。CPU 仍执行部分整数 shape 运算。

LayerNormalization 保留 ONNX opset 17 的 FP32 累积规则，未把整个归一化或注意力分区改回 FP32。初始化张量中的浮点权重从 248,561,536 字节减到 124,280,778 字节，约减半。这是权重数据量，不是显存实测；零插值产生的临时张量仍须计入后续完整请求峰值。

## 重复波动的定位和修复

最初只用 ORT 转换器的默认 FP32 算子排除列表。四类波形均通过预定幅度和频谱筛查，但诊断图与生产图最大差约 0.0007–0.0014，超过原 `atol=1e-4, rtol=1e-5`。改用 BASIC 图优化后仍有差异，不能归因于扩展融合。

随后在同一个 Session 内重复执行。编码器、flow 和 decoder input 共 11 个阶段逐位相同，只有最终波形波动，最大约 0.0007–0.0015。开启 `use_deterministic_compute` 没有解决。ORT 1.19.2 的 [CUDA ConvTranspose 实现](https://github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/core/providers/cuda/nn/conv_transpose.cc#L256-L278) 在这里直接搜索 cuDNN backward-data 算法并缓存结果，没有读取该确定性开关。把 5 个 ConvTranspose 单独保留 FP32 后，波动减到约 0.00024–0.00049，仍未过门槛。上述失败记录全部保留。

最终将这 5 个一维 ConvTranspose 改写为零插值和普通 Conv。步长为 `s` 时，在相邻输入之间填入 `s-1` 个零，权重交换输入/输出通道并翻转 kernel 轴。实现保留最后一组尾随零，因此普通卷积的左右 padding 分别是：

```text
left  = (kernel - 1) * dilation - original_left
right = (kernel - 1) * dilation - original_right - (stride - 1) + output_padding
```

五层的参数按源图读取，未修改音频长度或裁去边界。当前改写支持显式 padding、group=1 的一维结构；需要负 padding、自动 padding 或显式 output_shape 的配置会报错。

改写先在 FP32 验证，再转成 FP16：

- 6 组 stride、kernel、非对称 padding、output_padding 和 dilation 配置，结合 5 种长度及随机、首端、尾端脉冲，共 90 组。原 ConvTranspose 和改写后的 Conv 都与独立 float64 scatter 公式比较，首尾采样点单独检查，全部通过。
- 完整真实声学图使用原包保存的 1 / 2 / 7 / 19 token 输入，48 个阶段均满足原 `atol=1e-4, rtol=1e-5`。最大阶段误差为 5.05e-5。
- 转成 FP16 后，用四类原始捕获执行最终筛查。诊断、生产、生产 profile 各次重复都保留输出并逐项检查；候选所有阶段的重复误差为 0，生产与诊断及 profile 的波形差也为 0。

这组结果支持将本轮波动归因于被替换的转置卷积路径；没有通过放宽重复性容差让候选过关。

## 工程筛查与误差

首次 GPU 执行前固定以下门槛，每轮保存到 `thresholds-before-run.json`：波形形状不变且全部有限；最大绝对误差不超过 0.05，RMSE 不超过 0.005，SNR 至少 25 dB；峰值不超过基线的 1.05 倍加 1e-4；频谱收敛误差不超过 5%，活跃频点 log 幅度 RMS 差不超过 1 dB。频谱使用 1024 点 Hann 窗、256 点 hop，活跃频点门槛为参考幅度峰值以下 60 dB。

这些门槛用于发现工程异常，没有经过主观听音标定。原 FP32 容差单独完整记录，不随工程筛查改变。

| 输入 | 最大绝对误差 | RMSE | SNR dB | 频谱收敛误差 | 活跃频点 log RMS 差 dB |
|---|---:|---:|---:|---:|---:|
| 短句 | 0.01417 | 0.001068 | 40.00 | 0.299% | 0.172 |
| 长句 | 0.03843 | 0.001040 | 39.54 | 0.477% | 0.151 |
| 多句 | 0.02498 | 0.000921 | 40.48 | 0.497% | 0.135 |
| 标点 | 0.03314 | 0.000918 | 40.20 | 0.385% | 0.163 |

四类波形长度均与基线相同，没有超出 [-1,1] 的样本。误差直接计算于未归一化的原始声学波形。上述结果不能证明没有听感差异、漏读或音色变化。

## 声学耗时

每个图在独立新进程中加载；每类首次执行后测 3 次。表中取无 profile 生产图的中位数，包含 CPU 输入到 CPU 波形返回，不含 NPZ 写盘。CUDA 使用 HEURISTIC、关闭最大卷积工作区、`kSameAsRequested`、4 个 intra-op 线程，不启用持续资源采样。

| 输入 | 纯声学音频 s | FP32 ms | FP16 候选 ms | 降低 |
|---|---:|---:|---:|---:|
| 短句 | 1.72 | 67.42 | 26.39 | 60.9% |
| 长句 | 27.00 | 650.80 | 235.06 | 63.9% |
| 多句 | 10.32 | 277.96 | 89.31 | 67.9% |
| 标点 | 8.08 | 207.22 | 66.71 | 67.8% |

这是小样本的本机声学结果，不是完整请求加速比例，也不代表另一种模型、GPU 或 ORT 构建已经验证。

## 完整请求组合验证

随后把声学候选接入完整日文请求，GPT 保持 FP32，并使用 `split-kv`、chunk 256、capacity 2048 和单步 CUDA Graph。四类文本各测 3 次热请求，另有首次请求、卸载后的首次请求和卸载后的热请求，共 15 次。前端、参考条件、随机采样输入、噪声和请求参数都与全 FP32 对照相同。

独立比较按请求名配对，逐项检查参考身份、数值参考路径、参数、音素、采样 token、声学 token、停止原因、返回索引和音频长度。15 对全部一致；WAV 中的 PCM 哈希也与运行报告一致。完整 int16 PCM 除以 32768 后，沿用本轮预先设定的工程门槛，15 对全部通过；去除追加静音再比较，结果仍全部通过。最大绝对误差为 0.03842，最低 SNR 为 39.55 dB。

原 FP32 PCM 容差仍失败，因此组合 benchmark 保留 `replay_validation_failed`，没有把工程筛查改写成严格等价通过。人工听音和 ASR 仍未验收。

| 输入 | 完整音频 s | 原 FP32 ms | FP32 + split-KV ms | FP32 GPT + split-KV + FP16 声学 ms |
|---|---:|---:|---:|---:|
| 短句 | 2.02 | 168.94 | 178.42 | 122.23 |
| 长句 | 27.30 | 3045.15 | 2007.75 | 1592.30 |
| 多句 | 10.62 | 932.13 | 793.92 | 651.49 |
| 标点 | 8.38 | 691.01 | 612.60 | 469.74 |

表中均为无持续资源采样的完整请求热中位数，n=3，包含原始文本到完整 PCM，不包含 WAV 写盘。相同 split-KV 下，声学 FP16 让长句请求再减少约 20.7%；与原 FP32 执行路径相比，两项改动合计减少约 47.7%。短句的 split-KV 单项收益不明显，应保留单项与组合数据，不能把全部提升归到声学转换。

独立资源轮使用相同捕获和数值参考，各类测 1 次，并包含加载、请求、卸载和重载。原 FP32 路径的全卡初始样本为 2208 MiB，峰值 4012 MiB；组合候选初始为 2218 MiB，峰值 3842 MiB。分别减去最早样本后是 1804 / 1624 MiB。这里同时改变了 attention 与声学路径，不能据此单独归因于 FP16 声学。

全卡数字来自 WDDM 下的 `nvidia-smi` 采样，包含桌面活动，也可能漏掉很短的峰值，不是精确的进程独占显存。主进程及子进程的主存 RSS 峰值从约 1880.7 MiB 增到 1993.7 MiB；主存和显存分别记录，不能相互替代。资源轮的请求耗时不用于上表。

组合结果、15 对完整 PCM 及去静音后的指标、原始容差失败、WAV/结果/配置哈希、资源采样摘要保存在 [完整请求数据](data/2026-09-20-windows-acoustic-full-request.json)。原始请求目录为 `outputs/windows-acoustic/combined-fp32-gpt-replay`，增量对照为 `outputs/windows-split-kv/fp32-correct-reference-256`，原 FP32 对照为 `outputs/windows-split-kv/fp32-correct-reference-baseline`。

## 双 FP16、共享进程与故障恢复

FP32 GPT split-KV + FP16 声学也执行了独立自然随机轮，19 次请求全部正常结束，覆盖四类文本各 3 次热请求、五个参考音色和新 seed。热中位数为 215.38 / 1511.05 / 835.93 / 437.48 ms，完整音频为 3.46 / 25.90 / 14.42 / 7.66 秒，声学 token 数为 79 / 640 / 353 / 184。运行时源码未改变、没有运行错误；自然生成的输出长度与固定回放不同，不直接相减比较。记录位于 `outputs/windows-acoustic/fast-natural`，已并入完整请求数据。

GPT 也切换到 FP16 时，采用 baseline attention，未沿用上面的 FP32 split-KV 组合。独立自然随机轮完成 19 次请求，覆盖四类文本、五个参考音色和新 seed，均正常结束。热请求中位数分别为 176.15 / 2416.67 / 995.82 / 443.96 ms；完整音频长度为 3.46 / 25.46 / 14.26 / 7.66 秒，声学 token 数为 79 / 629 / 349 / 184。它们与固定回放的工作量不同，不能据此计算相对上表的加速比例，也没有逐 token 等价或音质验收结论。

双 FP16 的独立资源轮全卡初始样本为 2196 MiB、峰值 3523 MiB，差值 1327 MiB；主进程及子进程的主存 RSS 峰值约 2141.3 MiB。这一轮同样使用自然生成，不能把它与固定回放的 1624 MiB 直接相减，声称相同工作量节省了多少显存。GPT 权重与 KV 的明确容量变化另见 [GPT 精度实验](2026-09-20-windows-gpt-fp16.md)。

另外用实验 harness 将 ORT 声学与 GPT 放进同一进程，没有把共享进程接入产品运行路径：

| 组合 | 分进程初始 / 峰值 MiB | 同进程初始 / 峰值 MiB | 同进程峰值减初值 MiB | 分进程 / 同进程主存 RSS 峰值 MiB |
|---|---:|---:|---:|---:|
| FP32 GPT split-KV + FP16 声学，固定回放 | 2218 / 3842 | 2200 / 3742 | 1542 | 1993.7 / 1650.3 |
| FP16 GPT baseline + FP16 声学，自然生成 | 2196 / 3523 | 2201 / 3419 | 1218 | 2141.3 / 1558.0 |

两种同进程资源轮各有 7 个请求，分别与相同组合的分进程输出配对。参考、参数、音素、采样 token、停止和长度均一致，完整 PCM 逐位相同。表中是资源实验，不使用其请求耗时评估速度。共享进程后的错误隔离、依赖打包和常驻生命周期仍需单独验证；当前产品保留持久声学 worker。

持久 worker 的故障恢复已对两种组合分别执行真实 GPU 验证：常驻 worker 被终止后重试、分阶段运行在 GPT Prefill 后取消、在声学执行前取消，共 3 种情况。两组全部通过，重试后的 PCM 与各自正常执行逐位相同，busy 状态、模型释放和 worker 重建也通过检查。这里只覆盖一个中性短句；它证明同一精度路径可以恢复，不代表 FP16 与 FP32 等价。证据为 `outputs/windows-acoustic/lifecycle-fast/result.json` 和 `lifecycle-both-fp16/result.json`，完整报告已并入上述完整请求数据。

## 使用与证据

`scripts/convert_sovits_onnx_fp16.py` 创建候选，并将验证初始状态设为失败；`harness/windows_acoustic_precision.py` 完成工程筛查后才写入绑定图、权重和执行设置的结果。当前通过的候选位于 `outputs/windows-acoustic/fp16-candidate-lowered`。

最终复核还补了转换参数检查：`--lower-conv-transpose` 不能与 `--keep-fp32-op ConvTranspose` 或选中原转置卷积的 `--keep-fp32-node` 同用。改写会删除原算子，继续接受这类选项会让 FP32 保留请求静默失效，因此现在显式报错。两个拒绝路径及原有降级验证共 3 个局部测试通过。已测候选没有使用这些排除项，也未重新转换；候选中保留转换当时的脚本哈希，数据文件另记本次参数检查后的源码哈希。

```powershell
.venv/Scripts/python.exe -B scripts/convert_sovits_onnx_fp16.py `
  --source models/windows-sakura/sovits-onnx-v1 `
  --output outputs/windows-acoustic/fp16-candidate-rerun `
  --lower-conv-transpose --deterministic-compute

.venv/Scripts/python.exe -B harness/windows_acoustic_precision.py `
  --baseline models/windows-sakura/sovits-onnx-v1 `
  --candidate outputs/windows-acoustic/fp16-candidate-rerun `
  --runtime-python data/windows-ort-runtime/python.exe `
  --captures outputs/windows-baseline/replay-captures.json `
  --output outputs/windows-acoustic/fp16-screen-rerun --repeats 3
```

`ORTSoVITS.load` 和 `ORTProcessSoVITS` 需要显式传入 `allow_experimental_fp16=True`；持久 worker 使用 `--allow-experimental-fp16`。默认仍拒绝 FP16 包。候选加载绑定本次筛查的 CUDA/session 配置，不能更换工作区策略后沿用原验证结论。包的 `dtype=float16` 描述实际内部计算，`precision.keep_io_types=true` 描述 FP32 公共边界；验证种类为 `fp16-engineering-screen`，并不代表音质验收。

完整引擎参数为 `allow_experimental_acoustic_fp16=True`，CLI 对应 `--allow-experimental-acoustic-fp16`。实际 CLI 已使用原始日文文本、自然随机生成，输出 3.62 秒完整 PCM；报告确认 GPT FP32、声学 FP16、split-KV chunk 256，未导入 PyTorch。该次请求包含首次加载，不能与热耗时比较，也未做听音或 ASR。证据为 `outputs/windows-acoustic/combined-cli.json`。

仓库内的 [实验数据](data/2026-09-20-windows-acoustic-fp16.json) 保存最终转换记录、90 组 CPU 测试结果、输入与脚本哈希、失败候选摘要、完整 screen v2 和 profile 摘要。原始数组、日志和 profile 位于 `outputs/windows-acoustic/fp16-screen-lowered`。同目录的 `fp16-screen-default`、`fp16-screen-basic`、`fp16-screen-deterministic`、`fp16-screen-deconv-fp32` 保留各轮失败，不用新结论覆盖它们。

本实验未测精确进程显存、流式首包、人工听音、ASR、其他权重和干净机器部署。完整请求及资源测量应继续使用与原捕获一致的参考条件；不能用内容身份相同但数值不同的 CPU 参考替代 CUDA 捕获参考。

本轮代码合并后，开发环境执行 `python -m unittest discover -s tests` 共 141 项，139 项通过，2 项因 Windows 符号链接权限跳过。转换边界、加载准入、公共参数和生命周期回归都在这次检查中；真实 GPU 的数值、资源与恢复结果另按上文的 Harness 记录，不由 CPU 单测替代。
