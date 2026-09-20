# Windows 声学相位打包与请求后显存回收

本轮去掉了声码器上采样中的零插值大张量，并验证了请求结束后收缩 ORT arena 的组合。在独立声学进程中，polyphase 候选的长句后 Dedicated Usage 从不收缩时的 783.52 MiB 降到每次收缩时的 259.52 MiB；后续短句也保持在 259.52 MiB。它测量的是选定边界的进程归属计数，不是完整引擎的独占显存，也没有测到执行中的峰值。

新的相位打包候选通过了独立工程筛查，但相对旧 FP16 零插值实现仍有浮点差异，超出原严格容差。完整请求的长句耗时与此前快路径接近，本轮不能作为稳定加速的证据。默认模型精度与 arena 行为保持不变，候选和回收功能均需显式启用。

## 环境与改写范围

本轮使用 RTX 5060 8 GB、Sakura V2ProPlus、Windows 11 build 22631，以及独立声学运行时中的 ORT 1.19.2、Python 3.9.13、NumPy 1.23.4。普通推理没有导入 PyTorch。CUDA 使用 `HEURISTIC`、`kSameAsRequested`、关闭最大卷积工作区和 memory pattern，4 个 intra-op 线程。

此前的[声学 FP16 实验](2026-09-20-windows-acoustic-fp16.md)将 5 个 ConvTranspose 改为零插值与普通 Conv，以避开同一 Session 内的波形波动。本次新增 `tools/conv_transpose_polyphase.py`，直接把各输出相位打包到普通卷积的通道维，再交错恢复时间轴。

| 上采样层 | 输入 / 输出通道 | 原 kernel / stride | 原 padding | 打包后 Conv kernel |
| --- | ---: | ---: | ---: | ---: |
| 0 | 768 / 384 | 20 / 10 | 5 / 5 | 3 |
| 1 | 384 / 192 | 16 / 8 | 4 / 4 | 3 |
| 2 | 192 / 96 | 8 / 2 | 3 / 3 | 5 |
| 3 | 96 / 48 | 2 / 2 | 0 / 0 | 1 |
| 4 | 48 / 24 | 2 / 2 | 0 / 0 | 1 |

对输出 `y[j*S+r]`，原权重索引是 `(j-i)*S+r+pad_left`。实现按模 `S` 的余数取核元素，再翻转该相位的核用于普通互相关；不同相位的 padding 用打包权重中的零对齐。Conv 输出形状为 `(N,Cout*S,L)`，经 `(N,Cout,S,L) → (N,Cout,L,S) → (N,Cout,L*S)` 恢复输出。偏置按相位重复，不改变输出长度或首尾采样。

当前 helper 只接受 group=1、dilation=1、无 output_padding 且输出长度为 `L*S` 的一维算子。它没有扩大模型兼容范围。前三层打包权重分别增加 50%、50%、25%；候选的 `weights.bin` 从 124,183,808 增到 131,346,176 bytes，多约 6.83 MiB。减少零插值计算不等于所有资源都减少。

## 正确性与候选边界

CPU 独立 scatter 验证包含 6 组 stride/kernel/padding、5 种长度，以及随机、首端和尾端脉冲，共 90 组；另补动态 batch、无偏置和共享权重保留的 4 组数值对照。5 个测试方法全部通过，合计 94 次数值对照；还检查了遇到后置非法算子时不会部分修改原图。

转换为 FP16 之前，完整 FP32 改写图对源包 4 个保存用例、48 个输出阶段沿用 `atol=1e-4, rtol=1e-5`，全部通过，最大绝对误差约 5.05e-5。

随后对固定语义、参考、音素和声学噪声进行 GPU screen v2。最终候选采用 `use_deterministic_compute=true`，4 类输入全部通过此前已确定的工程门槛：最大误差 0.05、RMSE 0.005、SNR 至少 25 dB，并检查幅度和频谱。候选重复执行、生产图与诊断图、带 profile 与不带 profile 的波形都逐位相同；profile 未发现 CPU 神经计算 fallback。相对 FP32 的最大误差约 0.03843，最低 SNR 约 39.54 dB。原 FP32 容差仍失败，人工听音和 ASR 未验收。

另将最终候选与旧零插值 FP16 生产输出直接比较，保留相同精度下的严格结果：

| 输入 | 最大绝对误差 | RMSE | 原严格容差 |
| --- | ---: | ---: | --- |
| 短句 | 0.00073242 | 7.42e-5 | 失败 |
| 长句 | 0.00048828 | 4.26e-6 | 失败 |
| 多句 | 0.00105286 | 6.94e-5 | 失败 |
| 标点 | 0.00097656 | 7.10e-5 | 失败 |

因此 polyphase 保留为单独筛查过的实验候选，不能称为旧 FP16 路径的严格等价替换。它通过自己的重复性检查，也不能据此抹掉跨方法的误差。

## 实际执行图与内存来源

ORT profile 确认 5 组节点仍以 CUDA FP16 `Conv → Reshape → Transpose → Reshape` 执行，没有被优化器还原成 ConvTranspose。每次请求仍有 20 个相关节点事件，与此前 `Unsqueeze → Pad → Reshape → Conv` 的数量相同。

固定长例使用 675 个语义 token、1350 个声学帧。旧图最大相关输出是 82,944,000 bytes，约 79.10 MiB，来自零插值 Pad/Reshape；新图最大输出是 41,472,000 bytes，约 39.55 MiB。Reshape 可能复用存储，不能把这些输出尺寸相加当作分配量或峰值。

profile 能确认拓扑、设备和张量形状，不能解释全部耗时。初次 polyphase 长句的 ORT 节点事件合计约 31 ms，而 `model_run` 约 241 ms；这些主机事件不是逐个同步测得的 CUDA kernel 时间。现有证据不足以把速度结果归因于大输出通道、交错拷贝或某个 cuDNN 算法。

## 相同输入下的 arena 对照

使用 `research/tools/windows_acoustic_arena.py`，每个候选、每个策略都启动新进程，执行“短 → 长 → 短 → 多句 → 标点”。`off` 不收缩，`after-long` 仅在名为 long 的测试请求结束后收缩，`always` 每次收缩；`after-long` 不是产品中的自动长度阈值。原理与零插值候选的独立结果见 [arena 实验](2026-09-20-windows-acoustic-arena.md)。

资源轮每个边界等待 150 ms 再采样。下表只列声学进程、同一 LUID 的 Dedicated Usage，单位 MiB；不跨 PID 求和。

| 候选 / 策略 | 加载后 | 首短后 | 长句后 | 再短后 | 多句后 | 标点后 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 零插值 / off | 241.52 | 303.52 | 911.52 | 911.52 | 911.52 | 911.52 |
| 零插值 / after-long | 241.52 | 303.52 | 509.52 | 509.52 | 541.52 | 541.52 |
| 零插值 / always | 241.52 | 281.52 | 509.52 | 347.52 | 381.52 | 419.52 |
| polyphase / off | 247.52 | 291.52 | 783.52 | 783.52 | 783.52 | 783.52 |
| polyphase / after-long | 247.52 | 291.52 | 259.52 | 291.52 | 433.52 | 433.52 |
| polyphase / always | 247.52 | 259.52 | 259.52 | 259.52 | 259.52 | 259.52 |

各进程在加载前没有 GPU Process Memory 实例；这是缺失状态，不按零解释。显式卸载后，各轮仍有约 105.51 MiB 的进程计数，CUDA context 随进程继续存在。本轮没有将它认定为泄漏。

单独的无资源采样计时轮先执行一轮预热，再测三轮相同序列。短句每轮出现两次，因此 n=6，其余 n=3。以下是 polyphase 候选的 `session.run` 中位数，包含 CPU 输入、完整 CPU 波形返回与选定的回收动作，不含前端、GPT、PCM 转换或写盘。

| 策略 | 短句 ms | 长句 ms | 多句 ms | 标点 ms |
| --- | ---: | ---: | ---: | ---: |
| off | 26.93 | 221.13 | 83.91 | 63.56 |
| after-long | 28.34 | 227.95 | 85.34 | 73.41 |
| always | 29.01 | 228.29 | 88.17 | 71.75 |

回收确实有代价：这轮 always 的长句中位数增加约 7.16 ms，短句约 2.08 ms；标点增加约 8.19 ms。样本较少，不能视为稳定上界。两种候选的资源与计时实验合计 150 次声学执行，波形和最终 PCM 均与各自已筛查输出逐位相同；100 对跨策略输出比较也全部逐位相同。这个结论只比较同一候选内部的收缩策略，不跨候选混用。

两候选使用相同输入哈希、ORT/NumPy、provider 设置及确定性设置。跨候选时 `ort_sovits.py` 的哈希有所变化：期间新增了默认关闭的公开回收参数。已核对改动，探针仍按默认参数加载，并直接使用自己的 RunOptions 调用 `session.run`，没有经过新增的 `decode` 分支。数据中保留两个源码哈希，没有写成“所有源码完全相同”；每个候选内部的三策略则使用相同源码，运行过程中均未变化。

## 完整请求验证

公开入口已增加 `--acoustic-arena-shrink`，默认关闭。FP32 GPT、split-KV chunk 256、polyphase FP16 声学和每次 arena 收缩组合执行了 15 个完整请求，包含首次请求、四类文本各 3 次热请求，以及卸载后重载。

独立读取这 15 个 WAV，并分别与原 FP32 基线和此前“FP32 GPT split-KV + 零插值 FP16 声学”配对。45 个 WAV 的 PCM 哈希与报告一致；30 对比较的参考与捕获身份、参数、音素、采样和语义 token、停止原因、返回索引、波形和 PCM 长度全部一致。

完整 int16 PCM 除以 32768 后沿用同一工程门槛，30 对全部通过；去掉记录中的固定 0.3 秒尾部静音后也全部通过，没有裁掉语音。相对原 FP32 最大误差为 0.03842、最低 SNR 为 39.55 dB；相对旧快路径分别为 0.001038、62.39 dB。两组各 15 对原严格容差均失败，保留 `replay_validation_failed`，没有把工程筛查结果替换成严格等价。

| 输入 | 完整音频 s | 原 FP32 ms | 旧零插值 FP16 快路径 ms | polyphase + 收缩 ms |
| --- | ---: | ---: | ---: | ---: |
| 短句 | 2.02 | 168.94 | 122.23 | 126.01 |
| 长句 | 27.30 | 3045.15 | 1592.30 | 1597.12 |
| 多句 | 10.62 | 932.13 | 651.49 | 603.90 |
| 标点 | 8.38 | 691.01 | 469.74 | 474.38 |

表中为完整请求热中位数，n=3，无持续资源采样。与旧快路径相比，长句没有继续变快，短句和标点略慢，多句本轮较快。这些不同时间运行的小样本不足以支持统一加速比例；主要结果仍是请求后回收量。也不能把相对原 FP32 的全部提升归因于本次改写，因为旧快路径已经包含 GPT split-KV 和声学 FP16。

同一配置另跑 7 个资源请求，逐个与计时轮配对。配置哈希、精度、attention、容量、回收选项与捕获输入相同；参数、token、停止与长度均一致，7 对完整 PCM 逐字节相同。资源轮有 150 个样本，全卡初始值 2213 MiB，采样峰值 3721 MiB，差值 1508 MiB；进程树主存 RSS 峰值约 1981.63 MiB。其请求耗时不并入计时轮。

| 同一配置的独立计时 / 资源结果 | 完整长音频 s | 热长句 ms | 全卡采样峰值减初值 MiB |
| --- | ---: | ---: | ---: |
| 原 FP32 | 27.30 | 3045.15 | 1804 |
| 旧 FP32 GPT split-KV + 零插值 FP16 声学 | 27.30 | 1592.30 | 1624 |
| FP32 GPT split-KV + polyphase FP16 声学 + 收缩 | 27.30 | 1597.12 | 1508 |

三组使用相同固定回放，长句都是 675 个语义 token。全卡数值包含桌面活动，采样也可能漏掉短峰，不能解释为每个进程的独占量。这里没有混入双 FP16 自然生成或共享进程的数据；那些配置的计时、输出长度和显存应分别配对，详见 [arena 集成结果](2026-09-20-windows-acoustic-arena.md)。

## 使用与证据

转换脚本接受 `--lower-conv-transpose polyphase --deterministic-compute`；原先不带值的 `--lower-conv-transpose` 仍选择零插值。最终候选是 `outputs/windows-acoustic/fp16-candidate-polyphase-deterministic`，screen 位于 `outputs/windows-acoustic/polyphase-deterministic-screen`。候选 manifest 保存转换器、polyphase helper、图和权重的哈希。

当前可用配置是 `outputs/windows-acoustic/runtime-fp16-polyphase-deterministic.json`。CLI 显式传入 `--allow-experimental-acoustic-fp16 --acoustic-arena-shrink` 才启用对应行为；公开布尔选项每次声学 decode 后收缩，没有自动长度选择策略。候选仍需绑定自身图、权重和执行设置的 screen v2，不能靠另一个候选的验证文件绕过准入。

初次 polyphase screen 和 arena 实验保留在 `polyphase-screen`、`arena-polyphase-memory`、`arena-polyphase-timing`。当时确定性设置为 false，而旧候选为 true，因此没有用它们作为最终匹配对照。后来重新转换、筛查并运行 `arena-polyphase-deterministic-memory` / `arena-polyphase-deterministic-timing`；新旧 polyphase 的四类生产波形逐位相同，但仍保留最初的设置差异。

早期完整引擎阶段探针 `outputs/windows-wddm-memory/polyphase-both-fp16-worker` 的文件缺失失败也保留，后续 `-v2` 成功记录没有覆盖它。该探索轮与旧轮的三个自然请求 token 和长度相同，只有两个短句 PCM 哈希相同，长句不同；它们不替代上述固定声学输入和完整 WAV 对照。

[实验数据](data/2026-09-20-windows-acoustic-polyphase.json)保存转换与筛查摘要、跨方法严格失败、原始报告与 profile 哈希、匹配 arena 结果，以及 30 对完整 PCM / 去静音指标。数据提取脚本位于 `outputs/windows-acoustic/summarize_polyphase.py`。进程计数的解释见 [WDDM 采样实验](2026-09-20-windows-wddm-memory.md)，公开入口的资源轮和故障恢复见 [arena 实验](2026-09-20-windows-acoustic-arena.md)。

本轮尚未验证人工听音、ASR、流式首包、其他模型 / GPU、干净机器部署或执行中的精确显存峰值。259.52 MiB 只属于声学进程的特定请求后边界，不能据此宣布完整引擎已达到 Lite 的 0.8 GB 目标。
