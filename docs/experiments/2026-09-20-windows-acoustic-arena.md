# Windows 声学 arena 回收实验

日期：2026-09-20。对已通过工程筛查的零插值卷积 FP16 声学包启用 ORT arena 回收后，长句结束时声学进程的 WDDM Dedicated Usage 从 911.52 MiB 降到 509.52 MiB。每次请求都回收时，后续短句结束后降至 347.52 MiB。无资源采样计时轮中，长句中位耗时从 231.65 ms 增至 237.81 ms，约增加 2.66%。

75 次请求的完整声学波形与最终 PCM 均与已筛查 FP16 候选逐位相同。这个结论说明回收选项在本轮固定条件下没有改变输出；原候选的 FP32 容差失败、听音和 ASR 待验收状态仍然保留。后续完整引擎验证另列在文末，精确进程峰值、其他模型和设备仍未验证。

[汇总数据](data/2026-09-20-windows-acoustic-arena.json)保存原始结果哈希、执行源码哈希、全部请求与比较、逐 PID / LUID 边界字节值及计时样本。整理这份报告时只读取结果和 NPZ，没有重复运行 GPU。

## 选项与测量边界

[ORT 1.19.2 对应提交的头文件](https://github.com/microsoft/onnxruntime/blob/ffceed9d44/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h)定义 `memory.enable_memory_arena_shrinkage`，支持值 `gpu:0`，默认值为空。[官方说明](https://onnxruntime.ai/docs/get-started/with-c.html#features)指出，arena 默认不回收；指定 Run 结束时可检查并释放空闲区域，用于偶发大动态请求抬高后续保留量的情况。

[同版本 BFCArena 实现](https://github.com/microsoft/onnxruntime/blob/ffceed9d44/onnxruntime/core/framework/bfc_arena.cc)在当前 `kSameAsRequested` 策略下检查全部分配区域，包括第一个区域。区域内还有活跃 chunk 时不会释放。因此该选项首先影响请求后的保留量，不能据此承诺降低执行中的真实峰值。

`harness/windows_acoustic_arena.py` 在六个新进程中运行两个独立轮次，每轮分别测试三个策略：

| 策略 | 传给声学 Run 的选项 |
| --- | --- |
| `off` | 不传 RunOptions |
| `after-long` | 仅名为 `long` 的固定案例传入 `gpu:0` 回收选项 |
| `always` | 每次声学 Run 都传入 `gpu:0` 回收选项 |

`after-long` 是实验中的案例选择规则，没有实现自动判断长句的长度阈值。原报告每个策略都有相同 `run_option` 元数据，是否实际使用须看逐请求的 `shrink` 字段；`off` 的所有请求均为 false。

三种策略都执行“短句 → 长句 → 短句 → 多句 → 标点”。显存轮每策略执行一次序列，共 15 次声学请求；没有计时合格样本。计时轮每策略先预热一次序列，再测三次序列，共 60 次请求，其中 15 次预热、45 次正式计时。每策略短句有 6 个计时样本，其余案例各 3 个。

计时范围是已准备 CPU 输入进入 `session.run()`，直至完整 CPU 波形返回，包含选中的 arena 回收，不含前端、GPT、模型加载、PCM 转换、NPZ 写盘或跨进程 IPC。输入整理和 RunOptions 构造也在计时外。计时轮不启用 WDDM 采样，显存轮在边界等待 150 ms 后读取 PDH。

模型为 Sakura V2ProPlus，包位于 `outputs/windows-acoustic/fp16-candidate-lowered`；使用 Python 3.9.13、NumPy 1.23.4、ORT 1.19.2 CUDA，六个进程均未导入 PyTorch。GPU 和系统背景见[此前 WDDM 实验](2026-09-20-windows-wddm-memory.md)。四类输入、参考结果、图和权重、运行版本、provider 选项以及执行源码哈希在六个进程间一致，运行期间未记录源码变化。

原 FP16 包使用 `ORT_ENABLE_ALL`、确定性计算、4 个 intra-op 线程、禁用 memory pattern；CUDA 继续使用 `HEURISTIC`、`kSameAsRequested`、关闭最大卷积工作区和 TF32。此次只增加每次 Run 的选项，没有改包或已筛查的 Session / provider 设置，也没有重新进行 profile。FP16 图的工程准入证据见[声学 FP16 实验](2026-09-20-windows-acoustic-fp16.md)。

## 请求结束后的保留量

显存轮的 PID 为 `off=35652`、`after-long=45588`、`always=47864`。三者有效 GPU 进程计数均属于 LUID `0x00000000_0x0001df70`、physical adapter 0；未按适配器枚举顺序推断 CUDA ordinal，也没有跨 PID 或 LUID 求和。

下表是各自进程的 Dedicated Usage，单位 MiB：

| 边界 | `off` | `after-long` | `always` |
| --- | ---: | ---: | ---: |
| 加载后 | 241.52 | 241.52 | 241.52 |
| 第一个短句后 | 303.52 | 303.52 | 281.52 |
| 长句后 | 911.52 | 509.52 | 509.52 |
| 第二个短句后 | 911.52 | 509.52 | 347.52 |
| 多句后 | 911.52 | 541.52 | 381.52 |
| 标点后 | 911.52 | 541.52 | 419.52 |
| 卸载 Session 后 | 105.51 | 105.51 | 105.51 |

两种回收策略在长句结束后均比 `off` 少 402 MiB。`always` 在第二个短句后比 `off` 少 564 MiB，比 `after-long` 少 162 MiB。仅在长句后回收仍会保留后续请求产生的新区域；每次回收的结果也没有恒定回到首个短句的水平。本轮没有逐区域 allocator 日志，不能把表中的每一项增减归因到某个算子或具体 buffer。

加载前均没有匹配的 GPU 进程实例，记作缺失，不是零字节。三个进程的 Shared Usage 在加载及请求后均为 76 MiB，卸载后均为 74 MiB；模型卸载时进程仍活着，CUDA context 等分配仍可存在。Total Committed 单独保存在 JSON，不视为实际驻留量。

这些是声学进程的边界计数，没有包含 GPT 或前端，也不是进程独占 VRAM。[Microsoft 对 GPU 计数器的说明](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)指出，进程值还包含跨进程共享分配，跨 PID 相加会重复计数。这里不能将 347.52 MiB 或 509.52 MiB 宣称为 SakuraTTS 的完整峰值，更不能据此认定已经达到 Lite 的 0.8 GB 目标。

## 无采样计时

下表单位 ms，格式为中位数（最小值–最大值）。短句合并同一序列的两个位置；每个进程先经历完整序列预热，长句后的保留状态也会影响后续轮次。

| 案例 | n / 策略 | `off` | `after-long` | `always` |
| --- | ---: | ---: | ---: | ---: |
| 短句 | 6 | 27.43（26.67–42.70） | 28.40（26.99–36.83） | 28.18（27.95–29.34） |
| 长句 | 3 | 231.65（231.17–231.80） | 233.51（232.15–242.87） | 237.81（234.74–245.60） |
| 多句 | 3 | 91.09（86.83–97.83） | 88.59（88.03–89.83） | 89.74（88.38–93.79） |
| 标点 | 3 | 66.11（65.50–66.15） | 67.41（66.47–80.18） | 67.52（67.20–67.78） |

`always` 的长句中位数增加 6.16 ms，约 2.66%；短句增加约 0.75 ms。`after-long` 的长句中位数增加 1.87 ms，约 0.81%。每策略只有一个新进程，顺序固定，样本有限；不能将多句中位数的小幅下降解释为回收带来的确定加速，也不能从两个不同进程的中位数精确分离释放、重新分配和计算的各项耗时。

## 输出验证与复现

四类原始声学输出分别有 55,040 / 864,000 / 330,240 / 258,560 个 FP32 样本，现有 PCM 转换追加 0.3 秒静音后分别有 64,640 / 873,600 / 339,840 / 268,160 个 int16 样本。所有比较覆盖完整数组和首尾，没有裁切或跳过静音。

75 次请求均与 `fp16-screen-lowered/candidate-production` 的对应结果逐位一致。两种回收策略又与各自轮次中新进程运行的 `off` 按请求名比较，共 50 对：显存轮 10 对，计时轮 40 对。这些配对使用同一批输出，不增加请求次数。整理报告时重新读取全部 NPZ，逐项核对 dtype、形状、原始字节和记录的 SHA256，结果一致。

原 JSON 中的 `waveform_original_tolerance` / `original_tolerance` 沿用原容差数值，但这里的对照是 FP16 已筛查候选或新进程 `off`；它们不是对原 FP32 波形的新增验收。候选原有的 FP32 容差失败没有被回收选项消除。

原始结果及 SHA256：

- `outputs/windows-acoustic/arena-lowered-memory/result.json`：`0187cd5207cb8f0d43fc040e49ee08e6ec8f40a49c08c2dc9552a84fea656261`。
- `outputs/windows-acoustic/arena-lowered-timing/result.json`：`4d514c57aa1aa9036164b5b3e071f591131c67a88c3bb1f576258d90b18a1cf4`。
- 包 manifest：`7fd8cb0160c24939f5d46b0c3426e597036f4c0435e37f3bcbb3c553b028ad18`。
- 执行时 arena harness：`d929a138883431520153a6ceac57e84119dee08f29a882f261aa076a5ed3dfca`。

其余执行源码哈希见汇总 JSON 的 `common_controls.source_sha256`。运行后源码可能继续变化，应以这些哈希解释已有结果，不能用新源码身份替换历史记录。

复现时使用新输出目录，显存与计时分别运行：

```powershell
python -B harness/windows_acoustic_arena.py `
  --runtime-python data/windows-ort-runtime/python.exe `
  --package outputs/windows-acoustic/fp16-candidate-lowered `
  --inputs outputs/windows-acoustic/fp16-screen-lowered/inputs.json `
  --reference-output outputs/windows-acoustic/fp16-screen-lowered/candidate-production `
  --output outputs/windows-acoustic/arena-lowered-memory-rerun `
  --mode memory --repeats 1

python -B harness/windows_acoustic_arena.py `
  --runtime-python data/windows-ort-runtime/python.exe `
  --package outputs/windows-acoustic/fp16-candidate-lowered `
  --inputs outputs/windows-acoustic/fp16-screen-lowered/inputs.json `
  --reference-output outputs/windows-acoustic/fp16-screen-lowered/candidate-production `
  --output outputs/windows-acoustic/arena-lowered-timing-rerun `
  --mode timing --repeats 3
```

运行时入口保留默认关闭，显式 `acoustic_arena_shrink=True` / CLI `--acoustic-arena-shrink` 对应每次声学 Decode 后回收。FP16 包仍需单独开启实验性精度准入，回收选项不替代该检查。本报告中的 GPU 结果来自 harness 直接传 RunOptions；完整引擎的参数传递、恢复和显存效果须按各自验证记录判断。

后续若目标是降低真正执行峰值，应继续定位声学激活与卷积工作区，再验证 polyphase 卷积或有限感受野的 vocoder 分块。arena 回收的收益已足以支持一个明确的可选项，但还不支持修改默认策略或宣称完整引擎达到固定显存目标。

## 后续：公共入口与 polyphase 组合

上述 75 次请求使用零插值卷积包，以下组合改用[polyphase 声学包](2026-09-20-windows-acoustic-polyphase.md)，同时通过公共参数开启 arena 回收。两组证据分别保存；组合结果不用于单独计算 arena 或 polyphase 的收益。[集成数据](data/2026-09-20-windows-acoustic-arena-integration.json)保存运行结果、配置、WAV 和采样文件哈希，以及各轮记录的执行源码身份。

公共参数已贯通 CLI、完整引擎、直接 ORT loader、持久声学 worker，以及共享进程和 WDDM 实验 wrapper。默认值仍为 false；CPU 路径拒绝该选项，FP16 的单独准入检查和 Session / provider 设置保持原样。此处使用的是每次声学 Decode 后回收的 `always` 语义。

先用原 FP32 声学图和 FP32 GPT 开启回收执行固定捕获回放。`outputs/windows-acoustic/shrink-fp32-original-replay` 的 7 次请求全部通过原参考检查；整理报告时重新读取捕获 WAV 和输出 WAV，完整 PCM 仍在原 `atol=1e-4, rtol=1e-5` 内，最大差为 3 LSB。它验证了原 FP32 路径可通过公共开关运行，不代表与官方逐位相同。

### 双 FP16 完整请求与资源

以下均使用 GPT FP16 baseline attention、capacity 2048、单步 CUDA Graph、polyphase 确定性声学包和 arena 回收，自然随机生成，seed 1234。相同的数值参考、请求参数、前端、语义 token 和停止条件均已配对核对。

独立 worker 的无采样计时轮位于 `outputs/windows-acoustic/polyphase-shrink-both-fp16-timing`，完成 15 次请求，四类各 3 次热请求，另有首次请求和卸载后的两次请求。热中位数如下：

| 案例 | 完整 PCM 秒数 | 语义 token | 热请求中位数 ms |
| --- | ---: | ---: | ---: |
| 短句 | 3.46 | 79 | 183.44 |
| 长句 | 25.46 | 629 | 2390.18 |
| 多句 | 14.26 | 349 | 995.71 |
| 标点 | 7.66 | 184 | 461.07 |

这里包含原始文本到完整 PCM，不包含 WAV 写盘。不能与固定回放的不同音频长度直接相减，声称完整请求加速多少。

独立资源轮位于 `outputs/windows-acoustic/polyphase-shrink-both-fp16-memory`；共享进程资源轮位于 `outputs/windows-unified/polyphase-shrink-both-fp16-memory`，各完成 7 次请求。两轮逐请求完整 PCM 字节相同，也分别与上述计时轮中同名的 7 次请求逐字节相同。各轮没有运行错误或执行期间源码变化。

资源轮的全卡 `nvidia-smi` 采样和主进程及子进程 RSS 如下，单位 MiB：

| 组合 | 全卡最早样本 | 全卡采样峰值 | 峰值减最早样本 | CPU tree RSS 采样峰值 |
| --- | ---: | ---: | ---: | ---: |
| 旧零插值 FP16，独立 worker，关闭回收 | 2196 | 3523 | 1327 | 2141.30 |
| polyphase FP16，独立 worker，开启回收 | 2178 | 3358 | 1180 | 2130.54 |
| 旧零插值 FP16，共享进程，关闭回收 | 2201 | 3419 | 1218 | 1557.95 |
| polyphase FP16，共享进程，开启回收 | 2168 | 3246 | 1078 | 1541.05 |

两次组合改动后的全卡采样差值分别少 147 MiB 和 140 MiB。这些值包含桌面活动，采样可能错过短暂峰值；不等于进程独占显存，也不是 WDDM 边界计数。资源轮的耗时不用于上面的速度表；共享 wrapper 预先导入 ORT，也不能用这两轮评估冷启动速度。

旧、新两组都使用相同请求和 GPT token，但更换声学图后并非所有 PCM 都逐位一致。每种进程模式的 7 对中，4 个短句请求逐位相同；长句、多句、标点分别有最大 8 / 32 / 32 LSB 差异，原严格容差失败。完整 PCM 和去除现有尾部静音后的语音段都通过原工程门槛，最大归一化差为 0.0009766，最低 SNR 为 62.00 dB。这个较小误差属于图改写组合的比较，不能覆盖前文 arena 单项逐位一致的边界，也不能替代听音。

共享进程的主存释放仍有代价：独立 worker 卸载后 CPU tree RSS 为 980.92 MiB，关闭后为 806.55 MiB；共享进程分别为 1532.26 MiB 和 1358.30 MiB。两者卸载后 CuPy used / total 均为零。共享进程继续属于实验路径，没有因此替换产品的持久声学 worker。

### 公共开关下的 WDDM 边界

`outputs/windows-wddm-memory/polyphase-shrink-both-fp16-worker/result.json` 再次执行“短 → 长 → 短 → 卸载 → 关闭”，三次请求的 PCM 哈希均与上述独立资源轮对应结果相同。声学 PID 为 8292，主进程 PID 为 8660；各自 Dedicated Usage 如下，单位 MiB：

| 边界 | GPT 主进程 | 声学 worker |
| --- | ---: | ---: |
| 加载后 | 263.53 | 247.52 |
| 短句后 | 377.53 | 259.52 |
| 长句后 | 431.53 | 259.52 |
| 再次短句后 | 431.53 | 259.52 |
| 卸载后 | 101.53 | 已退出，实例缺失 |
| 关闭后 | 101.53 | 已退出，实例缺失 |

有效行仍位于 LUID `0x00000000_0x0001df70`、physical adapter 0；前端没有 GPU 计数实例，不当作零字节，也不与上述进程相加。长句后声学保留量没有延续增长，但 GPT pool 在长句后仍保留约 67.69 MiB 空闲容量。

同轮独立读取的全卡初始值为 2155 MiB，长句后为 2844 MiB，差值 689 MiB。这个数字是请求结束后的全卡边界差；上面的连续资源轮仍测到 1180 MiB 的峰值差，不能把 689 MiB 当作完整引擎的峰值或 Lite 0.8 GB 目标已完成。

### 恢复与 CLI

`outputs/windows-acoustic/polyphase-shrink-lifecycle/result.json` 使用真实 GPU 双 FP16 polyphase 包和公共回收开关，通过 3 个故障恢复案例：终止常驻声学 worker 后重试、GPT Prefill 后取消再重试、进入声学计算前取消再重试。busy、模型释放、worker 重建及重试结果检查全部通过，3 次重试 PCM 均与各自正常请求逐位相同。覆盖范围仍是一个中性短句，没有测试共享进程的故障隔离。

真实 CLI 自然生成也完成，结果为 `outputs/windows-acoustic/polyphase-shrink-cli.json` 与同名 WAV。报告确认 GPT FP16、声学 FP16、baseline attention、arena 回收开启且未导入 PyTorch，输出 3.46 秒 PCM。WAV 哈希已重新核对，完整 PCM 与独立 worker 首次短句逐字节相同。CLI 报告保存配置和 WAV 身份，没有记录执行源码哈希，因此不为它补写推断的历史源码身份，也不从这一次含加载请求推导热性能。
