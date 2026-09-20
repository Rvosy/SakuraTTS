# ORT 1.19.2 的 CPython 3.9 / 3.11 构建对照

日期：2026-09-20。使用相同 CUDA 参考后，cp311 共享进程与 cp39 独立 worker 各 15 次完整 FP32 请求均通过官方回放。此前长句 14 个、多句 170 个 PCM 样本超差来自混用 CPU / CUDA 准备的参考条件，不能据此拒绝进程整合；原始失败结果保留。共享进程另有两组资源对照，显示峰值降低，但卸载后的主存保留更多，尚未接入产品。

本报告通过 CPU 读取已有 GPU 结果、文件和上游源码，核对 PE 节哈希、数组传输和保存的 WAV，没有重跑 GPU 基准、改动依赖或调整容差。可提交的[汇总 JSON](data/2026-09-20-windows-ort-abi-comparison.json) 保存逐请求比较、资源边界和原始文件哈希。中文说明按 `humanizer-zh` 校订。

## 参考条件混配与已有结果

`outputs/windows-unified/fp32-replay` 的 cp311 共享进程和 `outputs/windows-split-kv/fp32-full-baseline` 的 cp39 独立 worker 均使用 `models/windows-sakura/runtime.json`，指向 CPU 准备的 `references/`。两轮 `run-config.json` 的四类 `reference_arrays_equal` 均为 `false`，原 harness 只记录这个值，没有阻止回放。

历史通过轮 `outputs/windows-cuda/timing-classic-replay-resident` 使用 `runtime-validation.json`，指向 `references-cuda/`，四类数组身份均相同。模型和参考音频身份相同，不代表不同设备准备的条件数组相同。

直接读取当前中性参考与官方短句捕获旁的参考数组，`reference_phones`、`prompt_semantic`、`reference_bert` 逐字节相同；CPU 准备的 `ge` 最大绝对差为 `2.0122528076171875e-4`，`ge512` 为 `8.20159912109375e-5`。CUDA 参考的全部数组均与捕获参考逐字节相同。

| 执行与配置 | 短句 | 长句 | 多句 | 标点 |
| --- | --- | --- | --- | --- |
| 历史 cp39，CUDA 参考 | 最大 1 LSB，通过 | 最大 2 LSB，通过 | 最大 3 LSB，通过 | 最大 1 LSB，通过 |
| 当前 cp39，CPU 参考 | 最大 1 LSB，通过 | 最大 5 LSB，14 个超差 | 最大 6 LSB，170 个超差 | 最大 3 LSB，通过 |
| cp311 共享进程，CPU 参考 | 最大 1 LSB，通过 | 最大 5 LSB，14 个超差 | 最大 6 LSB，170 个超差 | 最大 3 LSB，通过 |

当前 cp39 与 cp311 共享进程的四类热请求彼此最大均只差 1 LSB，全部通过原容差。声学 `load_events` 确认当前 cp39 为 Python 3.9.13、ORT 1.19.2、`float32`，不是误用了 FP16 包。

`harness/windows_nvidia_benchmark.py` 已将参考数组身份变成回放前置条件。混配会在导入和构造 GPU 引擎前失败，并明确提示 CPU / CUDA 准备结果不能互换。新增 CPU 测试使用同模型、同音频身份但不同 `ge` 的实际数组及哈希验证提前拒绝；真实配置的 `--check-only` 也确认 `runtime.json` 被拒绝、`runtime-validation.json` 四类通过，全程未导入 CuPy、ORT 或 NVIDIA 引擎。

## 原先记录的构建候选

旧环境为 `data/windows-ort-runtime` 的 CPython 3.9 / NumPy 1.23.4。候选 ORT 位于 `outputs/windows-unified/ort-python311`，主解释器为 CPython 3.11 / NumPy 2.4.6。

| 已有 GPU 实验 | 原 FP32 回放结果 |
| --- | --- |
| `outputs/windows-unified/fp32-replay`：CuPy 与 cp311 ORT 同进程 | 短句通过；长句最大 5 LSB、14 个样本超差；多句最大 6 LSB、170 个样本超差。 |
| `outputs/windows-unified/cp311-worker-replay`：相同 cp311 ORT，独立声学进程 | 多句三次均最大 6 LSB、170 个样本超差。 |

两组多句的音素、采样 token、返回语义、停止结果和 PCM 长度检查都通过。比较两组 `hot-neutral-multi-00.wav`，339,840 个 PCM 样本中有 63 个相差 1 LSB，最大差为 1 LSB；这只是候选间的 PCM 比较，未改变它们对原参考的失败结论。

两组使用相同的 `ort_sovits.py` 哈希和声学 manifest。生产图均为 `acoustic.onnx`，SHA256 `f85337dac20793c27150585ae4a50f5db6a25a15b01633a9ecd4e3b1c2f51b06`；没有把诊断图换成普通请求图。

进一步将四类现有固定声学输入经两套 NumPy 的 `ORTSoVITS._inputs()`、`array_protocol` 写入/读取及 worker 输入重建：六个 feed 的形状、dtype、内容 SHA256 和完整传输帧均相同，所有传输回读逐字节相同。`noise_scale=0.5` 也相同。该 CPU 检查排除了相同数组的类型转换和 pipe 编码差异，未验证完整请求是否选中了同一份参考；这正是随后发现的遗漏。

## 诊断图、生产图与 PCM 边界

读取此前固定捕获声学输入的 `outputs/windows-acoustic/fp16-screen-default/baseline-diagnostic` 和 `baseline-production`。这两组使用捕获参考，四类 FP32 波形的诊断 / 生产最大绝对差为 `1.192e-7` 至 `1.788e-7`，全部通过原容差。

将保存的生产图 FP32 波形经当前 `single_fragment_pcm()` 转成 PCM，四类对官方捕获的最大差分别为 1、2、3、1 LSB，全部通过；对历史完整请求的 PCM 最大均只差 1 LSB。这组证据不支持把 14 / 170 个超差解释为生产图切换或 PCM 转换造成。随后使用相同 CUDA 参考的完整链路对照也全部通过，确认了先前比较中的参考混配问题。

## 相同参考下的共享进程对照

新对照为 `outputs/windows-unified/split-fp32-correct-reference` 与 `outputs/windows-split-kv/fp32-correct-reference-256`：GPT FP32、Split-KV 256、声学 FP32、相同 CUDA 参考、相同官方随机输入，各运行 15 次，包括四类热请求与卸载后重载。

两轮配置、模型 manifest SHA256、推理参数和捕获身份相同，参考数组检查全为 `true`，运行期间没有源文件变化。双方各 15 次均通过官方原容差回放。独立读取 WAV 互比，15 对的音素、采样 token、语义 token、停止原因、返回位置、波形长度和 PCM 长度全部相同；每对 PCM 最大差为 1 LSB，均在原容差内，但不逐位相同。cp311 构建和共享进程本身没有造成先前所报的严格回放失败。

## 两组资源对照

两组均使用同一个已筛查的声学 FP16 包和相同 CUDA 参考；共享进程由 `harness/windows_shared_cuda_process.py` 提供，classic 日文前端仍有自己的 worker。

- GPT FP32 / Split-KV 256：`outputs/windows-unified/combined-shared-memory` 对 `outputs/windows-acoustic/combined-memory-fp32-gpt`，使用官方随机输入回放。
- GPT FP16 / baseline attention：`outputs/windows-unified/both-fp16-shared-memory` 对 `outputs/windows-acoustic/both-fp16-memory`，使用同 seed 的自然随机生成。

每组各 7 次完整请求。独立读取全部 WAV 并逐项核对，两组各 7 对的参数、参考身份、音素、采样与语义 token、停止原因、返回位置、输出形状和 PCM 均相同；14 对 PCM 全部逐位相等。回放组双方仍标记 `replay_validation_failed`，因为声学 FP16 未满足原 FP32 波形容差，这不因共享进程互比相同而变成原精度通过。自然生成组仅比较两个执行安排下的同工作量输出，没有新增官方 FP32 或音质验收。

以下显存为每轮全卡采样值，单位 MiB；增量分别减去各轮最早样本。空闲和卸载行使用对应边界最近的 GPU 样本，完整时间戳保留在汇总 JSON。

| GPT / 声学精度与执行安排 | 最早样本 | 全卡峰值 | 峰值增量 | 卸载前空闲增量 | 卸载后增量 | 引擎关闭后增量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 / FP16，独立 worker | 2218 | 3842 | 1624 | 1574 | 102 | 93 |
| FP32 / FP16，共享进程 | 2200 | 3742 | 1542 | 1542 | 128 | 122 |
| FP16 / FP16，独立 worker | 2196 | 3523 | 1327 | 1327 | 110 | 103 |
| FP16 / FP16，共享进程 | 2201 | 3419 | 1218 | 1218 | 112 | 113 |

共享进程的峰值增量分别少 82 MiB 和 109 MiB。它没有在所有生命周期边界都更省资源，尤其是主存：

| GPT / 声学精度与执行安排 | 进程树 RSS 峰值 MiB | 卸载前空闲 MiB | 卸载后 MiB | 引擎关闭后 MiB |
| --- | ---: | ---: | ---: | ---: |
| FP32 / FP16，独立 worker | 1993.7 | 1993.7 | 791.8 | 615.4 |
| FP32 / FP16，共享进程 | 1650.3 | 1650.2 | 1626.4 | 1451.8 |
| FP16 / FP16，独立 worker | 2141.3 | 2141.3 | 980.9 | 806.0 |
| FP16 / FP16，共享进程 | 1558.0 | 1557.9 | 1536.8 | 1361.9 |

共享进程的采样主存峰值分别低约 343.4 MiB、583.4 MiB；但显式卸载后，进程树 RSS 分别高约 834.6 MiB、555.8 MiB。引擎关闭后主进程仍存活，共享方案的 RSS 也没有降到独立 worker 的水平。本轮未逐项定位这些原生分配，不能直接称为泄漏；它说明独立 worker 退出能够回收的部分资源，在共享进程中仍会保留。

显存是 WDDM 下每 100 ms 采样的全卡数据，包含桌面负载并可能漏掉短峰值；进程树 RSS 可能重复计入共享页。资源轮的计时不用于宣称加速，wrapper 在计时前已经导入 ORT，加载时间也不能当作完整冷启动节省。共享方案尚未接入产品，尚未单独做共享进程故障注入；新的结果支持继续推进集成实验，也要求保留卸载后的主存权衡。

## 构建与加载情况

两套 `get_build_info()` 完全相同：ORT 1.19.2、`ffceed9d44`、Release，以及相同的 MSVC C++ flags。`version_info.py` 均记录 CUDA `12.2.128`、cuDNN `9`。这些元数据没有证明两份原生文件相同。

| CUDA provider 文件范围 | cp39 bytes | cp311 bytes | SHA256 是否相同 |
| --- | ---: | ---: | --- |
| `onnxruntime_providers_cuda.dll` 全文件 | 607,457,312 | 607,438,776 | 否 |
| `.text` 节 | 8,849,408 | 8,849,408 | 否 |
| `.nv_fatb` 节 | 562,454,528 | 562,436,096 | 否 |
| PE 签名区域 | 10,272 | 10,168 | 未单独比较 |

GPU fatbin 所在节本身相差 18,432 bytes，文件差异不能仅解释为签名尾部不同。`onnxruntime.dll`、pybind `.pyd` 也不同；shared provider 的 `.text` 相同。节差异可能包含代码、编译布局和元数据，不能单凭哈希认定某个浮点实现有错。参考条件混配排除前，这些差异不能用来解释本次失败。

共享进程的原生映射显示 cuBLAS、NVRTC、cudart 从主 venv 载入，cuDNN/cuFFT 从 worker 的 `cuda` 载入。[此前清点](2026-09-20-windows-runtime-inventory.md) 已核对两边五个重复 CUDA DLL 的完整 SHA256 相同，因此路径不同本身没有提供不同 CUDA 库内容的证据。独立 cp311 worker 实验的 `loaded_native_maps` 仍记录父进程映射，不能把它当作声学子进程 DLL 加载清单。

两个完整请求路径都调用相同的 `ORTSoVITS.load()` 默认参数：生产图、`ORT_ENABLE_ALL`、`HEURISTIC`、`kSameAsRequested`、`use_tf32=0`、`cudnn_conv_use_max_workspace=0`、禁用 memory pattern。当前 cp39 基线与 cp311 独立 worker 的 `load_events` 保存了实际 `provider_options`，完整字典相同。cp311 共享进程轮没有保存实际 session options，后续应补记。

核对构建提交的 [CUDA provider 源码](https://github.com/microsoft/onnxruntime/blob/ffceed9d44/onnxruntime/core/providers/cuda/cuda_execution_provider.cc)，ORT 为其线程 context 创建自己的 cuBLAS/cuDNN handle。GPT 的 cuBLAS handle 设置不能直接推断成 ORT 的全局数学模式。该版本 [Conv 实现](https://github.com/microsoft/onnxruntime/blob/ffceed9d44/onnxruntime/core/providers/cuda/nn/conv.cc) 在 FP32 且关闭 TF32 时设置 `CUDNN_FMA_MATH`；`HEURISTIC` 路径调用 `cudnnGetConvolutionForwardAlgorithm_v7`。基于可用显存选最大搜索 workspace 的代码位于另一条 `EXHAUSTIVE` 路径，不能未经验证就用它解释本次 `HEURISTIC` 偏差。

## 下一步与证据位置

相同参考的完整 FP32 回放和两组共享资源对照已经完成。接下来应把实验 wrapper 中的加载安排整理成明确的运行路径，补齐共享进程故障恢复与清理验证，再根据常驻和卸载场景选择默认策略。没有必要仅因旧的 14 / 170 个超差去替换 CUDA provider DLL，也没有依据放宽原容差。

证据位于 `outputs/windows-unified/ort-build-comparison/`：`report.json` 保存 CPU 导入与 PE 清单，`numpy-feed-comparison.json` 保存四类完整输入哈希，`reference-boundary-comparison.json` 保存参考身份及旧 / 新完整 PCM、诊断 / 生产波形比较，`upstream.json` 保存上游源码和 PyPI 元数据。`summarize_shared_process.py` 仅用 CPU 读取六轮原始记录与 WAV，生成本报告链接的汇总 JSON。各轮 GPU 失败与通过结果均未改写。
