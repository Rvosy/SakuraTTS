# Radeon 780M：CPU 与 DirectML 推理实测

2026-09-27，在 Ryzen 7 7840HS / Radeon 780M 上，SakuraTTS 的 CPU 与 DirectML 路径均完成了实际语音生成。使用同一个原生模型、参考条件和采样参数，DirectML 的短句热请求中位数为 **2.699 秒**，CPU 为 **4.389 秒**；长句分别为 **18.235 秒**和 **29.125 秒**。这两条输入的完整生成耗时减少约 **37.4%–38.5%**，声学阶段耗时降至 CPU 的约五分之一。GPT 仍在 CPU 执行，DirectML 会增加内存占用。

正文采用最终测量，每条输入、每个后端各有两次热请求。[最终数据](../experiments/data/2026-09-27-cpu-directml-780m-final.json)保存 21 次请求、各阶段耗时、音频与输入身份、GPU 验证、进程生命周期和资源补测；[首轮数据](../experiments/data/2026-09-27-cpu-directml-780m-initial.json)单独保留初始结果。

## 环境与输入

| 项目 | 测量条件 |
| --- | --- |
| 系统 | Windows 11，10.0.22631 |
| CPU | AMD Ryzen 7 7840HS，8 核 16 线程 |
| 物理内存 | 系统报告 16,295,776,256 字节，约 15.18 GiB |
| GPU | AMD Radeon 780M Graphics，DXGI `device_id=0` |
| 显卡驱动 | 31.0.14005.11001 |
| SakuraTTS 环境 | Python 3.12.14、NumPy 2.4.6、onnxruntime-directml 1.24.4、pyopenjtalk-plus 0.4.1.post9 |
| Genie 环境 | genie-tts 2.0.2、Python 3.13.7、NumPy 2.4.4、onnxruntime 1.22.1、pyopenjtalk-plus 0.4.1.post8 |
| 原生配置 | FP32 GPT 与声学图；2 个推理线程；resident；capacity=2048；GPT Prefill 查询块长 128；CPU 内存 arena 关闭 |
| CPU GPT 线性层权重 | Fortran 连续布局，转置视图共享存储 |
| 原生采样 | seed=1234，NumPy RNG；top_k=15、top_p=1、temperature=1、repetition_penalty=1.35、noise_scale=0.5、speed=1、cut0；early_stop_num=2700 |

短句为「おはよう。今日もよろしくね。」。长句全文保存在最终数据的 `inputs.texts.long`。原始权重使用 `N.A.V.I.-e15.ckpt` 与 `N.A.V.I._e10_s1310.pth`；参考音频为 `VO08_0011.ogg`，日语转写为「私は“学園”の開発したナビゲーションシステム。通称Ｎ．ａ．ｖ．ｉ．です」。完整 SHA-256 见最终数据的 `inputs.reference_identity`，转换包身份见同文件的 `package_identity`。

原生 CPU 与 DirectML 使用同一个转换包和准备好的参考条件。Genie 使用机器上已有的 ONNX 模型，文件哈希已经保存；该模型没有携带原始 checkpoint 的来源清单，尚未证明其权重与本次原生转换逐张量相同。

`IDXGIFactory.EnumAdapters` / `IDXGIAdapter.GetDesc` 的实际枚举确认设备 0 为 AMD Radeon 780M，设备 1 为 Microsoft Basic Render Driver。设备编号依据 DXGI，不使用 WMI 显卡列表顺序推断。设备 0 的 LUID 为 `0x00000000_0x0000e59f`，资源补测的 GPU 实例与它一致。原始枚举记录及哈希已收录进数据；DXGI 的独立显存和共享内存字段表示容量，不是推理占用。

## 完整请求耗时

每条路径在独立新进程中执行，顺序为 Genie、原生 CPU、原生 DirectML。每个进程先测一个短句首请求，然后分别对短句、长句预热一次，各测两次热请求，合计 21 次。计时截止到完整 PCM 交付，不等待播放，不包含写 WAV 的时间。后台每 100 ms 采样进程树内存。尚未轮换后端顺序，也未独立量化后台负载、电源策略和温度变化；两次样本的中位数只是两值的中点，不适合估计稳定的尾延迟。

| 输入与后端 | 完成时间中位数 | 两次范围 | GPT 中位数 | 声学中位数 | 声学波形时长 | 波形 RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 短句 CPU | 4.389 s | 4.352–4.425 s | 2.259 s | 2.128 s | 3.12 s | 1.407 |
| 短句 DirectML | 2.699 s | 2.683–2.716 s | 2.262 s | 0.434 s | 3.12 s | 0.865 |
| 长句 CPU | 29.125 s | 28.696–29.554 s | 14.952 s | 14.161 s | 20.40 s | 1.428 |
| 长句 DirectML | 18.235 s | 18.215–18.255 s | 15.482 s | 2.735 s | 20.40 s | 0.894 |

DirectML 的完整短句耗时减少 38.5%，长句减少 37.4%；声学阶段分别加速约 4.90 倍和 5.18 倍。GPT 没有从 GPU 加速获益，是 DirectML 路径的主要耗时。

上表 RTF 以声学波形时长为分母。实际交付的 PCM 包含 0.3 秒尾静音，短句为 3.42 秒，长句为 20.70 秒；按完整 PCM 计算的 RTF，短句 CPU / DirectML 为 1.283 / 0.789，长句为 1.407 / 0.881。两条路径本次均在完整片段生成后交付一个 PCM 块，首块延迟接近完成时间。长句即使 RTF 小于 1，也仍需等待约 18 秒才能拿到首块。

## 内存与 CPU 占用

| 热请求 | 工作集采样峰值 | 私有内存采样峰值 | 工作进程 CPU 时间中位数 |
| --- | ---: | ---: | ---: |
| 短句 CPU | 894.7 MiB | 1320.0 MiB | 1.641 CPU-s |
| 短句 DirectML | 1054.4 MiB | 1796.1 MiB | 0.344 CPU-s |
| 长句 CPU | 1344.8 MiB | 1654.8 MiB | 11.773 CPU-s |
| 长句 DirectML | 1700.1 MiB | 2314.7 MiB | 4.844 CPU-s |

工作集与私有内存均为 Windows 进程树采样值，多个进程的共享页面可能重复计入，瞬时峰值也可能漏采。CPU-s 是进程消耗的 CPU 时间，不是墙钟耗时，也不是整机功耗。DirectML 在这两条输入上减少了完成时间和工作进程 CPU 时间，同时增加了工作集与私有内存。

### GPU 内存补测

另一次 DirectML 短句、长句运行使用 Windows PDH 的 `GPU Process Memory` 计数器，按 100 ms 间隔采样完整工作进程树。247 次采样取得 714 条有效计数记录，没有无效状态、重复实例或 PID / LUID 不匹配；实际 GPU 记录属于模型进程 PID 15776。该补测启用额外采样，不替换前面的耗时表。

| 阶段 | Dedicated Usage 峰值 | Shared Usage 峰值 | Total Committed 峰值 |
| --- | ---: | ---: | ---: |
| 短句 | 92.3 MiB | 375.9 MiB | 468.2 MiB |
| 长句 | 94.1 MiB | 832.9 MiB | 927.0 MiB |
| 长句后空闲约 3 秒 | 94.1 MiB | 832.9 MiB | 927.0 MiB |

长句后的 28 次空闲采样保持了上述分配量，约 2.95 秒内未观测到进程树 CPU 时间增长；这一短窗口受计时分辨率限制，不能推断长期空闲占用。显式 sleep 过程中计数下降，随后所属进程全部退出，GPU 计数实例消失。实例消失表示没有可读记录，不能记作测得零占用。采样从模型唤醒完成后开始，不覆盖加载过程的峰值。

这些数值是 WDDM 归属到进程的计数，不能解释为独占显存，也不能与工作集直接相加；共享分配可能重复归属，`Total Committed` 也不是驻留量。最初一次补测只选中了 Windows venv 启动器，未取得 GPU 实例；完整进程树补测解决了这一采样对象问题，原始失败记录仍保留，未当作零显存使用。

## 加载与首请求

| 指标 | Genie CPU | 原生 CPU | 原生 DirectML |
| --- | ---: | ---: | ---: |
| 模型加载 API 耗时 | 7.520 s | 0.705 s | 0.434 s |
| 工作进程就绪耗时 | 13.755 s | 1.163 s | 0.703 s |
| 短句首请求完成时间 | 3.334 s | 7.745 s | 7.726 s |
| 整个进程工作集采样峰值 | 4230.2 MiB | 1344.8 MiB | 1700.1 MiB |
| 整个进程私有内存采样峰值 | 6577.0 MiB | 1654.8 MiB | 2314.7 MiB |

各路径的模型加载 API 边界不同，原生路径的部分初始化发生在首请求，不能只用加载 API 耗时比较冷启动。原生模型转换和参考准备已在测量前完成，其下载、转换成本不在表内；本次也没有清空系统文件缓存。

## Genie 对照的范围

| Genie 热请求 | 完成时间 | 交付 PCM 时长 |
| --- | ---: | ---: |
| 短句第 1 次 | 1.807 s | 2.24 s |
| 短句第 2 次 | 1.699 s | 2.16 s |
| 长句第 1 次 | 16.963 s | 19.24 s |
| 长句第 2 次 | 5.853 s | 7.04 s |

Genie 短句热请求的工作集峰值为 3071.4 MiB，私有内存峰值为 5771.4 MiB；长句分别为 3627.8 MiB 和 6339.0 MiB。本机原生路径的占用较低，但还不能把耗时差异解释成同一生成任务的加速比。

Genie 的公开接口不提供这里使用的采样参数、随机种子或具体停止原因。它沿用现有 ONNX 图中的 `RandomNormalLike` / `ArgMax` 抽样及最多 500 次解码循环，文本与音频处理也不同。两次长句的输出时长相差很大，尚未通过试听或转写确认完整性；不能把 5.853 秒的那次与原生固定 510 个语义 token、20.40 秒波形直接比较。Genie 的默认线程设置没有改动。

## GPU 执行与输出一致性

可复用工具 `research/tools/ort_device_probe.py` 读取声学包内全部四个 `validation-*.npz`，固定语义 token、参考条件与噪声，每例运行两次。DirectML profile 实际记录 **10096 条 `DmlExecutionProvider` 节点执行事件**和 **6136 条 `CPUExecutionProvider` 事件**，确认 GPU 实际承担声学计算，同时存在 CPU 算子。

| 验证样例 | 波形样本数 | 相对包内 expected_waveform 的最大绝对误差 |
| --- | ---: | ---: |
| 0 | 1280 | 1.23e-7 |
| 1 | 2560 | 1.33e-7 |
| 2 | 8960 | 1.43e-6 |
| 3 | 24320 | 1.50e-5 |

八次输出均为有限值，没有样本超出 `atol=1e-4, rtol=1e-5`。DML 节点累计 profile 时长为 1.208 秒，CPU 节点为 0.065 秒；事件数包含重复执行，累计时长不是 GPU 利用率或独占内核耗时，也不替代完整请求计时。合成条件下的数值检查不能代替内容和音色验收。

真实文本下，两次热请求均得到以下结果：

| 输入 | CPU / DirectML 音素与语义 token | 比较的 PCM16 样本数 | 不同样本数 | 最大差异 |
| --- | --- | ---: | ---: | ---: |
| 短句 | 27 个音素、78 个返回语义 token，逐项相同 | 109440 | 416 | 1 LSB |
| 长句 | 311 个音素、510 个返回语义 token，逐项相同 | 662400 | 5025 | 1 LSB |

完整采样序列也逐项相同，停止记录均为 EOS。每个后端内部，两次相同输入的热请求 WAV 哈希一致；跨后端 PCM 接近，但不逐字节相同。

## 生命周期与回归验证

CPU 和 DirectML 各完成两轮真实 HTTP `wake → tts → sleep`。四轮均返回 3.42 秒 PCM，sleep 后状态为 sleeping、模型未加载、worker PID 为空，所属进程树全部退出。这验证了显式唤醒、生成和释放，不覆盖自动空闲定时或反复取消。该 smoke 原始记录中的 `memory_after_tts` 只采到了约 3 MiB 的 Windows venv 启动器，因此不用于推理内存结论；上面的主基准与 WDDM 补测均按进程树采样。

产品测试运行 401 项，400 项通过、1 项跳过；研究工具的 6 项 benchmark 测试和 2 项设备 probe 测试通过。CPU 与 DirectML doctor 的依赖、provider、包和参考身份检查均通过。doctor 本身不执行模型，GPU 执行依据上面的实际 profile。依赖锁检查和 `pip check` 也已通过。

## 初始结果与 CPU 布局探索

首轮短句测量记录了 CPU 4.530 秒、DirectML 2.893 秒的热请求中位数，详细条件与原始证据保留在首轮数据，不与最终样本合并。

CPU GPT 权重布局探索按 C / Fortran 顺序交替运行三个对照。固定 78 步解码历史下，C 布局中位数为 2154.0 ms，Fortran 布局为 1961.4 ms，减少约 8.9%；最大 logits 差异为 1.53e-5，没有 argmax 分歧，独立抽样检查的 79 步采样序列相同。这只是本机固定历史的解码探索，不能泛化为所有文本或整段请求的固定收益。正文最终基准使用 Fortran 布局。

## 证据与适用范围

最终数据按路径与 SHA-256 记录下列原始产物：

- `results/cpu-amd-final/summary.json` 与各后端的逐请求音频、日志和内存采样。
- `results/ort-directml-validation/result.json` 及其 ORT profile，`results/dxgi-adapters.json`。
- `results/cpu-amd-managed-smoke/summary.json` 与运行脚本身份。
- `results/cpu-amd-wddm-tree/result.json`、采样脚本 `.cache/cpu_amd_wddm_tree.py` 与 `research/tools/windows_wddm_memory.py` 的身份；首次 WDDM 采样失败记录。
- `results/cpu-gpt-profile/layout-alternating.json`、`sampling-fortran.json`、最终产品测试日志与两份 doctor 结果。

这些大体积本地产物保留在工作区，紧凑证据随研究记录保存；权重与私人参考音频不随研究记录提交。测量保留了执行 harness 的 SHA-256，但没有冻结完整的运行时代码快照，后续代码状态不能反推成本次实际执行版本。

本次验证覆盖这台机器、这个模型与参考音频的两条日语输入，以及四个声学验证样例。人工试听、ASR 内容检查、中文与英语、模型切换、持续取消、干净安装、后台共存负载、长时间稳定性和整机功耗尚未由这些测量覆盖。
