# Windows / RTX 5060：Sakura V2ProPlus 实测

本轮实现了 SakuraTTS 自己的日文原文 → GPT → V2ProPlus → 完整 WAV 链路。普通运行使用 CuPy/cuBLAS 和 ONNX Runtime CUDA，不导入 PyTorch；官方源码、训练环境及 Lite / Genie checkout 只参与准备与开发对照。

验证对象只有本机这组 Sakura 权重，不能据此扩展为所有 V2ProPlus 或 Windows GPU 已兼容。自然生成的人工听音与 ASR 全文检查尚未完成。下面区分固定随机输入的数值通过、实际性能，以及未解决的产品边界。

## 环境和可复现入口

- Windows、RTX 5060 8 GB、WDDM、驱动 610.62。
- 官方 g50：Python 3.9.13、PyTorch / torchaudio 2.7.0+cu128。目录没有独立 Git 身份；以 `TTS.py` 的内容哈希标识来源：`83a849f2d0accc8a9d51e7ad1cb7475fa135eb7952de8dfae377ea95e5d85c09`。
- SakuraTTS 日常环境：Python 3.11、CuPy CUDA12 14.2.0；声学组件为独立 CPython 3.9 / ORT CUDA 1.19.2。CUDA 12.9 运行库和 cuDNN 来自本地已安装文件。
- GPT：`Sakura-e15.ckpt`，SHA-256 `dc0960f365affd723e096f03dc5d0fc37857e68ba30862754ff31caa9645fea4`。
- SoVITS：`Sakura_e8_s7176.pth`，SHA-256 `47f8d5d2c6d1a6e83d6225f0f83c3bc018b5910d7aa2d76aac3d6faaacce992d`。

安装、准备和运行命令见 [Windows 使用说明](../setup-windows-nvidia.md)。本机可直接使用：

```powershell
.venv-windows-runtime\Scripts\sakuratts.exe synthesize `
  --config models\windows-sakura\runtime.json `
  --reference 中性 --text "おはよう。今日もよろしくね。" `
  --output outputs\sakura-example.wav
```

输出路径必须尚不存在。普通配置使用 CPU 准备的参考条件；`runtime-validation.json` 使用官方 CUDA FP32 条件，供严格对照。两份配置均指向包内经典日文前端。CPU 与 CUDA 准备结果的 `ge` 存在最大约 `2.923e-4` 差异，不能混用后声称相同输入。

用户要求完全离线后，没有再下载。ORT 1.30 GPU 与本机 CUDA12 不匹配，最终采用本机已有的 ORT 1.19.2 CPython3.9 二进制，通过持久工作进程解决 ABI 差异。其业务代码、运行库和模型全部位于 SakuraTTS 目录。这个离线组件已经实际运行，但还不是在第二台干净机器验收过的通用发行包。此前已发生的依赖下载没有从安装成本中隐藏；本轮不再为重试而卸载或下载运行库。

## 实现和参考依据

固定参考版本及逐项代码分析见 [Lite / Genie 研究记录](../research/windows-nvidia-reference-implementations.md)。Lite 参考提交为 `6c049397142f4c9147a85f86b6ba37546e93a188`，Genie 为 `d347fd0f8683e9a362b69f59fa0a4799ddb5e828`。没有引用社区性能数字作为本机结果。

GPT 复用现有 FP32 模型包、采样与停止代码。Prefill / Decode 共用一份权重，K/V 预分配并原地写入，仅当前长度参与注意力。CUDA Graph 捕获单步 Decode，包含 Embedding、24 层 Transformer 与输出投影，采样仍在图外逐步进行。没有移植 Lite 的前十步额外 token 抑制、每五步 EOS 检查，也没有使用 Genie 的正态分布除数采样。

声学转换按权重内部配置与文件头一致性建立真实 V2ProPlus：`ge` 为 1024 维、`ge512` 为 512 维，上采样率为 `[10,8,2,2,2]`，初始通道 768。导出前严格核对 tensor schema；普通图只保留生成目标波形所需计算。调试图与普通图共享磁盘权重，运行时只加载其中一个 Session。

HuBERT、SV 编码器与参考音频处理在准备阶段执行。普通请求读五组已校验条件，不重复加载辅助模型。日文零 BERT 遵循官方日文路径，没有关闭必需语言特征来节省显存。

官方前端实际使用 classic `pyopenjtalk 0.3.4`。原有 plus 版本对长句、多句、标点分别得到 311/112/75 个目标音素，官方为 317/115/74。单独换字典或 `use_vanilla=True` 不能消除差异。现将 classic 模块、原主字典及许可封装进 `frontend-classic`；保留 plus 路径自身的行为，未通过修改音素或目标文本对齐测试。

## 固定输入验证

四类文本的原文见 [官方基线](2026-09-20-windows-official-baseline.md)。双方保持 `top_k=15`、`top_p=1`、温度 1、重复惩罚 1.35、语速 1、声学噪声系数 0.5、尾静音 0.3 秒、batch1 与 cut0。正常采样使用 NumPy RNG，相同 seed 不等价于 Torch seed；严格回放只传入官方捕获的指数随机数和声学噪声，不传目标 token 或强制停止位置。

| 输入 | GPT 采样步数 | 声学 token | 完整 PCM 秒数 | GPT logits 最大绝对差 | PCM 最大差，LSB |
|---|---:|---:|---:|---:|---:|
| 短句 | 44 | 43 | 2.02 | 4.20e-5 | 1 |
| 长句 | 676 | 675 | 27.30 | 8.77e-5 | 2 |
| 多句 | 259 | 258 | 10.62 | 4.58e-5 | 3 |
| 标点 | 203 | 202 | 8.38 | 4.58e-5 | 1 |

四类目标音素、完整采样 token、声学 token、返回位置和 PCM 长度全部一致；整链各重复 3 次。保持原来的 `atol=1e-4, rtol=1e-5`，没有失败后扩大阈值。长句最后采样 token 并非 EOS，但重复惩罚后 argmax 为 EOS，双方按同一停止规则结束；没有用未经惩罚的 logits 误判。

声学单独对照覆盖 12 个中间输出，真实中性短句波形最大误差 `8.49e-6`、MRTE 最大误差 `8.33e-5`，均通过原容差。动态 shape 1/2/7/19 个语义 token 另做 CPU 导出验证。ORT profile 显示浮点 Conv、MatMul、Softmax 运行于 CUDA，CPU 节点仅处理 int64/bool shape 数据，最大 64 字节；这不是仅更换 Execution Provider 的声明。

证据位于 `outputs/windows-cuda/gpt-warp-*`、`request-replay-*-classic` 和 `outputs/windows-acoustic/ort119-cu128-path`。旧 plus 前端失败记录、错误容差方向的导出试验与修复前数据均保留，没有覆盖失败产物。

## 计量口径

请求时间从完整原文提交到完整 PCM 返回，含前端、采样、数据传输、缺失模型重载和最终合并；WAV 磁盘写入另计。本文 benchmark 的 `rtf_pcm` 使用含配置尾静音的完整 PCM 时长，原始记录同时保留 speech-only 时长。CLI 和整链回放中的旧字段 `rtf` 使用 speech-only，不能直接混为同一指标。严格随机回放的数组在计时前读入，计时后才核对结果；自然采样另测，不把预计算随机数的成本隐藏为完整自然生成速度。

资源轮使用 `nvidia-smi` 每 100 ms 记录全卡占用，psutil 记录主进程和递归子进程 RSS。WDDM 的逐进程显存不可用，下面的 GPU 数字包含桌面负载，不能称为进程独占显存；短于采样周期的峰值可能漏掉。CPU RSS 可能重复统计共享页。PyTorch 分配器和 CuPy pool 另列，二者不能直接相减，也不能替代 ORT / 驱动 / 上下文成本。

CPU 进程树采样每次约 7 ms，会干扰 Python 调度，尤其是自有 Decode 循环。因此资源轮和无持续采样的速度轮分别保留。每类 n=3 的中位数与范围只作本机探索测量，样本不足以发布稳定 p95。

## 显存与生命周期

资源轮使用完全相同的 FP32 模型、经典前端和 CUDA 参考条件，回放生成相同 token 和长度。官方请求仍执行发行包原有的清理行为。

| 场景，全卡采样 MiB | 官方 FP32 普通单条路径 | SakuraTTS 常驻 |
|---|---:|---:|
| 最早空载样本 | 2274 | 2302 |
| 加载后最近样本 | 4540 | 2930 |
| 全轮峰值 | 5899 | 4117 |
| 连续请求后的空闲 | 5179 | 4103 |
| 卸载模型后 | 2442 | 2319 |

全卡观测峰值减少 1782 MiB（30.2%）；两个峰值均发生在相同的 675-token 长句热请求。各自序列后的空闲减少 1076 MiB（20.8%），但官方空闲前还执行了参考切换和随机短句，不能把空闲差值称作严格相同请求历史的控制结果。扣除各轮最早空载样本，峰值增量从 3625 降至 1815 MiB，空闲增量从 2905 降至 1801 MiB。这些是带桌面负载的采样差值，不是精确进程显存。官方 FP16 探索轮全卡峰值 4472 MiB，不能把相对官方 FP32 的节省量套给 FP16。

官方 FP32 allocated 加载后 2503.38 MiB、空闲 2718.09 MiB、请求峰值 3223.53 MiB、卸载后 8.77 MiB。SakuraTTS 的 GPT tensor 权重为 318,617,608 字节（加载后 CuPy pool 为 318,618,624 字节）、完整 2048 容量 K/V 为 201,326,592 字节；空闲 CuPy pool used/total 约 495.88/584.85 MiB，另有 ORT 与两个 CUDA 上下文。两条路径的全部 CPU 进程树 RSS 峰值分别约 4715.16 和 1891.48 MiB，均计入重载。

请求状态释放和分阶段加载是可选策略，默认仍为常驻。释放 K/V 与 Graph 可以降低空闲，但下一句需要重新分配及捕获；分阶段策略每片重载模型，代价更明显。具体速度表随无采样对照单列。

显式卸载后重载并生成短句，SakuraTTS 资源轮需要 2670.23 ms，常驻热短句为 287.38 ms：增加 2382.85 ms（829.2%）。官方 FP32 对应重载及首句为 2657.31 ms。不能因为卸载后的显存低，就把它作为连续对话默认策略。

## 已测优化

注意力内核初版按跨行地址读取 K/V。改为 warp 内连续读取后，旧 plus 前端的相同长句、646 个 token，完整请求由约 8.36 秒降至 3.89 秒（约 53%）。前后固定同一输入与 token，源码快照保留在 `benchmark-resident-v1`。这组数据用于解释内核优化，不用于与官方经典前端比较。

声学小工作区默认采用 HEURISTIC、禁用最大卷积工作区。真实 43 token 热声学用时 71.41 ms，大工作区 EXHAUSTIVE 为 49.18 ms，前者慢 22.23 ms（45.19%）；但后者首次搜索达 6797.51 ms，全卡峰值 7331 MiB，而小工作区为 2775 MiB。430 个重复 token 的合成长 shape 也保留，但不作为真实长文质量结论。小工作区是明确的速度 / 显存取舍，其代价已经包括在整链时间里。

没有将权重转移至 CPU、重算或量化作为默认优化；本轮未验证这些策略的完整代价与质量，因此不宣称它们已完成。

## 独立运行、错误和限制

干净日常环境不可导入 Torch、torchaudio、Transformers、ONNX 导出器或 MLX。使用公开 `synthesize` CLI 的独立检查生成 3.46 秒 WAV，记录的 `run_cli` 入口至文件完成为 4022.18 ms；它不含解释器启动、此前的模块导入及退出清理。该轮有文件审计，磁盘与编译缓存未清空，不是首次安装性能。

`harness/windows_runtime_isolation.py` 对主进程和两个工作进程安装 Python 审计，禁止读取 g50、SakuraTTS-References、Lite/Genie checkout，并拒绝网络连接。三进程均无违规、无 Torch 导入。证据在 `outputs/windows-cuda/standalone-isolation-v2`。Python 审计不覆盖全部原生库文件 I/O，不宣称是操作系统级沙箱。

安装前的日常主环境约 1.45 GiB，声学独立组件约 2.86 GiB，合计约 4.31 GiB，未含模型、共享 Python、缓存与开发环境；测量后从现有本地缓存增加了 psutil。为保持离线保留两个 Python ABI，有额外进程、CPU 内存和 IPC 成本，不能只报告单进程依赖体积。

590 字原文另外按双方相同的 cut2 规则分成 7 片，长度为 69/80/86/84/78/81/112 字，逐片完整重组原文。官方捕获与本项目回放均生成 123.82 秒 PCM，全部自然 EOS。七片规范化文本、音素、采样 token、语义 token、停止原因及返回位置均相同；完整 PCM 最大误差 3 LSB，原 FP32 阈值外样本为零。证据在 `outputs/windows-baseline/fp32-590` 与 `outputs/windows-cuda/replay-590`。这证明多片行为和计算对齐，不替代对实际发音内容的听音验收。

真实故障测试覆盖杀死声学 worker 后下一请求新建 worker，以及 staged 模式在 Prefill 后、声学执行前取消。已修复 staged 异常后权重残留的问题；三种重试均完整完成，音素、采样和停止行为一致。重建声学 Session 后 PCM 有 14–16 个样本相差 1 LSB，位级一致检查仍为失败，原始 `passed=false` 没有改写；它们全部满足此前的 FP32 容差。分层汇总及原始数组位于 `outputs/windows-cuda/failure-lifecycle-quantified`。

还未验收：自然随机生成的人工音质与 ASR 全文、其他 V2ProPlus 权重、另一台干净机器、流式首包、宿主播放及共存负载。当前明确支持日文与固定参数范围，不把中文、多语言、变速或未验证 Top-p 宣称为已有能力。达到容量报错，达到生成上限标记 `stopped_at_limit`，不能把部分音频写成完整文本通过。
