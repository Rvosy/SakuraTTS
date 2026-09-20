# Windows / RTX 5060 官方基线

2026-09-20 在 RTX 5060 8 GB 上，使用原有 g50 环境完成了 Sakura 的 GPT + V2ProPlus SoVITS 日文推理。FP32 和 FP16 分开运行，覆盖中性参考的短句、长句、多句、标点，随后切换其余四个参考并切回中性。另实测了发行包 WebUI 默认的 CUDA Graph 路径。本文只报告官方基线，不代表 SakuraTTS 后端已经完成质量或性能验收。

实测入口是 `tools/windows_official_baseline.py`。它直接调用发行包的 `TTS_infer_pack.TTS`，使用项目输出目录中的独立配置；原 g50 配置、Python 源码和角色模型、参考文件的 SHA-256 在结束时均保持不变。没有升级官方环境，也没有下载模型。

## 环境、模型与输入

| 项目 | 本次运行 |
|---|---|
| 官方目录 | `D:\Project\sakura\tts\g50`，无 `.git` 的本地发行包 |
| 官方 Python | 原目录 `runtime/python.exe`，3.9.13 |
| PyTorch / torchaudio | 2.7.0+cu128 / 2.7.0+cu128 |
| transformers / NumPy / librosa | 4.43.0 / 1.23.4 / 0.9.2 |
| GPU / 驱动 | NVIDIA GeForce RTX 5060，8151 MiB，WDDM，610.62 |
| 推理路径 | API v2 的普通、非流式、单条推理；未替换任何计算算子 |
| 精度 | FP32 与 FP16 各自独立；matmul 和 cuDNN TF32 均关闭 |
| 语言前端 | 官方 `ja` 路由及现有日文依赖；中文 BERT 按官方加载方式保留 |

发行包没有可核实的提交号，不能为它填写上游 Git commit。本次公共来源标识为：

```text
source-sha256:83a849f2d0accc8a9d51e7ad1cb7475fa135eb7952de8dfae377ea95e5d85c09
```

这是 `GPT_SoVITS/TTS_infer_pack/TTS.py` 的 SHA-256；两轮 `environment.json` 另外保存了发行包 Python 源文件哈希、完整依赖版本、原配置及输入哈希。

| 模型 | SHA-256 |
|---|---|
| `Sakura-e15.ckpt` | `dc0960f365affd723e096f03dc5d0fc37857e68ba30862754ff31caa9645fea4` |
| `Sakura_e8_s7176.pth` | `47f8d5d2c6d1a6e83d6225f0f83c3bc018b5910d7aa2d76aac3d6faaacce992d` |

SoVITS 按 `v2ProPlus` 实际实例化，启用 SV 条件。导出的条件包含 `reference_phones`、`prompt_semantic`、`reference_bert`、`ge` 和 `ge512`。后两者分别为 `(1,1024,1)` 和 `(1,512,1)`；参考计算保留 SV 投影、PReLU、单参考列表求均值和 `ge_to512`，没有按旧 V2Pro 结果推定兼容。

普通单条路径保留 `top_k=15`、`top_p=1`、`temperature=1`、`repetition_penalty=1.35`、`speed_factor=1`、`fragment_interval=0.3`、`cut0`、`batch_size=1`、`parallel_infer=False`、`split_bucket=False`。主输入 seed 为 1234；另用 4321 生成一条自然随机样本。没有为优化改采样参数或截断原文。后文单列默认并行和 WebUI Graph 路径，不把不同执行规则混作同一数值基线。

| 输入 | 日文原文 |
|---|---|
| 短句 | おはよう。今日もよろしくね。 |
| 长句 | 今日は朝から少し雨が降っていたけれど、窓を開けると涼しい風が入ってきて、気持ちがよかった。午後になったら図書館へ行って、前から読みたかった本を探してみようと思う。帰りには駅の近くのお店で温かいお茶を飲みながら、明日の予定をゆっくり考えたいな。 |
| 多句 | おかえりなさい。今日はどんな一日だった？私は図書館で本を読んでいたよ。あとで、一緒にお茶を飲もうね。 |
| 标点 | えっ、本当？……それなら、約束だよ！「また明日」って、ちゃんと言ってね。 |

`cut0` 在这里将每行完整原文作为一片处理。五个参考及其转写均从角色 `character.json` 指向的 `ref.txt` 读取，没有改动原音频。

## 测量结果

请求耗时从提交完整输入计到完整 PCM 到达 CPU，不含 WAV 磁盘写入。包括官方文本前端、必要的参考计算、GPT、SoVITS 和官方每次请求结束时的 `gc.collect()` / `torch.cuda.empty_cache()`。FP16 另计入最终 PCM 合并及数值检查；初始 FP32 轮在该 CPU 合并前停止计时，这个小口径差异已保留，没有修订历史数值。

最初的 FP32 每类热请求只有 1 次，FP16 每类有 3 次；样本量不足以给出稳定的 p95。下面保留这一探索轮数据，列 FP16 中位数和全部范围。随后按与原生后端相同的进程树采样方式补跑了 FP32，每类 3 次，结果单列在下方。

| 输入 | FP32 请求 ms，n=1 | FP32 音频 s | FP16 请求 ms，中位数 [范围]，n=3 | FP16 音频 s | FP16 RTF |
|---|---:|---:|---:|---:|---:|
| 短句 | 557.76 | 2.02 | 515.60 [487.32, 553.31] | 2.02 | 0.255 |
| 长句 | 4946.37 | 27.30 | 3640.41 [3419.90, 4142.52] | 27.30 | 0.133 |
| 多句 | 1950.63 | 10.62 | 1417.16 [1368.59, 1570.76] | 10.62 | 0.133 |
| 标点 | 1390.33 | 8.38 | 1159.90 [1084.65, 1183.34] | 8.34 | 0.139 |

FP16 的短、长、多句分别生成 43、675、258 个送入声学模型的 token，标点生成 201 个；均以 EOS 采样或 EOS argmax 结束，没有触及 1500 步循环上限。FP32 标点对应 202 个 token。FP16 的部分参考切换结果长度也与 FP32 不同；这两种精度不能因为使用相同 seed 就视为相同随机输出。

| 场景 | FP32 | FP16 |
|---|---:|---:|
| 加载模型 ms | 2617.88 | 1898.69 |
| 模型加载后的首句 ms，参考未缓存 | 3200.16 | 1868.08 |
| 模型加载后 allocated MiB | 2503.38 | 1270.72 |
| 首句后 allocated MiB | 2715.21 | 1378.38 |
| 全部文本请求的最大 allocated 峰值 MiB | 3220.44 | 1712.71 |
| 空闲 allocated / reserved MiB | 2716.32 / 2796 | 1380.18 / 1452 |
| 卸载模型后 allocated / reserved MiB | 8.77 / 44 | 9.61 / 26 |

这里的 `allocated` 和 `reserved` 都是 PyTorch 分配器数据，不能称为整个进程的显存。FP16 加载时还有 1745.76 MiB 的临时 allocated 峰值，重载时为 1752.78 MiB；它们高于推理峰值，不能遗漏。

`nvidia-smi` 以 100 ms 间隔记录整张 GPU 的使用量，FP32 / FP16 全轮最大值为 5534 / 4472 MiB。机器仍运行桌面和其他图形应用，WDDM 无法提供可用的逐进程显存，因此不能把这两个全卡数字或其差值当作引擎独占显存。短于采样周期的峰值可能漏掉，完整采样保存在 `memory-samples.json`。

CPU RSS 由 psutil 同步采样。FP32 全轮峰值为 3360.30 MiB；FP16 首轮加载及请求阶段约 3488.24 MiB，包含重载后全轮峰值升到 5070.07 MiB。FP16 卸载前 / 后 RSS 为 2752.55 / 2750.69 MiB，第二次卸载后仍有 4241.47 MiB。显存释放不意味着 Python 进程会立刻归还全部 CPU 内存。

FP16 模型卸载后，重新加载需要 1785.42 ms，随后同一句需要 589.46 ms，共 2374.88 ms。相对热短句中位数 515.60 ms，恢复增加 1859.28 ms，约 360.60%。空闲卸载应作为有代价的可选策略。官方原本就每次清理请求分配器缓存；额外 `empty_cache()` 未降低这轮的空闲 allocated / reserved。

冷启动统计保留在原始记录中，但这个开发 Harness 含来源文件审计，不是纯推理应用启动。FP16 导入阶段为 8414.16 ms，来源与输入哈希计算另用 2020.23 ms。OS 文件缓存没有清空，不能把这些数值写成首次安装或无缓存冷启动性能。

FP32 追加轮保存在 `outputs/windows-baseline/fp32-final/`：不启用诊断捕获，完整输入、参考和参数不变；使用 `--process-tree-memory --repeats 3 --reload-check`，共 21 次请求。计时包含完整 PCM 合并和有限值检查，与原生 Harness 对齐。

| 输入 | 请求中位数 ms | 范围 ms | 音频 s | 语义 token | RTF |
|---|---:|---:|---:|---:|---:|
| 短句 | 632.88 | 606.78–645.36 | 2.02 | 43 | 0.313 |
| 长句 | 5432.79 | 4828.96–5969.14 | 27.30 | 675 | 0.199 |
| 多句 | 1991.41 | 1885.28–2133.18 | 10.62 | 258 | 0.188 |
| 标点 | 1526.82 | 1470.33–1550.25 | 8.38 | 202 | 0.182 |

这一轮加载用时 2141.32 ms，allocated 加载后为 2503.38 MiB、空闲为 2718.09 MiB、请求峰值为 3223.53 MiB、卸载后为 8.77 MiB。重载及下一句共 2657.31 ms。全卡采样峰值为 5899 MiB，包含重载的 CPU 进程树 RSS 峰值为 4715.16 MiB；原始文件哈希检查通过。

进程树 RSS 枚举本身有成本：本机另行 CPU 测量每次约 6.6–7.7 ms，采样间隔中位数为 100.56 ms。这个 Python 采样线程会影响推理主线程的调度。追加轮与探索轮的速度差不能算作后端变化。以下将关闭持续采样的速度轮单列，显存峰值和连续 RSS 仍使用资源轮；不能把有采样的官方结果与无采样的原生结果直接比较。

## 无持续采样的速度基线与官方默认入口

`Monitor(enabled=False)` 不启动 `nvidia-smi` 或采样线程。`--no-memory-sampler` 仍保留请求边界的 RSS、PyTorch 分配器及峰值读取；这些读取位于请求计时之外。普通 FP32 / FP16 各完成 21 次请求，每类热请求 n=3，包含全部五个参考及卸载重载。两轮原始文件哈希检查均通过。

| 输入 | 普通 FP32 中位 ms [范围] | 普通 FP16 中位 ms [范围] | FP32 / FP16 音频 s |
|---|---:|---:|---:|
| 短句 | 565.70 [531.34, 578.27] | 470.51 [435.52, 489.51] | 2.02 / 2.02 |
| 长句 | 4633.84 [4611.38, 4844.54] | 3227.42 [3114.85, 4198.93] | 27.30 / 27.30 |
| 多句 | 1783.22 [1729.56, 1954.45] | 1393.05 [1348.37, 1532.35] | 10.62 / 10.62 |
| 标点 | 1503.69 [1461.72, 1568.42] | 1188.73 [1156.69, 1393.17] | 8.38 / 8.34 |

FP32 / FP16 模型加载分别为 1827.36 / 1693.31 ms；首句分别为 1998.68 / 1791.19 ms；卸载后重载加下一句分别为 2469.07 / 2329.53 ms。完整记录位于 `fp32-timing/`、`fp16-timing/`，与先前有采样的结果分开保留。

发行包启动方式不能只看 API。`go-webui.bat` 和 `go-webui.ps1` 启动 `webui.py`；其中“启用并行推理版本”默认关闭，实际打开 `GPT_SoVITS/inference_webui.py`。该页检测 CUDA Graph 支持后默认开启 Graph，FP16 也默认开启。另一方面，`api_v2.py` 的 `parallel_infer` 默认是 True；并行 WebUI 也默认开启并行，但不走这份 CUDA Graph 实现。

API 默认并行执行路径也完成了一轮 FP16 无采样实测：保持完整文本、batch1、cut0 和全部生成参数，仅开启 `parallel_infer=True`，共 21 次请求，原始文件哈希检查通过。

| 输入 | API FP16 并行中位 ms [范围] | 音频 s | RTF |
|---|---:|---:|---:|
| 短句 | 511.09 [508.31, 542.12] | 2.02 | 0.253 |
| 长句 | 4886.88 [4810.67, 4993.23] | 27.30 | 0.179 |
| 多句 | 2043.62 [1968.68, 2261.94] | 10.62 | 0.192 |
| 标点 | 1601.28 [1566.74, 1711.99] | 8.34 | 0.192 |

该路径在本轮单请求、单片输入上没有获得并行收益；不能据此推论多片或多条批处理的性能。加载为 1754.48 ms，首句为 1883.34 ms，重载加下一句为 2403.08 ms；空闲 allocated 1380.18 MiB、请求 allocated 峰值 1712.71 MiB。完整记录位于 `fp16-parallel-timing/`。此处只比较实际自然生成耗时与输出长度，没有把相同 seed 当作 token 完全一致的证据。

`research/tools/windows_webui_graph_benchmark.py` 直接执行本机 WebUI 的模型加载及 `get_tts_wav`，只略过 Gradio UI 构建和服务器启动。权重路径记录重定向到输出目录，原模型和源码不变，网络连接被禁止。WebUI 自己设置的高进程优先级予以保留并记录。测量包括每次重新计算参考条件到完整 PCM 的全部过程，没有替它补做缓存优化。FP32 / FP16 各完成 13 次请求，Graph 确实捕获；源码哈希检查均通过。

| 输入 | WebUI Graph FP32 中位 ms [范围] | WebUI Graph FP16 中位 ms [范围] | FP32 / FP16 音频 s |
|---|---:|---:|---:|
| 短句 | 651.23 [644.79, 746.77] | 440.04 [437.85, 529.28] | 1.94 / 1.94 |
| 长句 | 6264.80 [6150.37, 6860.85] | 3783.48 [3683.67, 4511.33] | 25.54 / 25.58 |
| 多句 | 3116.78 [3102.95, 3192.36] | 1661.37 [1626.94, 1714.55] | 12.18 / 10.82 |
| 标点 | 2123.86 [2060.14, 2248.38] | 1293.52 [1224.86, 1364.43] | 8.06 / 8.06 |

Graph 另加载一份 GPT，保留原 GPT；其注意力使用固定容量 2500 的 K/V 缓存。FP32 / FP16 空闲 allocated 为 3218.85 / 1620.66 MiB，请求 allocated 峰值为 3680.36 / 1929.00 MiB，均高于对应普通路径的分配器占用。这是该发行包在本机的测量，不能推论所有 Graph 实现都有相同代价。

这份 Graph 仅首步屏蔽 EOS，普通路径前 11 步屏蔽；随机向量长度也不同。Graph 首次请求延迟创建第二份 GPT 和缓存，初始化还会消耗随机数：同 seed 的首句为 84 token、3.66 秒，热短句为 41 token、1.94 秒。因此这些数据是实际默认路径的性能参照，不是相同采样历史的数值对照。不能将输出变短的耗时差称为等质量提速，也不能因为使用了 Graph 就预设它更快。

WebUI 原始文件 `inference_webui.py` 的 SHA-256 为 `7bbedea8b62cfba507a3b04cb66c57b3d832d3836ee588fddddbfc45ec1c64e2`。输出保存在 `webui-graph-fp32-timing/` 和 `webui-graph-fp16-timing/`；`timing-summary.json` 保存无采样轮的中位数、范围、RTF 和音频时长。

## 可复现证据和限制

本地输出目录：

- `outputs/windows-baseline/fp32-naive/`：13 次请求、5 个参考包、WAV、诊断 NPZ、配置、环境和显存采样。
- `outputs/windows-baseline/fp16-naive/`：22 次请求、5 个独立 FP16 参考包，以及卸载和重载记录。
- `outputs/windows-baseline/fp32-capture-{long,multi,punctuation}/`：三个独立进程的完整历史诊断，分别保留 676、259、203 步的 logits 和随机数；生成 675、258、202 个声学 token。原始文件哈希检查均通过。这些带捕获的请求不加入性能统计。
- `outputs/windows-baseline/fp32-final/`：包含递归进程树采样的 FP32 资源轮。
- `outputs/windows-baseline/{fp32-timing,fp16-timing,fp16-parallel-timing}/`：三轮各 21 次请求的无持续采样速度结果，均包含参考切换与卸载重载。
- `outputs/windows-baseline/webui-graph-{fp32,fp16}-timing/`：原始 WebUI Graph 两种精度，各 13 次请求。
- `outputs/windows-baseline/fp32-590/`：590 字完整原文在 `cut2` 下的 7 片诊断捕获，每片均正常 EOS，保留独立随机数、历史、参考条件和声学噪声。请求只在开头设 seed，未逐片重置随机数；它用于完整多片回放，不作速度统计。
- `outputs/windows-baseline-fp32.log`、`outputs/windows-baseline-fp16.log`：官方原始日志。

运行命令如下；它只使用现有本地环境和模型。

```powershell
& D:\Project\sakura\tts\g50\runtime\python.exe -B `
  tools/windows_official_baseline.py `
  --official-root D:\Project\sakura\tts\g50 `
  --character D:\Project\sakura\characters\Sakura `
  --output outputs/windows-baseline/fp16-naive `
  --precision fp16 --suite all --repeats 3 --diagnostics --reload-check
```

FP32 首轮将精度改为 `fp32`、重复次数改为 1，并省略 `--reload-check`。诊断请求与性能请求分开：`diagnostic-neutral-short.npz` 保存完整历史的原始 logits、采样指数随机数、实际 token、目标音素、BERT、声学噪声及编码器中间输出，供后端固定输入回放。诊断的额外复制不计入热请求表。

无持续采样速度轮加 `--no-memory-sampler --process-tree-memory`，不加 `--diagnostics`；API 默认并行路径另加 `--parallel`。WebUI Graph 使用独立入口，默认关闭持续采样：

```powershell
& D:\Project\sakura\tts\g50\runtime\python.exe -B `
  research/tools/windows_webui_graph_benchmark.py `
  --official-root D:\Project\sakura\tts\g50 `
  --character D:\Project\sakura\characters\Sakura `
  --output outputs/windows-baseline/webui-graph-fp16-rerun `
  --precision fp16 --repeats 3
```

单独捕获其他输入可使用 `--precision fp32 --diagnostic-only --diagnostic-case long`，后一个参数也支持 `multi` 和 `punctuation`。长句最后采样 token 为 536，原始 logits 的 argmax 为 280；官方在重复惩罚原地写回后，argmax 变为 EOS 1024，因此正常结束。停止判断必须使用这个顺序，不能用保存的原始 logits 的 argmax 代替。

FP32 中性短句的重复 WAV 哈希并非字节完全一致。与首句相比，其他同 seed 短句只有 11–16 个 PCM16 采样点相差 1 LSB，最大归一化差为 `3.0517578125e-5`。这记录了官方自身波动，没有据此修改后端数值阈值。

还未完成的部分：

- 尚未进行人工听音或 ASR 全文核对，不能从 WAV 有效、EOS 正常推定内容和音质已验收。
- WebUI CUDA Graph 已完成实际运行，但尚未建立该路径与普通路径之间的完整数值和内容对照，不能默认两者能力或音质相同。
- API 默认并行模式已测单条、单片输入；还没有测批量或多片并行吞吐，不能宣称已经找到该发行包所有执行方式的最佳速度。
- GPU 桌面负载、样本量和热状态仍限制统计结论。没有将 Mac 测量或社区性能数字用于本表。
- 未测分阶段模型加载、共存负载、取消或宿主播放；这些属于后续完整后端验收。
