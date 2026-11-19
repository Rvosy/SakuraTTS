# Windows 普通话前端的内存与耗时

在 Ryzen 7 9700X 上，两张模型都参与判音和特征提取时，中文前端少占 404 MiB 私有内存。首句只需查词典判音时，少占 735 MiB。普通话、日语和英语继续共用同一组推理档位。

## 普通话请求

Windows 11 build 22631，Ryzen 7 9700X，ONNX Runtime 1.24.4，CPUExecutionProvider，每张中文图使用两个推理线程。用同一份 Wass V2ProPlus 前端分别加载旧包的内嵌 FP32 权重和新版导出的外部 FP32 权重，调用 `TextFrontend.prepare_target(..., "zh", "cut0")`。短句是“今天下午三点，我们在图书馆门口见。”，长句是“明天下午三点，我们在银行门口见面。去之前先看一下天气预报，如果下雨，记得带伞。”。

| 前端进程树 | 原版 | 新版 |
| --- | ---: | ---: |
| 处理短句后新增工作集 | 2,141 MiB | 1,739 MiB |
| 处理短句后新增私有内存 | 2,133 MiB | 1,729 MiB |
| 短长句连续处理时的工作集峰值增量 | 2,156 MiB | 1,742 MiB |
| 短长句连续处理时的私有内存峰值增量 | 2,156 MiB | 1,732 MiB |
| 首次只处理“森林。”后新增私有内存 | 2,131 MiB | 1,397 MiB |

基线取自公共前端初始化完成、尚未提交文本的时刻。工作集是驻留在物理内存中的页，私有内存是 Windows 为进程承诺的私有字节数。峰值按 10 ms 间隔采样并汇总同一时刻的前端进程树。每个版本分别在三个新进程中处理短长句，另起三个进程测“森林。”，表中报告三次的中位数。GPT、SoVITS 和参考音频处理不参与计时。

| 普通话文本处理 | 原版 | 新版 |
| --- | ---: | ---: |
| 首次短句 | 3.34 秒 | 3.03 秒 |
| 首次“森林。” | 3.46 秒 | 2.62 秒 |
| 热短句 | 52.8 毫秒 | 53.8 毫秒 |
| 热长句 | 107.2 毫秒 | 109.3 毫秒 |

首次处理使用新进程和已有的系统文件缓存。两个版本各用本地压缩包解出的解释器、代码和转换得到的 CPU 模型，按旧新、新旧、旧新的顺序轮换。热测交替输入短句与长句，预热后每种长度共计 15 次。计时包含分句、分词、判音、BERT 和音素对齐。[测量入口](../tools/windows_mandarin_frontend_memory.py) · [逐次数据](../experiments/data/2026-11-20-windows-mandarin-frontend-memory.json)

## 内存从哪里省下来

Hugging Face Optimum 支持把一张 ONNX 图的权重放入一个独立文件。这里用相同的 ONNX 标准格式，把 G2PW 和 BERT 的大张量分别写入同目录的权重文件，保留原算子和 FP32 参数。在相同旧版前端里单独更换权重布局，两张模型加载并连续处理三句后，进程工作集从 2,046 MiB 降到 1,685 MiB；私有内存从 2,510 MiB 降到 2,148 MiB。两张图和权重合计仍约 1.83 GB。[Optimum 的导出配置](https://github.com/huggingface/optimum-onnx/blob/main/optimum/onnxruntime/configuration.py) · [ONNX 外部权重格式](https://onnx.ai/onnx/repo-docs/ExternalData.html) · [11 种会话与权重布局对照](../experiments/data/2026-11-20-windows-chinese-session-memory.json)

G2PW 原有的字典能够直接给出一部分字的读音。新版直到句中出现需要神经网络判音的字才创建其 ONNX 会话，此后复用会话。发音掩码也只为这次查询的字分配。对“森林。”，只需加载 BERT；对“银行行长在重庆开会”，两张图都会用到。[g2pW](https://github.com/GitYCC/g2pW)

## 读音和特征

固定 96 句，覆盖多音字、地名、连读变调、日期、数字和繁体文本。原版和新版的规范化结果、2,681 个音素及逐字对齐全部一致。按音素展开的 2,745,344 个 BERT 特征、660 次 G2PW 查询返回的 861,300 个概率和 660 个首选读音置信度逐字节一致。[语料](../experiments/configs/mandarin-pronunciation-corpus.txt) · [数值对照与重跑命令](../experiments/data/2026-11-20-mandarin-frontend-equivalence.json)

动态 INT8 把两张图压到约 0.46 GB，但“这一着棋”和“单县”的读音发生变化，96 句的 BERT 特征均超出现有容差。把 BERT 矩阵以 FP16 存储、在 CPU 上转回 FP32 计算，六句特征通过容差检查，短句的 BERT 处理从约 30–50 毫秒增加到约 110–160 毫秒。[ONNX Runtime 量化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) · [FP16 权重存储实测](../experiments/data/2026-11-20-windows-mandarin-fp16-storage.json)

单独预存本机的矩阵预排结果能把热私有内存降至 758 MiB，热工作集仍有 1,679 MiB，两张图的磁盘占用增至 3.28 GB，并引入运行库和目标 CPU 的兼容条件。完整包使用标准 FP32 外部权重。[离线图优化的硬件条件](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)

## 解压包与普通话合成

本地 Windows 中日英包共 3,577,555,348 字节，SHA-256 为 `3ab82f4931f65769dad8cfc84414b9a18060ff8b7edf69c1b88f52eefd4c5fe7`。使用快速 LZMA2 压缩，解压并核对全包 21,305 个文件。原始 GPT / SoVITS 权重、普通话参考录音和转写都来自本地，CPU 和 CUDA FP32 各从空转换缓存启动。

Wass V2ProPlus 朗读“今天是星期五，出门记得带伞。”，种子 1234。CPU 均衡档的四份音频与旧版逐样本一致，每份 96,640 个 PCM16 样本，时长 3.02 秒。CUDA FP32 每份有 4–9 个样本相差一个最低位，旧包在同机重复运行时观察到的范围为 0–12 个。另在 ONNX Runtime 1.30.0 的 CPU 后端用“银行行长去重庆开会。”核对了两张新旧图，G2PW 概率和 BERT 特征也逐字节一致。

桌面 RTX 5070 Ti 16 GB 上，三个共享 FP16 档各合成两句普通话，每句预热一次、热测三次。24 份 WAV 均与旧包对应文件逐字节一致。表中耗时为从本机 HTTP 请求发送到完整 WAV 返回的热测中位数，短句音频 3.70 秒，长句 10.22 秒。显存按 Windows 服务进程树的专用显存统计，峰值取全部请求中的最高观测值，单位为十进制 MB。[请求、RTF、显存与 PCM 数据](../experiments/data/2026-11-20-mandarin-frontend-cuda.json)

| CUDA 档位 | 短句整句耗时 | 长句整句耗时 | 请求后显存 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: |
| FP16 标准 | 0.257 秒 | 0.661 秒 | 767 MB | 886 MB |
| FP16 低显存 | 0.895 秒 | 1.252 秒 | 584 MB | 773 MB |
| FP16 极限 | 2.266 秒 | 2.650 秒 | 205 MB | 601 MB |
