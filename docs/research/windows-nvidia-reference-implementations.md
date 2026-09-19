# Windows / NVIDIA 参考实现核对

核对日期：2026-09-20。本文检查源代码与 ONNX 图结构，没有运行 Lite / Genie 的 GPU 性能测试。图中张量字节数是静态估算，不是 RTX 5060 的显存实测。

| 项目 | 固定版本 | 核对内容 |
|---|---|---|
| [GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite/tree/6c049397142f4c9147a85f86b6ba37546e93a188) | `6c049397142f4c9147a85f86b6ba37546e93a188` | 普通 / Flash GPT、CUDA Graph、SoVITS、加载和参考缓存 |
| [Genie-TTS](https://github.com/High-Logic/Genie-TTS/tree/d347fd0f8683e9a362b69f59fa0a4799ddb5e828) | `d347fd0f8683e9a362b69f59fa0a4799ddb5e828`，包版本 `2.0.2` | 转换器、五阶段 Session、V2ProPlus 图、参考和模型生命周期 |

研究 checkout 位于已忽略的 `outputs/reference-research/`。它们只供开发核对，不属于 SakuraTTS 的运行依赖。未修改官方 g50 环境、角色原文件或这两个 checkout。两仓库顶层许可证均为 MIT；若复制实现或图资产，仍需保留相应版权和许可，模型权重另按其来源处理。

## Lite：可复用的是缓存所有权和参考生命周期

[普通 GPT 实现](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/gsv_tts/GPT_SoVITS/GPT/t2s_model.py) 的 `T2SBlock` 使用同一套 QKV、输出投影和 FFN 权重执行 Prefill / Decode。Prefill 在第 51–52 行写入前缀 K/V；Decode 在第 90–91 行按有效位置写入一个 K/V，不拼接整个缓存。第 95–96 行对容量缓冲区做注意力，布尔 mask 排除未使用部分。这里能从代码确认原地写入，不需要以 `scatter` 名称猜测行为。

`initialize_runtime` 第 244–280 行分配一对根 K/V 缓冲区。容量由配置中最大的 `batch × sequence` 决定，同一 batch 的小档位是最大档位的视图，各 batch 档位也使用同一根存储。第 293–297 行每个档位单独捕获 Transformer Decode；权重、KV 和输入地址在重放期间保持不变。

图捕获范围不包含完整生成循环。第 418–460 行显示，Prefill、输出投影、Embedding、采样、历史 token 拼接和 EOS 检查仍由外层执行。不能据此宣称“一张图完成全部生成”。各 `torch.cuda.graph` 没有显式传入共享 pool；即使 KV 共享，也需要单独测量各图的私有分配和预热峰值。

默认配置还包含 batch 4 档位：`TTS.py` 第 41 行为 `(1,512)、(1,768)、(1,1024)、(4,512)、(4,1024)`。SakuraTTS 当前只要求 batch 1，不应照搬这组预分配。按 24 层、512 隐藏维度估算，K/V 在 FP16 下的 `batch × capacity = 4096` 是 192 MiB，batch 1、capacity 1024 是 48 MiB；这些只包含 K/V，且必须先确认目标 GPT 的真实配置。

[Flash 版本](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/gsv_tts/GPT_SoVITS/GPT/t2s_model_flash_attn.py) 第 83–91 行使用 `flash_attn_with_kvcache`，将本步 K/V 和有效长度交给内核。它提供另一种缓存更新方式，但引入 PyTorch / FlashAttention 二进制依赖。Windows 和 RTX 5060 上是否有可用构建、实际速度与精度如何，本文均未验证，不能作为最终轻量运行包的现成解决方案。

[SoVITS 实现](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/gsv_tts/GPT_SoVITS/SoVITS/models.py) 第 345–366 行共享各容量档位的输入缓冲区，图只包含 flow 和 waveform decoder。第 387–404 行的量化解码、文本编码、MRTE 和声学噪声仍在图外执行。第 418–420 行仅覆盖当前长度的输入和 mask，没有在此处清空剩余尾部；移植时应明确清零无效区，并用长句后接短句检验状态隔离，不能直接复用这段写法。

Lite 已区分 `v2Pro` / `v2ProPlus`，并使用配置中的 `gin_channels`、上采样通道和卷积参数构建模型。第 315–318、371–377、394 行包含 `sv_emb`、PReLU 和 `ge_to512`。`Loader.py` 识别文件头 `05` / `06`，但 SoVITS 权重加载使用 `strict=False`；SakuraTTS 仍需逐键核对结构和形状，不能把“未抛异常”当作完整适配。

参考准备见 [TTS.py](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/gsv_tts/TTS.py)：

- `cache_spk_audio` 第 1364–1406 行缓存 ERes2Net 结果与模型相关的 `ge`，默认随后释放 SV 模型。
- `cache_prompt_audio` 第 1409–1453 行缓存参考语义、音素和 BERT 特征，默认随后释放 HuBERT。
- 卸载 GPT / SoVITS 会删去模型引用；SoVITS 卸载还删去相应的 `ge`。`gc.collect()` / `empty_cache()` 不证明没有残留引用，仍需观察卸载后占用。

参考缓存键不能直接沿用。Lite 的 prompt 缓存仅以音频路径索引，并使用第一个已加载的 SoVITS 提取语义；它没有把音频内容、转写、模型哈希和预处理配置全部纳入身份。SakuraTTS 已有内容身份和离线参考包，应继续复用。

Lite 的输出行为也不能直接作为官方行为：第 174 行定义 `[280, 486, EOS]` 抑制集合，普通生成在默认前 10 步应用，EOS 默认每 5 步才检查一次。移植缓存和执行安排时，应保留 SakuraTTS 已对照的官方采样和停止规则，不移植这些差异。

## Genie：阶段拆分可用，原图不能直接充当公平 GPU 后端

[V2ProPlus 转换器](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/Converter/v2ProPlus/Converter.py) 使用独立 `prompt_encoder_fp32.onnx` 和 `vits_fp32.onnx`，GPT 继续复用 V2 图模板。转换器没有根据目标 checkpoint 重新导出网络，而是按固定键表将权重写入外部文件，再修改模板中的 external-data 索引。顶层 `Converter/Converter.py` 仅按 SoVITS 文件是否大于 150 MiB 选择 V2ProPlus；这不满足 SakuraTTS 的结构识别要求。

使用 ONNX 1.23.0 只读解析提交内图资产，得到以下事实。所有图均为 opset 20；统计只包含 initializer 声明，不包含执行器复制、常量折叠、arena 或工作区。

| 图 | 主要输入 / 输出 | initializer 声明大小 |
|---|---|---:|
| `t2s_encoder_fp32` | 参考 / 目标音素、BERT、SSL → `x`、参考语义 | 10.93 MiB |
| `t2s_first_stage_decoder_fp32` | `x`、参考语义 → 历史 token、Embedding、48 个 K/V | 292.61 MiB |
| `t2s_stage_decoder_fp32` | 历史 token、Embedding、48 个 K/V → 新历史、停止标志、新 K/V | 292.61 MiB |
| V2ProPlus `prompt_encoder_fp32` | 32 kHz 音频、`sv_emb[1,20480]` → `ge[1,1024,1]`、`ge_advanced[1,512,1]` | 84.42 MiB |
| V2ProPlus `vits_fp32` | 目标音素、语义、`ge`、`ge_advanced` → 音频 | 237.17 MiB |

V2ProPlus 的图确认了真实条件维度：`sv_emb.weight=[1024,20480]`、`ge_to512.weight=[512,1024]`、PReLU 为 1024 通道；waveform decoder 的条件权重为 `[768,1024,1]`，四个 flow 条件卷积的 `weight_v` 均为 `[1536,1024,1]`。这说明不能只把 V2Pro 版本标签放宽。实际 Sakura 权重还必须逐项对照自己的配置、tensor shape 和官方输出。

[ModelManager.py](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/ModelManager.py) 第 122–178 行在加载时把 FP16 文件整体还原为 FP32，再序列化为内存模型。`t2s_shared_fp16.bin` 使两个 decoder 共用磁盘产物；两个 `InferenceSession` 分别加载其完整 initializer，源码没有实现设备权重共享。这里的 FP16 会丢失原 FP32 权重信息，也不会自动降低 FP32 执行的显存。

该管理器固定 `CPUExecutionProvider`，默认 LRU 保留 3 个角色。每个角色的 encoder、first decoder、step decoder、vocoder、prompt encoder 一起加载。HuBERT 和 SV 则由管理器按需加载后保留；角色卸载不清空这两个辅助模型。Genie 的“分阶段”指计算图边界，不是按推理阶段自动卸载权重。

[Core/Inference.py](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/Core/Inference.py) 以 NumPy 调用 `session.run`。每步把完整 K/V 送入图，再取回全部 K/V；step 图有 48 个对 `past_k/v` 执行拼接的节点。仅替换为 CUDA EP 仍保留这套 CPU 数据流和增长缓存，会产生反复传输 / 分配。I/O Binding 可以减少传输，但不自动消除图内 `Concat` 或保证输入输出允许别名。

图内采样存在明确的语义差异：两个 decoder 都将 `Softmax` 除以 `RandomNormalLike` 后 `ArgMax`。官方与 Lite 的采样除数为指数分布，两种分布不等价。图中还固定重复惩罚 `1.35`、温度 `1.0`、Top-k `15`，没有这些参数的输入；不能用相同 seed 推定采样一致。SakuraTTS 应使用已有采样实现，或导出只返回 logits 的计算图，再用固定随机输入对照。

同一入口还会给目标文本前置 `。`、最多循环 500 步，并把最后一个 token 改为 0。`MAX_T2S_LEN=1000` 没有控制这段循环。不能用这些行为产生的短音频作为节省时间或显存的证据。

[ReferenceAudio.py](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/Audio/ReferenceAudio.py) 缓存 SSL、音素和两项 global embedding，说明这些输出可与普通发声分离。其缓存键只有 `(音频路径, language)`，`update_global_emb` 只检查结果是否存在，没有核对 prompt encoder 身份。相同音频跨模型使用会有陈旧条件风险；SakuraTTS 已有身份校验，应保留。

Genie 的普通安装依赖没有 PyTorch，转换器单独导入 PyTorch；这与 SakuraTTS 的交付边界一致。可复用的是“转换产物 + 小型运行依赖”的组织方式，不能依赖 Genie 的本地源码目录、自动下载目录或二进制模板来完成本项目普通推理。

## 对本轮实现的建议

先复用 SakuraTTS 的模型包、日文前端、参考准备、采样、停止与生命周期代码，以自有导出图加 ONNX Runtime CUDA 建立可检查的 Windows 计算路径。ONNX Runtime 本身不要求日常安装完整 PyTorch，但 CUDA / cuDNN DLL、CUDA EP 的实际设备执行和干净环境运行仍须在本机验证。本文不把这一候选写成已交付能力，也不预设它一定比官方快。

第一版计算图应接收已经准备的条件和显式声学噪声，返回 logits / PCM；不要把固定采样参数或不透明 RNG 埋进图。先保存中间结果，再逐项替换实现。

优化优先级由本机测量决定，可先验证以下项目：

1. 离线计算参考语义、`ge` / `ge512`，普通运行不加载 HuBERT、SV、参考编码器；同模型换参考只换经过身份验证的条件。
2. 核对 Prefill / Decode 是否重复存储权重。单一 Session 或显式共享权重才有机会消除重复；共享文件名本身无效。
3. 预分配 batch 1 KV，只更新本步槽位，并明确有效长度。保留完整 mask；若 ORT 基线仍复制整段 KV，应如实记录并测量，不能叫原地缓存。
4. 在正确性通过后验证 WeightNorm 折叠、不变参考投影、声学工作区和按需执行图。捕获前后均报告加载、预热、空闲和请求峰值。
5. 将常驻、释放请求状态、GPT / SoVITS 分阶段加载和空闲卸载分别计时。显存下降需与下一句完整延迟、主存峰值和加载传输成本一起报告。

FP16、量化和高频 CPU / GPU 转移另设实验，不进入首个正确性基线。Lite 的社区速度数字、Genie 的 CPU 数字及项目已有 Mac 数字均不能填入 RTX 5060 的性能结论。
