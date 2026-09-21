# GPU 推理研究与证据

整理日期：2026-09-19。需求来源：项目讨论《重构轻量GPU推理》。原始聊天记录未随仓库发布，本文保留需求整理及可公开访问的技术来源。

本次读取了完整的两轮聊天记录，并补做三个主仓库的定点源码核对、两个补充项目的 README 核对，以及下文列出的论文题录 / 摘要核对。随后已在 Mac 上运行官方和 Lite 的 MPS / FP32 路径，结果单列于 [Mac 首轮验证](../experiments/2026-09-19-macos-reference-smoke.md)；CUDA 推理和所有论文的全文复核仍未完成。原聊天中的引用编号没有可用目标，因此不把编号直接当作证据。

## 从讨论落实的方向

用户的目标是使用已有 GPT-SoVITS 模型，通过自己的 GPU 推理后端发声，同时减少安装空间和运行资源，改善速度。现有 CPU TTS 方案继续沿用。

后续回复将早期“精简 PyTorch 作为起点”推进为“转换与验证可用 PyTorch，日常运行包采用独立原生路径”。文档采用这个方向。用户随后确定先跑通 Mac，并选用现有“朱雀院红叶”V2Pro；C++、CUDA、单请求及具体后端仍属于产品设计建议，尚未实现。

本文把资料分为源码事实、作者报告、理论估算和待核实线索。工程方案由这些资料支持，但不直接继承它们的性能数字。

## 两份后续评估如何影响当前实现

2026-09-19 又读取了《仓库进度评估》和《分析仓库与论文方向》。后者审阅时固定在 `a67074a7`，早于本轮中日语言段、G2PW 内存映射和持久参考包的提交。采用其关于独立请求、通用优化和 Windows 执行组织的建议，不能把其中的旧状态照抄为当前进度。用户随后明确日文优先，中文专项暂缓。

本轮重新核对自有源码：`generation.py` 每步将完整 logits 转为 NumPy，`mlx_gpt.py` 仍使用函数式 `slice_update`；这支持在 Windows 测量主机参与和 KV 物理更新，而不是直接承诺消除 PCIe 传输后的加速。Prefill 的参考语义能看到全部文本，整段参考 KV 不能跨目标文本复用。Flow / decoder 的静态 WeightNorm 和不变条件投影属于可迁移的预计算候选，分别测量，不合并收益。

| 研究线索 | 目前采用的方向 | 启动性能实现的条件 |
|---|---|---|
| FlashInfer / CUDA Graph | 明确单步状态、KV、采样和停止的所有权，准备真实 shape 与固定输入 | Windows profiler 证明瓶颈；核对目标消费级 GPU、运行库和稳定地址，不引入服务框架 |
| VoXtream2 等流式工作 | 报告首次可播放、持续输出、打断和剩余音频；先从句子边界研究 | 日文完整请求稳定，弄清 SoVITS 非因果依赖；不直接更换现有模型 |
| AWQ / T-Mimi 的精度敏感性思路 | 保留高精度参考，逐模块评估精度与最终语音 | 基线和质量标准固定后再实验；论文中的层敏感结论不能直接当作 SoVITS 规则 |
| TurboQuant 等 KV 压缩 | 暂列低优先级 | 实测 KV 占比足够大，且含尺度、转换与工作区后的净收益成立 |
| SSD / PCG / Llasa+ 等多 token 策略 | 保留为独立后续研究 | 基础执行优化后仍有收益；补齐训练、适配、接受分布与质量证据，不改默认兼容语义 |
| FlashAttention-4 | 跟踪硬件与布局条件 | 在实际目标 GPU、短历史和模型 head dimension 下成立，不能按版本号选型 |

这些名称和新增论文描述来自评估记录；本轮没有重新核验新增论文全文或复现实验，不继承其加速倍数和质量结论。进入具体实验前，仍须固定可访问的论文、实现和资源许可。当前主线无需等这项文献补充：日文原始文本到 PCM、参考准备、生命周期和统一端到端验收都有已知实现依据。

## 三个主仓库

本次核对固定在以下提交。官方和 Lite 的 Mac 功能冒烟也使用这两个提交；Genie 仅用于源码参考，三者均无本项目的 CUDA 基准结果。

| 项目 | 固定提交 | 主要用途 |
|---|---|---|
| GPT-SoVITS | [`48b1a0169a28582a8984402f82cf438d3bfa6aca`](https://github.com/RVC-Boss/GPT-SoVITS/tree/48b1a0169a28582a8984402f82cf438d3bfa6aca) | 模型结构、行为和官方优化路径参考 |
| GSV-TTS-Lite | [`6c049397142f4c9147a85f86b6ba37546e93a188`](https://github.com/chinokikiss/GSV-TTS-Lite/tree/6c049397142f4c9147a85f86b6ba37546e93a188) | GPU 执行方式和资源生命周期参考 |
| Genie-TTS | [`d347fd0f8683e9a362b69f59fa0a4799ddb5e828`](https://github.com/High-Logic/Genie-TTS/tree/d347fd0f8683e9a362b69f59fa0a4799ddb5e828) | 转换、阶段拆分和独立模型分发参考 |

### GPT-SoVITS

[CUDA Graph 实现](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/GPT_SoVITS/AR/models/t2s_model_cudagraph.py) 已有预分配 KV、`scatter_` 更新及图捕获代码。因此后续需要比较官方普通路径和可用的 CUDA Graph 路径，不能把早期实现代表整个官方项目。

原聊天提到旧路径中的 `torch.cat`。即使存在该操作，也只能据张量位置判断其分配、复制成本，不能直接推出发生 CPU / GPU 往返。是否有主机传输和同步需要执行轨迹证明。

[文本预处理代码](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py) 请求 hidden states 后取 `[-3:-2]`，提供了裁剪无输出依赖计算的机会。要先确定已有导出器是否已经裁剪，并验证输出层、特殊 token 和音素对齐。

### GSV-TTS-Lite

[语义生成代码](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/gsv_tts/GPT_SoVITS/GPT/t2s_model.py) 可见按位置写入 KV、容量档位共享缓冲区、Prefill 与 Decode 拆分、CUDA Graph 捕获。较小的视图不代表较大的底层缓冲区已经释放；共享缓冲区也不能由多个请求同时修改。

[TTS 入口](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/gsv_tts/TTS.py) 默认 `use_bert=False`，并提供参考处理后不长期保留 CNHuBERT 的安排。中文性能对照必须明确是否启用特征及其设备，不能把默认结果等同于完整中文能力的峰值。

原聊天还分析了增量 FlashAttention、图外采样、SoVITS 流式重叠和其他参考条件缓存。本次只核对上述关键入口；正式复用前仍需逐项检查实现和依赖。

[固定版本 README](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/README.md) 公布了 RTX 3050 Laptop 上的下列结果，且声明支持 V2 / V2Pro / V2ProPlus。它们是作者报告，尚无本项目复测。

| 路径 | 作者标注的首包延迟 | RTF | 显存 |
|---|---:|---:|---:|
| Original，`streaming_mode=3` | 436 ms | 0.381 | 1.6 GB |
| Lite，FlashAttention 关闭 | 150 ms | 0.125 | 0.8 GB |
| Lite，FlashAttention 开启 | 133 ms | 0.108 | 0.8 GB |

表格不足以确定全部模型、中文特征、预热、缓存和测量边界。它支持优先研究执行安排的判断，不能用作 SakuraTTS 的性能承诺。

### Genie-TTS

[ModelManager](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/ModelManager.py) 将 Provider 配为 `CPUExecutionProvider`，并有 FP16 文件读取后转换为 FP32 再创建 Session 的逻辑。[Inference](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/Core/Inference.py) 主路径调用 `t2s_cpu()`，使用 NumPy 和多个 Session 逐步生成。

[V2 转换器](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/Converter/v2/T2SConverter.py) 将原权重写入共享 FP16 文件，并修改既有 ONNX 模型的外部数据引用。值得参考的是转换和分发边界；固定模板仍依赖模型结构，磁盘共享文件也不证明设备权重共享。

原聊天引用的“4070 Laptop 从 10 秒降到 0.6 秒”缺少对应 GPU 代码版本和测试记录，不采纳为该项目已实现的性能。当前核对到的 CPU 路径不能仅修改 Provider 就视为高效 GPU 后端。

## 补充社区实现与部署文档

| 资料 | 本次核对内容 | 对本项目的作用 |
|---|---|---|
| [GPT-SoVITS_minimal_inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference) | 2026-09-19 读取主分支 README：ONNX / TensorRT 路径及作者性能表 | 作为图导出、资源与性能取舍的实验参考；未核对全部实现 |
| [GPT-SoVITS-cpp](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp) | 同日读取主分支 README：Python 负责导出，C++ SDK 区分说话人准备与推理 | 支持转换与运行分离的设计；不直接继承其速度或最低配置 |
| [ONNX Runtime I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) | 官方文档说明默认 CPU 输入输出可能产生设备复制 | 设备放置需要明确；没有给普通图的任意输入输出别名提供通用安全保证 |
| [TensorRT-RTX 文档](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html) | 官方介绍桌面 RTX 部署、AOT / JIT，以及小于 200 MB 的运行库描述 | 作为候选；运行库描述不包含本项目全部模型、语言资源、依赖与缓存 |

`minimal_inference` 作者在 i7-12700、RTX 2080 Ti（README 标注 22G）、CUDA 12.9、FP16 下报告：官方 PyTorch 为 148.65 token/s、3 G；ONNX 为 172.4 token/s、3.9 G；TensorRT fitted 为 291.6 token/s、3.4 G。单位沿用作者原文，未确认测量工具。它说明换框架可能同时提高速度和显存，不能与 Lite 的另一套环境横向排名。

聊天中关于 out-of-place `scatter` 的观察，以及 TensorRT-RTX 特定版本算子限制，需要在后端实验中重新核对固定源码和支持矩阵。本次未把这些版本相关细节写成永久产品限制。

## 论文与内核方向

下表的六篇论文已核对题录和摘要；MARLIN 核对的是项目 README。尚未在 GPT-SoVITS 上复现论文实验。优先级依据桌宠的单请求、小隐藏维度和短历史场景判断，不是论文加速倍数的排序。

| 来源 | 研究内容 | 使用方式与前提 |
|---|---|---|
| [FlashAttention，2022](https://arxiv.org/abs/2205.14135) | 用分块减少精确注意力的显存读写和中间存储 | 优先评估现有内核，分别验证 Prefill 掩码和 Decode；保留数值检查 |
| [FlashInfer，2025](https://arxiv.org/abs/2501.01005) | 可定制注意力、KV 布局和兼容 CUDA Graph 的调度 | 借鉴适合真实 head dimension、历史长度和缓存布局的实现；依赖与 JIT 也计入分发成本 |
| [FlashDecoding++，2023](https://arxiv.org/abs/2311.01282) | 解码中的 softmax 同步、小矩阵 GEMM 和硬件相关数据流 | 优先分析单 token 执行与融合；具体数值策略另行验证，不照搬 LLM 收益 |
| [AWQ，2023](https://arxiv.org/abs/2306.00978) | 用激活统计指导低比特权重量化 | FP16 通过后做 GPT 线性层实验，用语音任务数据校准 |
| [MARLIN](https://github.com/IST-DASLab/marlin) | 权重打包、低比特读取和反量化计算内核 | README 说明核心 CUDA 文件可独立集成；仍需核对布局、GPU 和小矩阵收益，不能假定 AWQ 产物直接通用 |
| [PagedAttention，2023](https://arxiv.org/abs/2309.06180) | 多请求 KV 的碎片、分配和共享 | 首版优先级低；分页不等于压缩每个 K/V 元素 |
| [KIVI，2024](https://arxiv.org/abs/2402.02750) | 针对 K / V 分布差异的非对称 KV 量化 | 先测缓存占比和 INT8 净收益，再考虑更低精度；LLM 的质量结果不等于 TTS 已验证 |

FlashAttention 的长序列优势、FlashInfer 的服务吞吐、MARLIN 的低比特矩阵运算针对不同瓶颈，不能将它们的加速倍数相乘。

## 待核实的后续线索

原聊天还提到以下研究，但导入记录未保留可追溯的原论文链接。本次不补写未经核实的论文 ID、版本和测试数字；这些名称只保留为检索线索。

| 线索 | 原聊天描述的方向 | 使用前要补齐的证据 |
|---|---|---|
| TurboQuant | 在线向量 / KV 压缩 | 原论文、误差假设、代码版本、反量化成本及 GPT-SoVITS 适用性 |
| Speech Speculative Decoding（SSD） | 语音 token 的草稿生成与验证 | 原论文、目标模型、训练成本、接受规则、内容错误率和资源增量 |
| VADUSA | 额外预测头与多 token 推测 | 原论文、结构改动、训练需求，以及能否适配用户微调权重 |
| FlashAttention-4 | 新硬件上的注意力流水线 | 正式论文 / 实现版本、支持 GPU、实际头维度与桌面显卡收益 |

原聊天给出的 SSD 速度及词错误率数字不纳入验收依据。标准投机解码是否保持目标分布取决于完整接受 / 修正规则；放宽接受条件的变体需要独立质量验证。研究优先级放在不需重训的兼容引擎之后。

## 容量估算与优化顺序

以下仅是配置推导。固定的官方 [s1longer-v2.yaml](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/GPT_SoVITS/configs/s1longer-v2.yaml) 包含 24 层、隐藏维度 512、16 头、FFN 2048、语义词表 1025。实际模型必须读取自身配置，不能由此推广到所有版本。

普通 MHA 的缓存数据近似为：

```text
KV 字节数 = 2 × 层数 × batch × 缓存长度 × KV 头数 × 头维度 × 元素字节数
普通 MHA 中，KV 头数 × 头维度 = 隐藏维度
```

| 单请求 FP16 容量 | KV 理论数据量 |
|---:|---:|
| 512 token | 24 MiB |
| 1024 token | 48 MiB |
| 2048 token | 96 MiB |

长度包含全部进入缓存的前缀和生成 token；不含权重、图、工作区和分配器开销。此配置头维度为 32，需要测试内核对这个形状的支持。

在同一配置下，Transformer 主要线性矩阵约有 `12 × 24 × 512² = 75,497,472` 个参数。FP16 约 144 MiB，理想 INT4 数据约 36 MiB，后者还未加入尺度和其他参数。减少这部分权重有价值，但不足以独自解释如何压缩数 GB 的整合包。

因此优先清点运行库、辅助模型、重复权重和工作区，再做执行图与算子优化，随后评估分模块量化。缓存优化是否优先取决于执行开销和占比，不能只依据理论字节数判断。

## 使用这些资料的边界

外部项目采用某种技术，只能支持“值得实验”，不能证明 SakuraTTS 已经有同样结果。实现前固定实际引用版本并核对代码、模型和运行库的再分发要求；这里只记录研究出处，不代表已经引入相应依赖。

下一步按 [实施路线](roadmap-20260920.md) 收集本项目证据，按 [基准协议](../../docs/specs/benchmark-protocol.md) 判断优化是否满足目标。
