# Windows CUDA 后续实验：采样、Decode 与声学工作区

核对日期：2026-09-20。范围是当前 RTX 5060 8 GB、Sakura V2ProPlus、batch=1 的自有 CUDA GPT 与 ORT 声学路径。本研究子任务只读取原文、官方文档、实现和同轮 profiler 结果，没有安装依赖或运行 GPU。中文说明按 `humanizer-zh` 校订。

结合本轮独立 profiler 结果，建议先小范围测试长句 Decode 的 KV 分块，GPU 采样与多步图放到后续。声学方面先验证有限感受野的 vocoder 子图分块，保留完整编码器和 reverse flow。现有权重、前端、停止与采样契约仍是约束。

## 已核对的资料及适用条件

| 原始资料 | 时间与证据 | 对当前工程的启发与限制 |
| --- | --- | --- |
| [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/html/2501.01005v2) | 2025-01-02 提交，2025-04-21 修订，MLSys 2025；已读全文的 §3、§4 与附录 D | 动态 attention 调度、固定 workspace 与 CUDA Graph 兼容。实验为 A100 40 GB SXM、H100 80 GB SXM，CUDA 12.4、PyTorch 2.4.0、FP16。调度器仍在 CPU 每步生成 plan 并异步上传，不能把它解读成整个生成循环没有主机参与。 |
| [LeanAttention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers](https://arxiv.org/html/2405.10480v2) | 2024-05-17 提交，2025-01-14 修订；已读全文的 §III–VI | 沿 KV 长度分配工作，减少固定分块造成的 SM 空闲。实验包含 A100 80 GB、H100 SXM 80 GB，head dimension 主要为 64/128，上下文可至数十万 token。当前模型 head dimension 为 32，实际请求短得多，论文收益不能直接外推。 |
| [Sorting-Free GPU Kernels for LLM Sampling](https://flashinfer.ai/2025/03/10/sampling.html) | FlashInfer 团队官方技术说明，2025-03-10；已读算法与实现说明 | 用融合内核与 rejection sampling 降低大词表排序和多次 kernel launch 成本，展示过单 H100 的 vLLM 采样结果。Sakura 词表只有 1,025 项；更值得测的是 CPU 概率计算和同步，而不是照搬为大词表设计的采样算法。 |
| [Dynamic Control Flow in CUDA Graphs with Conditional Nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/) | NVIDIA 官方技术说明，2024-05-10；正文注明 2025-02-03 更新 CUDA 12.8 内容 | WHILE 节点可在设备端更新条件并循环执行图。文章给出 CUDA C API 示例，不证明当前 CuPy 包装层已经支持。需要先验证本机驱动、图捕获与条件节点接口；它不是现成的 TTS 解码器。 |
| [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models](https://arxiv.org/html/2412.10117v3) | 2024-12-13 提交，2024-12-25 修订；已读全文的 §2.4–2.5 | 流式声学来自 chunk-aware causal flow matching；训练时在 non-causal、full-causal、chunk-M、chunk-2M mask 间随机选择。这需要相应训练与模型结构。原文这一机制段落没有给出可直接套用到 RTX 5060 的延迟条件，本报告不引用其延迟作为本机目标。 |

另外核对了 [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)。它发表于 2023-10-12，是本轮所讨论 KV 分块方法的较早来源。原文在 A100、FP16、batch=1 的长上下文条件下测量，并明确指出小 prompt 时各实现表现接近。这里使用它解释方法，不将其列为新论文。

FlashInfer 的 [安装文档](https://docs.flashinfer.ai/installation.html) 当前仍声明 **Linux only**，Python 包基于 PyTorch。已核对的 [2026-09-19 实现快照](https://github.com/flashinfer-ai/flashinfer/tree/0a761d12ae3ee6867278f864b26bdfd6cb4fd100) 虽列出 SM 12.0 的 RTX 50 支持，但也注明并非所有功能覆盖所有架构；[采样模块](https://github.com/flashinfer-ai/flashinfer/blob/0a761d12ae3ee6867278f864b26bdfd6cb4fd100/flashinfer/sampling.py) 直接导入 Torch。当前 Windows 无 Torch 运行环境适合借鉴其算法和布局，直接加入整个 Python 包会改变部署边界。

本轮主要核对了 2024–2025 年方法及其截至检索日的官方实现，没有完成所有 2026 年论文的系统检索，也不据此称这些资料为各方向的最新论文。

## 近期可测的三个候选

本机已有的 `outputs/windows-gpt-profile/fp16/result.json` 记录三次固定历史回放：长句 Decode 主机侧平均 3.39398 ms/token，CUDA Graph 设备区间平均 3.32169 ms/token，二者差 0.072289 ms，约为主机侧时间的 2.13%。实际 KV 位置为 529–1203，短句为 239–281，均不到 2,048。结果支持先研究 GPU 图内执行，而不是假定主机往返主导长句。

这组计量只覆盖固定历史的 `decode()`，不包含 NumPy 采样、前端或声学；2.13% 也不是完整生成的 CPU 占比。图外差值还包括传输、输入检查、同步和 Python。逐算子 CUDA event 的 eager 诊断提示 attention 值得测，但会改变提交与执行节奏，不能把其时间直接除以正常 Graph 时间得到算子占比。

### 1. 仅在长序列候选中测试 Split-KV Decode

当前 attention 每个 head 启动一个 block，因此本模型每层只有 16 个 attention block。KV 固定预分配，head dimension 为 32；沿已有序列长度再分块有机会增加并行度。Flash-Decoding、LeanAttention 和 FlashInfer 都支持这个方向，但它们主要解决长上下文下的占用问题。

最小实验保持现有 KV 布局，增加每个 head 的少量 split。每块保存局部输出和 softmax 归一化统计，再做一次有缩放的合并；不能直接相加各块独立归一化结果。固定 scratch 地址供 CUDA Graph 使用，先避免引入多请求调度、分页 KV 和每步 CPU plan。

先覆盖本机真实的 239–281 与 529–1203 位置区间，再按可容纳的长度扩展到 128、256、512、1,024、2,048 等诊断桶。比较 attention 内核、完整 GPT 与完整请求耗时，同时报告 scratch 和 pool 增量。这里的“长句”仍明显短于论文常见的 8k–256k 上下文，额外 kernel 与归并成本可能抵消收益；只有出现稳定交叉点才按长度选择路径。FP32 继续使用原参考容差，FP16 单独保留误差与采样变化记录。

这个候选不会减少 KV 容量，反而会增加少量临时空间。因此它服务于长句速度，不能用作“显存降到 0.8 GB”的证据。也不应把 Hopper 专用指令或 head dimension 64/128 的调参结果直接用于本机。

### 2. GPU 采样，随后验证少量连续 Decode 步留在设备上

这项保留为后续候选，先补含采样的时间分解。当前固定历史 profile 已表明，仅移除 `decode()` 的图外开销不应期待大幅改善长句速度。

当前 [`CUDAGPT.decode()`](../../src/sakuratts/cuda_gpt.py) 每步用 `state.set()` 上传 token 与位置，再通过 `cp.asnumpy()` 回传 1,025 个 FP32 logits 并同步。随后 [`generate_semantic()`](../../src/sakuratts/generation.py) 在 NumPy 完成重复惩罚、Top-k、随机抽样和停止判断。单步图已经覆盖 Transformer，但还没有覆盖这一段控制循环。

先增加独立实验路径，将重复惩罚、温度、Top-k、softmax、采样和停止标志放进同一设备执行路径，只取回 token 与摘要。诊断时仍可按需回传完整 logits。第一阶段即使只回传一个 token，也仍有逐步同步；不能把传输字节变少称为同步已经消失。随后才试固定少量步骤或 conditional graph 循环，测主机调用次数和实际等待时间是否下降。

必须保留 [`sampling.py`](../../src/sakuratts/sampling.py) 的具体行为：相同历史 token 只处罚一次，Top-k 阈值处保留并列项，前 11 步屏蔽 EOS，停止判断使用处罚后、过滤前的 argmax，随机抽样使用 `argmax(probabilities / exponential_noise)`。FlashInfer 的 inverse/rejection sampler 即使分布正确，也不意味着共享噪声下输出相同。先上传已有固定噪声回放，之后再评估设备 RNG；批量预生成噪声还需核对早停后 RNG 消耗是否改变。

验收至少包括固定 logits/噪声的 token 与停止等价、短句/长句和 worker 生命周期回归，再测完整请求。连续多步图还要处理取消：目前取消回调在每步边界检查，不能为了少一次主机参与而默默延后这一边界。小词表上若融合内核和状态维护比现有 NumPy 更慢，就保留当前路径。

### 3. 固定整句 latent，验证 vocoder 的带重叠分块

[`export_sovits_onnx.py`](../../scripts/export_sovits_onnx.py) 已明确区分 `flow_output`、`decoder_input` 和最终 waveform；现有独立 [vocoder 实现](../../src/sakuratts/mlx_sovits_decoder.py) 展示了卷积、转置卷积和局部残差结构。这提供了一个可验证的子图边界，但 Mac 实现的布局和内存结论不能直接当作 Windows ORT 结果。

先保留完整非因果编码器和 reverse flow，在相同语义、参考和噪声下固定整句 `decoder_input`。从这份模型的卷积参数推导左右感受野和各级步幅，再导出独立 vocoder 子图，按重叠区域计算并裁去边界。对照整句 PCM、各块接缝、整段时长、峰值工作区和总耗时，特别检查最后短块与转置卷积的相位对齐。不能先拍一个 overlap 长度，再用淡入淡出掩盖数值差异。

它可能限制 vocoder 随长句增长的激活和工作区，但会增加重叠计算，也不降低已完成的编码器/flow 峰值。若编码器或 flow 才是主要峰值来源，这个试验收益会很小。它仍需要先获得整句 latent，所以不等于从语义生成阶段就开始输出首包。

CosyVoice 2 的启发是先确认因果性与训练前提。当前模型包含非因果声学计算，不能直接切断整条 SoVITS 图，再把所得音频称为与整句等价的流式输出。真正从未完成语义前缀开始的声学流式，需要另行定义质量验收和模型兼容范围。

## 先后顺序与证据

固定历史 profile 已把长句优先级指向图内执行。先做小规模 Split-KV 对照，收益不足就保留现有 attention；再补含采样的阶段计时，决定是否推进 GPU 采样和多步图。声学试验先验证峰值属于哪一段。每项分别报告时间、空间和误差，不把不同生成长度的耗时或论文设备上的提升合并成一个数字。

成功获取的网页正文、原始 HTML 与官方代码快照保存在本机 `outputs/windows-literature-20260920/`，包括三组 `sources*.json`、对应阅读文本及 `leanattention-v2.txt`。论文全文和官方资料链接均对应本轮成功获取的内容；GitHub 页面指向已核对代码的固定提交。本研究子任务未运行 GPU，上述本机耗时来自同轮独立 profiler 证据；未新增吞吐、显存、首包延迟或音质结论。
