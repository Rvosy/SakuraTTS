# Windows GPT 单行 GEMV 探针

日期：2026-09-20。Windows 11 / RTX 5060 8 GB，Sakura V2ProPlus，GPT FP16 baseline attention。

CUDA Core 候选在本轮独立线性层探针中用时较低，但五类线性层整体替换未通过原严格数值检查。三个采样点的 QKV 均有失败，FFN-in 在第三点也失败，三次原始状态均为 `numerical_check_failed`。生产 GPT 继续使用 cuBLAS；本轮没有让候选参与完整 GPT 生成、声学合成或音质验收。

`attention_output`、`ffn_out`、最终 `output` 在三个点均通过严格检查，可作为下一轮完整历史实验的有限候选。三个点的通过不能代表全部层、token、文本或模型。

## 输入与计算约束

`research/tools/windows_gemv_probe.py` 分为记录和探针两步。记录步骤使用现有 FP16 `CUDAGPT` eager decode，依次注入原始官方 capture 的 token 历史，保存真实线性层输入及原 cuBLAS 输出。探针读取这些输入和模型包中的真实权重，逐个比较 cuBLAS、`warp4`、`warp8`。

三个记录点来自同一个中性短句历史，文本为「おはよう。今日もよろしくね。」。Transformer 层编号从 0 开始，decode step 从 1 开始：

| 记录点 | Transformer 层 | Decode step | 文本音素 / 参考语义长度 | KV 容量 |
| --- | ---: | ---: | --- | ---: |
| `layer0-step1` | 0 | 1 | 83 / 156 | 2048 |
| `layer12-step16` | 12 | 16 | 83 / 156 | 2048 |
| `layer23-step32` | 23 | 32 | 83 / 156 | 2048 |

每个点的前四类形状取所选 Transformer 层；最终 `output` 始终取经过全部 24 层后的最终投影。这里没有将随机向量、前一轮候选输出或不同层的权重拼成模型输入。

| 形状 | 权重维度 `[output, input]` | 输入 / 权重 | 累积 | 输出 |
| --- | --- | --- | --- | --- |
| QKV | `[1536, 512]` | HALF | FP32 | HALF |
| Attention output | `[512, 512]` | HALF | FP32 | HALF |
| FFN-in | `[2048, 512]` | HALF | FP32 | HALF |
| FFN-out | `[512, 2048]` | HALF | FP32 | HALF |
| Output logits | `[1025, 512]` | HALF | FP32 | FP32 |

基线调用现有 `_GraphBLAS`，保留 math mode 16 和 `CUBLAS_COMPUTE_32F`。候选用一个 warp 计算一个输出行，每个线程按步长 32 读取输入，以 `__fmaf_rn` 累积，再用固定的 `__fadd_rn` shuffle 树归约。`warp4` / `warp8` 只改变每个 block 的 warp 数，计算树相同；三组实测中两者输出始终逐位一致。

## 数值结果

严格检查保持 `atol=1e-4, rtol=1e-5`，参考值仍为同一输入的当前 cuBLAS 输出。下表的两项数字分别是超差元素数和最大绝对误差，`warp4` / `warp8` 的结果相同。

| 形状 | layer 0 / step 1 | layer 12 / step 16 | layer 23 / step 32 |
| --- | --- | --- | --- |
| QKV | **1 / 0.0009765625，失败** | **1 / 0.000244140625，失败** | **1 / 0.0001220703125，失败** |
| Attention output | 0 / 0，逐位一致 | 0 / 0，逐位一致 | 0 / 0，逐位一致 |
| FFN-in | 0 / 0.00006103515625，通过 | 0 / 0.0000152587890625，通过 | **1 / 0.0001220703125，失败** |
| FFN-out | 0 / 0，逐位一致 | 0 / 0，逐位一致 | 0 / 0，逐位一致 |
| Output logits | 0 / 0.0000057220458984375，通过 | 0 / 0.0000057220458984375，通过 | 0 / 0.0000057220458984375，通过 |

所有重复观察均稳定。探针中的 cuBLAS 输出与记录步骤中的 cuBLAS 输出通过严格检查，GPU 候选与 CPU 对相同归约树的仿真逐位一致。内核编译和全部计时正常完成；probe 返回码 1 来自数值失败。

失败位置符合 FP32 归约结果跨过 HALF 舍入中点的情况。例如，layer 0 / step 1 的 QKV 输出下标 581：

- cuBLAS 为 `-1.7578125`，候选为 `-1.7568359375`，差值为 `0.0009765625`。
- HALF 输入与权重的精确乘积和为 `-1.7573241781865363`；相邻 HALF 值中点为 `-1.75732421875`，相差约 `4.06e-8`。
- 该处候选与 FP64 点积再转 HALF 的诊断结果相同，但它仍未通过相对 cuBLAS 的严格检查。

layer 23 / step 32 的 FFN-in 下标 436 则为 cuBLAS `-0.20849609375`、候选 `-0.2083740234375`。精确和 `-0.2084351767989574` 在中点 `-0.20843505859375` 的另一侧。更精确的累积不保证复现 cuBLAS 的舍入方向，也不保证候选处处更接近精确和。

精确和诊断使用实际 HALF 二进制数的有理数乘积求和。完整结果还保留了 cuBLAS 与候选一致、但都偏离 FP64 转 HALF 结果的位置，例如第一点 QKV 下标 1071。诊断没有取代原检查；argmax 一致也不代表采样、内容或音质通过。

## 独立计时

每种方法预热 10 次，每种计时方式重复 50 次，执行顺序固定为 cuBLAS、warp4、warp8。CUDA Graph 内重复 32 次同一线性运算，输出不会成为下一次的输入。同一矩阵和向量反复读取，形成缓存较热的微基准，不能据此推算完整 GPT 的耗时占比或端到端加速。

下表为 CUDA event 测得的图总时长除以 32 后的中位数，单位 µs。每格依次是 **cuBLAS / warp4 / warp8**，没有根据各点结果事后挑选更快的候选。

| 形状 | layer 0 / step 1 | layer 12 / step 16 | layer 23 / step 32 |
| --- | --- | --- | --- |
| QKV | 10.1575 / 3.0210 / 3.0115 | 10.2895 / 2.9885 / 2.9985 | 10.1555 / 2.9480 / 2.9705 |
| Attention output | 4.7540 / 2.1855 / 2.1185 | 4.9525 / 2.0630 / 2.1820 | 4.9835 / 2.1840 / 2.1595 |
| FFN-in | 12.8420 / 3.3370 / 3.4740 | 12.9970 / 3.4840 / 3.3290 | 12.7610 / 3.3975 / 3.4380 |
| FFN-out | 5.5635 / 3.8210 / 4.2650 | 5.8210 / 3.8530 / 4.1805 | 5.5150 / 3.7915 / 4.1065 |
| Output logits | 6.4365 / 2.4910 / 2.5640 | 6.5525 / 2.6020 / 2.7525 | 6.4645 / 2.7230 / 2.9070 |

另用不插入 CUDA event 的调度测量图启动至 stream 同步返回的墙钟时间。下表仍为 cuBLAS / warp4 / warp8，单位 ms，覆盖整个 32 节点图。

| 形状 | layer 0 / step 1 | layer 12 / step 16 | layer 23 / step 32 |
| --- | --- | --- | --- |
| QKV | 0.31530 / 0.08580 / 0.08590 | 0.31540 / 0.08580 / 0.08585 | 0.31530 / 0.08575 / 0.08590 |
| Attention output | 0.14390 / 0.05720 / 0.05740 | 0.14355 / 0.05700 / 0.05730 | 0.14460 / 0.05700 / 0.05720 |
| FFN-in | 0.40110 / 0.09815 / 0.10165 | 0.40120 / 0.10250 / 0.10010 | 0.39790 / 0.09820 / 0.09840 |
| FFN-out | 0.16790 / 0.11050 / 0.14040 | 0.17600 / 0.11445 / 0.12260 | 0.16780 / 0.11245 / 0.12260 |
| Output logits | 0.19630 / 0.06930 / 0.07335 | 0.19740 / 0.06940 / 0.07350 | 0.19635 / 0.09010 / 0.07320 |

event 与墙钟来自不同调度，不能相减求 CPU 开销。另存的 eager event 区间可能包含主机入队不及时造成的空隙；主机 enqueue 耗时也不表示设备已经完成。权重加载、输入记录、编译、分配、预热、输出复制与比较、归档写盘均在计时外。本轮没有持续资源采样，也没有得到新的显存结论。

运行环境为 Python 3.11.15、CuPy 14.2.0、NumPy 2.4.6、CUDA runtime 12090、driver 13030、cuBLAS 120902，GPU compute capability 12.0。运行时未导入 PyTorch。

## 论文依据与下一轮范围

本轮核对了以下原始论文，并通过 arXiv 检索了 2025-01-01 至 2026-09-20 范围内的小 batch 解码、GEMV 和 GPU attention 研究。检索有明确范围，不代表穷尽全部近期论文。版本、原文哈希和查询记录见证据 JSON。

| 原始来源 | 对当前问题的启发 | 适用边界 |
| --- | --- | --- |
| [FlashDecoding++ §5](https://arxiv.org/html/2311.01282v4)，v1 为 2023-11-02，所读 v4 为 2024-01-05 | 按模型的少数固定矩阵形状，实测 CUDA Core GEMV 与 Tensor Core GEMM 的选择点。batch=1 的单行运算值得单独实现和计时。 | 论文的 Llama2 / A100 结果及更大的矩阵宽度不能直接套用到 512 维 GPT。未采用其统一最大值 softmax。 |
| [FlashInfer §3.2.3、附录 D.2](https://arxiv.org/html/2501.01005v2)，v1 为 2025-01-02，所读 v2 为 2025-04-21 | 单 query tile 使用 CUDA Core；短 KV 不拆分时直接写最终输出，省去部分结果工作区和归并。 | 可用于下一轮 `head_dim=32` 注意力的访存与调度设计。本轮未实现或测量新注意力内核，论文也未验证本机配置。 |
| [FlashAttention-4](https://arxiv.org/html/2603.05451v1)，v1 为 2026-03-05 | 依据特定硬件中矩阵计算、共享内存和指数运算的相对成本组织流水线。 | 原文聚焦 B200 / GB200 的 BF16 吞吐、异步 MMA 和 Tensor Memory。RTX 5060 同属 Blackwell 并不足以支持迁移这些性能结论；其 softmax 近似与反向计算优化未纳入本轮。 |
| [High-Performance Portable GPU Primitives… §V-C、VII](https://arxiv.org/html/2603.18695v1)，v1 为 2026-03-19 | 归约轴、连续访存、矩阵长宽比与线程划分需要一起调节。 | 主要相关结果是 A40 上 Float32、总元素数 `10^7` 至 `10^9` 的列主序 Julia 矩阵运算。本项目更小的行主序 HALF 形状仍需实测，不据此引入 Julia 或承诺提速。 |

下一轮先将 `attention_output`、`ffn_out`、`output` 放入单独、显式启用的开发实验，覆盖完整固定历史，比较每步 logits、采样与停止行为，再测完整请求。QKV 和 FFN-in 保留 cuBLAS，当前失败继续记录。没有必要只为复现 HALF 中点附近的某个结果而追逐 cuBLAS 的内部归约树；后续候选仍须通过完整模型的内容与音质检查。

注意力候选另行设计和测量，保留 FP32 累积、稳定 softmax、完整 KV、既有固定 CUDA Graph，避免增加逐 token 主机调度。两类候选的收益最终都由同配置完整请求计时确认，不能把本轮五个微基准的时间相加当作请求加速。

## 复现与证据

先只检查配置与身份，默认不会导入 CUDA 后端：

```powershell
.venv/Scripts/python.exe -B research/tools/windows_gemv_probe.py --mode record `
  --gpt models/windows-sakura/gpt `
  --capture outputs/windows-baseline/fp32-naive/diagnostic-neutral-short.npz `
  --reference models/windows-sakura/references-cuda/中性 `
  --layer 0 --decode-step 1 --check-only `
  --output outputs/windows-gemv/new-check-layer0-step1
```

实际记录和探针须显式启用 GPU，并分别使用新目录：

```powershell
.venv-windows-runtime/Scripts/python.exe -B research/tools/windows_gemv_probe.py --mode record `
  --gpt models/windows-sakura/gpt `
  --capture outputs/windows-baseline/fp32-naive/diagnostic-neutral-short.npz `
  --reference models/windows-sakura/references-cuda/中性 `
  --layer 0 --decode-step 1 --run-gpu `
  --output outputs/windows-gemv/new-record-layer0-step1

.venv-windows-runtime/Scripts/python.exe -B research/tools/windows_gemv_probe.py --mode probe `
  --gpt models/windows-sakura/gpt --layer 0 `
  --inputs outputs/windows-gemv/new-record-layer0-step1 --run-gpu `
  --warmup 10 --repeats 50 --graph-nodes 32 `
  --output outputs/windows-gemv/new-probe-layer0-step1
```

另两点将 record 的 layer / decode-step 改为 12 / 16 和 23 / 32；probe 的 layer 也必须同步改为 12 和 23，并指向对应记录目录。模型包、准备参考和原始 capture 均为本地实验输入，仓库克隆本身不包含这些资产。

[证据 JSON](data/2026-09-20-windows-gemv.json)保留三个原始失败状态、全部数值比较、计时的样本数与 min / p50 / max、模型与参考身份、源码及原始文件哈希。完整时序和数组保存在 `outputs/windows-gemv`，该目录未跟踪；哈希用于识别证据，不表示远端仓库已经携带这些归档。

独立 CPU 审计核对了 15 个所选 HALF 权重实例、30 个记录输入 / 输出、120 个探针数组、45 个不同测量输出及 6750 次输出哈希观察，并从原始值重算数值与计时统计。三组均确认源文件未变、每次输出观察能对应到保存的数组，审计没有重跑 GPU。第一点旧汇总保持原文件，另外两点使用参数化审计脚本；各脚本及汇总哈希一并保存。

已实际运行探针的 6 项 CPU 单元测试，全部通过；真实模型 / capture / reference 的 `--check-only` 通过。三个 GPU record 均返回 `captured`，三个 probe 均保留 `numerical_check_failed`。本轮未运行候选完整请求，未做人工听音或 ASR 内容验收。
