# Windows GPT 混合精度候选

基线为 `d323b50`，设备为 RTX 5060 8 GB，模型为 Sakura V2ProPlus。此次只改变 GPT 执行精度，声学保持已验证的 FP32，采样、停止、参考和文本前端不变。默认仍为 FP32。

## 运行前固定的检查范围

固定官方四类日文输入和完整 token 历史，分别测量 FP32 与候选的 Prefill、完整 Decode 历史耗时，以及实际分配的权重、KV、工作区字节数。计时包括现有 logits 回传；不把固定历史测试标为完整请求性能。重复运行保存原始数据。

原 FP32 参考继续使用 `atol=1e-4, rtol=1e-5`，不调整容差。候选保留同一严格检查的失败记录。另设工程筛查：所有 logits 有限，固定历史整体 RMS 误差不超过 0.05、最大绝对误差不超过 0.5、整体余弦相似度至少 0.999。此筛查仅排除明显计算异常，不代表采样等价或语音质量验收。

Graph / eager、长句后短句、释放重建分别比较同精度结果。自然生成另用相同 NumPy seed，保存完整文本、token、停止、音频和阶段时间；若输出历史改变，不据此声称相同工作量下的加速。无人工听音和内容核查结果时，候选不升级为默认。

## 实测结果

已完成 RTX 5060 上的固定历史、原始日文完整请求、五参考切换、取消、worker 故障恢复与 CLI。完整数据见 [JSON 记录](data/2026-09-20-windows-gpt-fp16.json)。以下只代表这一组模型和输入，候选未通过内容与听感验收，默认保持 FP32。

### 张量占用

| 实际分配 | FP32 | GPT FP16 | 减少 |
|---|---:|---:|---:|
| 共享 GPT 权重 | 318,617,608 bytes | 159,308,804 bytes | 151.93 MiB |
| 容量 2048 的 K/V | 201,326,592 bytes | 100,663,296 bytes | 96 MiB |
| Decode 工作区 | 26,628 bytes | 15,364 bytes | 11 KiB |

权重和 KV 合计减少 **247.93 MiB**。这是设备数组的实际 `nbytes`，不包含声学、CUDA 库、上下文或分配器保留空间，不是整个引擎显存减半。磁盘模型包未改；半精度仅在加载时生成，没有常驻第二套 FP32 GPU 权重。

GEMM 使用 `cublasGemmEx` / `cublasGemmStridedBatchedEx`，指定 FP32 累积并禁止 split-K 的低精度归约。Prefill 的 QK 输出和 softmax 保留 FP32，概率转成 FP16 后参与 AV 矩阵乘；Decode 的注意力归约、LayerNorm 与最终 logits 使用 FP32。其余主要张量为 FP16，声学仍为 FP32。

### 固定历史的计算与误差

最终一轮按 FP16 → FP32 顺序运行，每类各预热一次、计时五次。每次都用同一官方完整历史；下表是 Prefill 加所有 Decode 的中位数，包含逐步回传 logits，不含前端、采样和声学。

| 输入 / 步数 | FP32 ms | GPT FP16 ms | 差值 ms | 变化 |
|---|---:|---:|---:|---:|
| 短句 / 44 | 92.64 | 75.06 | -17.58 | -18.97% |
| 长句 / 676 | 2335.94 | 2303.62 | -32.32 | -1.38% |
| 多句 / 259 | 633.61 | 583.13 | -50.48 | -7.97% |
| 标点 / 203 | 462.91 | 404.07 | -58.84 | -12.71% |

前一轮 FP32 → FP16 的长句结果为 2326.69 → 2342.72 ms，慢 16.02 ms（0.69%）。两轮说明长句暂未得到稳定的明显加速；不能仅采用最终一轮的负差值宣布所有输入都更快。两轮原始五次数据均保留。

1182 步 FP32 logits 全部通过原容差。FP16 各类 RMS 为 0.01530–0.01685，最大绝对误差 0.07070–0.11228，满足运行前固定的工程筛查，但均不通过 FP32 严格容差；四类严格超差元素数量完整保留。固定历史的 argmax 在 1181 / 1182 步相同，多句有一处变化，这不代表随机采样 token 等价。

两种精度各自的五次重复、其他请求之后再短句、释放后重建和短句 Graph / eager 结果逐位相同。FP32 默认运算未改，新增代码未改变模型包、采样或停止规则。首次尝试遇到转置 BERT 非连续布局，第二次遇到 CuPy 对 NumPy 输入的 API 限制，分别保留在 `fp16-fixed`、`fp16-fixed-v2`；已通过显式 C-order 上传修正，并补回归。

### 原始文本到完整 PCM

两种精度各在新进程中运行 19 次请求：四类各三次热请求、首句、四次切换参考、切回中性及第二个 seed。所有请求均正常结束，普通进程未导入 Torch。下表取无持续资源采样的热请求中位数，参数与参考相同。

| 输入 | FP32 ms / PCM 秒 | GPT FP16 ms / PCM 秒 | 生成历史 |
|---|---:|---:|---|
| 短句 | 253.13 / 3.46 | 226.33 / 3.46 | 语义 token 相同 |
| 长句 | 2874.76 / 25.90 | 2818.63 / 25.46 | 不同，640 → 629 token |
| 多句 | 1302.76 / 14.42 | 1281.94 / 14.26 | 不同，353 → 349 token |
| 标点 | 636.35 / 7.66 | 596.78 / 7.66 | 数量同为 184，历史不同 |

短句少 26.79 ms（10.59%）。首句、热短句、五参考相关短请求和第二个 seed 共十个配对保持相同语义 token，PCM 最大差 1 LSB，仍在原 FP32 容差内；没有宣称逐字节相同。其他三类的自然生成工作量或历史已变化，表中的用时仅描述实际请求，不能作为等工作量加速比，也不能据此判断漏读或音质。

### 完整进程资源与恢复

资源轮与速度轮分开，两精度各用新进程完成短→长→多句→标点→空闲卸载→重载，`nvidia-smi` 每 100 ms 采样并记录进程树 RSS。资源轮会明显拖慢请求，其时间不用于上表。

| 资源指标 | FP32 | GPT FP16 |
|---|---:|---:|
| 全卡首个采样值 MiB | 2304 | 2302 |
| 全卡采样峰值 MiB | 4250 | 3829 |
| 峰值减首值 MiB | 1946 | 1527 |
| CuPy 观察到的 active 峰值 bytes | 530,049,536 | 267,299,328 |
| CuPy 观察到的 pool 峰值 bytes | 613,262,336 | 330,963,456 |
| 进程树 RSS 采样峰值 bytes | 1,964,298,240 | 2,171,432,960 |

本轮全卡峰值少 421 MiB，减初始值后差 419 MiB。但长句自然生成长度不同，且全卡包含桌面负载，这不等于精确的进程独占显存收益。WDDM 逐进程显存不可用，短峰值可能漏采。进程树 RSS 峰值反而多约 197.54 MiB；加载后的主进程 RSS 相近，差距主要在首次计算后出现，原因尚未通过库映射或分配轨迹确认，不能简单归因于权重转换。

2026-09-20 后续复核更正：worker 被终止后的 resident 重试实际使用 GPT FP16；两个 staged 取消/重试用例在第二次创建 Engine 时遗漏了 `gpt_precision`，实际使用默认 FP32。因此旧记录不能支持“全部生命周期用例均已验证半精度”的结论。三个用例当时的状态清理、停止和 token 检查通过，PCM 各有 15–19 个样本相差 1 LSB，原数值容差通过，逐字节检查失败。原 JSON、`passed=false` 和退出码 1 均保留；Harness 已修正为向两次 Engine 构造传递相同精度和 attention，修复后的低精度生命周期需另行验证。

后续 [GPT profile 汇总](data/2026-09-20-windows-gpt-profile.json)在新进程中复现了首次 Prefill 后约 196.6 MiB 的 FP16 额外 RSS，其中 USS 差约 192.5 MiB、实际文件映射驻留差约 4.3 MiB。单独分段诊断发现，首次使用不同输入/输出和批次形式的 cuBLAS GEMM 会触发大块私有常驻增长，关闭模型后仍保留。现有计数器支持“主要来自首次计算后的 CUDA 运行库/驱动私有分配”这一定位，尚不能区分具体内部 allocator 或缓存；没有证据把差额全部归给权重转换或 DLL 映射。每个 token 的 CUDA events 显示长历史约 98% 的 host 用时在 GPU Graph 内，后续 [split-KV 实验](2026-09-20-windows-split-kv.md)据此优先处理注意力计算。

公开 CLI 通过 `--gpt-precision fp16` 完成 3.46 秒短句 WAV，JSON 正确记录 GPT / 声学精度和 `torch_imported=false`。开发环境全量单测运行 118 项，116 通过、2 跳过；符号链接相关验证受当前 Windows 权限限制。最初误用无 Torch 日常环境运行全部开发测试时，两项 Torch 测试无法导入；最终全量测试使用已有 `.venv`，未为测试添加日常依赖。

## 复现与后续

固定历史映射为 `outputs/windows-precision/captures.json`，指向已有四类官方捕获；参考条件与模型 archive 均校验哈希。用新目录运行：

```powershell
.venv-windows-runtime\Scripts\python.exe research/tools/windows_gpt_precision.py `
  --gpt models/windows-sakura/gpt `
  --reference models/windows-sakura/references/中性 `
  --captures outputs/windows-precision/captures.json `
  --output outputs/windows-precision/new-fp16 `
  --precision fp16 --repeats 5
```

随后将精度改为 `fp32`，输出改为另一新目录，并加 `--compare outputs/windows-precision/new-fp16`。正式证据在 `fp16-final` / `fp32-final`；`fp16-natural` / `fp32-natural` 保存完整请求，`fp16-memory` / `fp32-memory` 保存资源，`fp16-lifecycle` 保存失败恢复，`fp16-cli.wav` 为公开入口样音。

完整请求使用 `research/tools/windows_nvidia_benchmark.py --config models/windows-sakura/runtime.json --gpt-precision fp16 --repeats 3 --no-memory-sampler --skip-idle-unload --idle-ms 0 --output 新目录`。资源轮改为 `--repeats 1 --skip-reference-switch --skip-random --idle-ms 500` 并移除 `--no-memory-sampler`。生命周期用 `research/tools/windows_failure_lifecycle.py --config models/windows-sakura/runtime.json --gpt-precision fp16 --output 新目录/result.json`。

下一步先补低精度样音内容和听感，再单独验证声学 FP16；用 profiler 定位长句 Decode 与主机同步，不把半精度直接等同于明显加速。并行的 [Lite 本机对照](2026-09-20-windows-lite-baseline.md)和[运行依赖清点](2026-09-20-windows-runtime-inventory.md)已经提供后续移植依据。声学半精度、GPU 采样、分块音频、约 829 MiB DLL 共用和干净机器部署尚未实现或验收，不计入本轮收益。
