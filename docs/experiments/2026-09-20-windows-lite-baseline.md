# Windows / RTX 5060：Lite 本机对照

2026-09-20，使用固定版本 GSV-TTS-Lite、同一组 Sakura GPT / V2ProPlus 权重和中性参考，完成了日文短句、长句、多句的本机测量。Lite 的低显存思路有实测支持：本轮长句的 PyTorch allocated 峰值约 846 MiB。但全卡采样峰值减初始值为 1307 MiB，不能把分配器数字写成整个引擎的显存。

另一个直接影响比较的结果是：Lite 默认 1024 容量会截短这条长句。追加上游公开配置支持的 1536 档后，长句才按其 EOS 规则结束。初次触顶结果仍保留，不计入完整长句性能。

## 环境和边界

- 上游：[GSV-TTS-Lite@6c049397142f4c9147a85f86b6ba37546e93a188](https://github.com/chinokikiss/GSV-TTS-Lite/tree/6c049397142f4c9147a85f86b6ba37546e93a188)，版本 0.4.7，checkout 未修改。
- 独立环境：Python 3.11.15、PyTorch / torchaudio 2.7.1+cu128、NumPy 2.4.6、Transformers 4.51.3、pyopenjtalk-plus 0.4.1.post9，RTX 5060 8 GB。依赖安装在 `outputs/reference-research/lite-runtime`，没有改动已有 SakuraTTS 或 g50 环境。
- FP16，`use_flash_attn=False`，保留 GPT 和 SoVITS CUDA Graph。上游探测后选择 `SDPBackend.EFFICIENT_ATTENTION`；matmul / cuDNN TF32 关闭。
- 日文不加载中文 BERT；参考准备后按 Lite 默认策略释放 HuBERT 和 SV。保留默认 SoVITS `[50,55]` 档，以及 GPT 的三个 batch 1 和两个 batch 4 档；正式轮另追加 `(1,1536)`。
- HuBERT / SV 来自 g50 现有资源的独立副本，g2p 来自上游指定的 [release](https://github.com/chinokikiss/GSV-TTS-Lite/releases/download/g2p/g2p.zip)。推理阶段禁止网络。资源哈希及依赖版本存于原始记录。

模型、参考及原文与 [官方基线](2026-09-20-windows-official-baseline.md) 相同。GPT 和 SoVITS 的 `load_state_dict` 均无缺失或多余键；运行前后的角色模型、参考文件和上游 Python 文件哈希不变。HuBERT 使用 Transformers 自己的分批加载流程，其内部局部加载日志不作为模型缺失键结论。

本轮保留 Lite 原生行为，包括额外 token 抑制、每 5 步检查 EOS、语义切片、波形头部裁剪以及 0.2 秒尾部静音。日文前端为 plus 版。相同 seed 不保证与官方或 SakuraTTS 产生相同 token，也不能用音频时长推定没有漏读。

## 完整 PCM 耗时

入口为 `TTS.infer`，从提交原文到完整 CPU PCM、拼接和有限值检查完成，不含 WAV 写盘。包括上游请求结束时的缓存清理。速度轮关闭持续资源采样，每类先预热一次再测 3 次，seed 为 1234；保留 `top_k=15`、`top_p=1`、温度 1、重复惩罚 1.35、声学噪声 0.5、速度 1。

| 输入 | 完整 PCM 中位数 ms | 范围 ms | 音频 s | 送入声学的 token | 最终 KV / 容量 |
|---|---:|---:|---:|---:|---:|
| 短句 | 268.33 | 265.81–274.98 | 2.04 | 46 | 275 / 1536 |
| 长句 | 2051.02 | 2028.88–2136.94 | 23.38 | 580 | 1070 / 1536 |
| 多句 | 1142.18 | 1120.91–1231.56 | 13.20 | 325 | 634 / 1536 |

三类都在 Lite 的 EOS 检查处结束，未触及容量。模型加载约 2249.26 ms；模型已加载、参考未缓存的首条短句约 1984.28 ms。这里没有测流式首包，完整 PCM 耗时不能标为首包延迟。

默认 1024 档的长句前缀已占 485 个位置：336 个参考加目标音素、149 个参考语义 token。生成用满容量后，返回 539 个语义 token、21.748 秒音频。1536 档则返回 580 个 token、23.38 秒音频，最终 KV 为 1070。扩大容量使用 Lite 原有 `gpt_cache` 参数，没有修改采样或停止代码。初次默认容量轮可能与另一项 CPU 单元测试启动重叠，其耗时仅留在原始记录，不用来比较速度。

正式轮的短、长、多句前缀分别占 225 / 485 / 304 个位置。长句头部裁掉 640 个采样点，即 20 ms；短句和多句没有裁头。上述上游行为仍须单独做听音和内容检查。

## 内存与卸载

资源轮使用新进程，配置同上，每类预热一次、测一次；`nvidia-smi` 每 100 ms 采样，同时记录进程树 RSS。速度取上一节无采样轮的数据。

| 阶段 | allocated MiB | reserved MiB | 本阶段 allocated 峰值 MiB |
|---|---:|---:|---:|
| 模型加载后 | 556.72 | 738 | 568.86 |
| 首句及参考准备后 | 566.58 | 738 | 812.14 |
| 热短句后 | 566.08 | 738 | 587.52 |
| 热长句后 | 601.80 | 792 | 845.58 |
| 热多句后 | 578.55 | 756 | 717.20 |
| 上游卸载两模型后 | 25.37 | 66 | — |

全卡采样初始值为 2370 MiB，峰值 3677 MiB，差 1307 MiB；这包括桌面等其他进程的变化，且 100 ms 采样可能漏掉短峰值。WDDM 下没有可用的逐进程显存数字。进程树 RSS 采样峰值为 3268.21 MiB，卸载模型后的边界 RSS 仍约 2449.93 MiB。未测卸载后重载的恢复代价。

这些数据支持继续迁移 Lite 的 FP16 权重执行、共享缓存和参考模型生命周期。后续需要核对 SakuraTTS 声学权重、工作区与库开销各占多少，才能确定约 0.8 GB 的分配器目标还差在哪里。不能把两项目生成长度不同的自然采样耗时相减，作为某个算子的加速收益。

## 复现与证据

`harness/windows_lite_benchmark.py` 检查固定 checkout、保存运行时依赖和源码哈希、禁止在线下载，分别记录完整返回、容量是否耗尽、语义 token、裁头、PCM 及资源边界。`--preflight-only` 在隐藏 CUDA 后检查依赖和日文前端，本轮通过且没有初始化 CUDA。独立环境的 `uv pip check` 通过。

```powershell
outputs/reference-research/lite-runtime/Scripts/python.exe -B harness/windows_lite_benchmark.py `
  --upstream outputs/reference-research/GSV-TTS-Lite `
  --models-dir outputs/reference-research/lite-models `
  --character D:/Project/sakura/characters/Sakura `
  --output outputs/reference-research/lite-full-fp16-kv1536-rerun `
  --max-kv 1536 --cases short long multi --repeats 3 --no-memory-sampler
```

资源轮去掉 `--no-memory-sampler`，使用新的输出目录和 `--repeats 1`。不要覆盖历史目录。隔离环境的全部固定版本保存在 `outputs/reference-research/lite-runtime-requirements.txt`；资源副本来源及逐文件哈希保存在 `lite-models-provenance.json`。

仓库内的 [测量数据](data/2026-09-20-windows-lite-baseline.json) 保存配置、依赖版本、完整参数、请求数据、文件哈希和容量触顶记录。原始环境、日志、WAV 和采样分别位于：

- `outputs/reference-research/lite-full-fp16-timing`：默认 1024 档，长句触顶。
- `outputs/reference-research/lite-full-fp16-kv1536-timing`：正式速度轮。
- `outputs/reference-research/lite-full-fp16-kv1536-memory`：正式资源轮。

尚未验证：人工听音、ASR 全文、流式首包、FlashAttention 版本、其他角色权重及干净机器部署。Lite 的包体包含 PyTorch，本轮也没有把它作为 SakuraTTS 的发行依赖。
