# 声学参考条件预计算与常驻资源

日期：2026-09-19。当前结论：单参考 V2Pro 的声学预计算保持 10 条固定输入的 WAV 不变，请求结束时 MPS allocated 边界减少 148.99 MiB。RSS 峰值未降低，尚不能称为整体峰值优化；该路径保留为显式实验选项。

## 改动与输出依赖

已有实验提前保存 CNHuBERT 参考语义和 ERes2Net embedding。本轮进一步检查官方声学 decode：`ref_enc`、`sv_emb`、`prelu` 和 `ge_to512` 的结果只依赖当前参考，与目标文本和生成 token 无关。`research/tools/prepared_acoustic.py` 按官方相同表达式提前计算 `ge` 和 `ge512`，保留单参考列表的 `stack().mean()` 操作，然后释放这些模块。训练后验网络 `enc_q` 不在 decode 依赖图中，也一并释放。

| 模块 | 删除的 FP32 参数字节 |
|---|---:|
| `enc_q` | 54,560,256 |
| `ref_enc` | 2,532,352 |
| `sv_emb` | 83,890,176 |
| `prelu` | 4,096 |
| `ge_to512` | 2,099,200 |
| 合计 | 143,086,080（136.46 MiB） |

缓存的两份条件共 6,144 字节，与独立声学诊断捕获的官方实际输入逐元素相同。剩余 quantizer、上采样、文本/语义编码、显式噪声表达式、flow 和声码器保持原计算。Lite 原本没有构造 `enc_q`，这部分不能算作新增超越 Lite 的优化。

实验只接受同一 V2Pro 参考对象、非流式、单参考调用；不同参考会明确报错。原 checkpoint 和参考资源不修改，新参考仍需经过完整准备流程。它尚未成为通用缓存加载器，也不能用于宣称多参考或流式兼容。

## 实际回归

两轮各包含中日文各 4 次生成，候选 WAV 分别与控制组逐字节相同。随后 10 条扩展输入全部保持 WAV 字节，包括两条用户报告样例、短句、长句、标点和中英/日英混合文本。扩展运行同时存在独立 CPU BERT 转换工作，故仅用于正确性，不使用其时间或进程资源作性能结论。

控制和候选正常测量均启用先前验证的 `--prepare-reference --prune-bert`，本轮只增加 `--prepare-acoustic`。正常计时期间没有其他研发 GPU 或重 CPU 工作；每种语言第一次运行单列，下面热运行中位数取后 3 次。所有输入和采样参数相同，FP32、seed 1234、中文 BERT 开启。

| 指标 | 控制 | 声学预计算候选 |
|---|---:|---:|
| 日文热运行中位数 | 1.988 s | 1.943 s |
| 中文热运行中位数 | 2.556 s | 2.399 s |
| 最后中文请求后 MPS allocated 边界 | 1953.19 MiB | 1804.20 MiB |
| 同一边界 MPS driver | 2827.33 MiB | 2747.33 MiB |
| 同一边界进程 RSS | 1615.86 MiB | 1816.84 MiB |
| 进程生命周期 peak RSS | 3117.70 MiB | 3909.80 MiB |
| 卸载后 MPS allocated | 9.01 MiB | 9.01 MiB |
| 卸载后 MPS driver | 107.33 MiB | 107.33 MiB |

这组速度略快，但前一组中文候选中位数为 2.571 s，控制为 2.445 s，方向相反，因此不宣称稳定加速。两组 allocated 减少量相同，候选初始化后的准备阶段增加约 0.2 秒，包含条件存盘；冷启动收益尚未成立。

RSS 边界与生命周期峰值在本组上升，需要继续区分 CPU 内存、GPU 统一内存驻留和声学临时空间。参数字节下降不能替代实际资源指标。以上 MPS 值均是执行边界，不是真实峰值，也不是 NVIDIA 独立显存。

## 独立资源采样复查

随后各启动一个新进程，以相同的两条输入、每条 4 次请求运行控制和候选。此轮启用 10 ms 轮询，没有诊断 trace，也没有其他项目重 CPU/GPU 推理并行。8 份 WAV 仍全部逐字节相同，两进程实际退出码均为 0。采样时间不用于速度结论。

| 资源口径 | 控制 | 候选 |
|---|---:|---:|
| 全程采样 allocated 最大值 | 2548.34 MiB | 2519.09 MiB |
| 请求阶段采样 allocated 最大值 | 2028.22 MiB | 1885.64 MiB |
| 全程采样 driver 最大值 | 6451.33 MiB | 6435.34 MiB |
| 全程采样 RSS 最大值 | 3004.53 MiB | 3046.23 MiB |
| OS 进程全生命周期 peak RSS | 3071.12 MiB | 3070.61 MiB |
| 最后请求后 allocated 边界 | 1953.19 MiB | 1804.20 MiB |
| 卸载后 allocated / driver 边界 | 9.01 / 107.33 MiB | 9.01 / 115.33 MiB |

常驻 allocated 的 148.99 MiB 改善再次出现，请求阶段观察到的 allocated 最大值少 142.58 MiB。但全程 allocated 最大值都发生在更早的参考准备阶段，driver 最大值都发生在第一条中文请求。此轮 driver 观察最大值只少约 16 MiB，尚无整体高水位明显改善的证据。

OS 的 RSS 高水位本轮几乎相同，上一轮候选上升约 792 MiB 没有复现。轮询到的 RSS 最大值均在模型加载阶段；该阶段两边执行相同代码，发生在声学预计算之前。不能将这部分差异归因于新增预计算。首个中文请求的 RSS 采样最大值仍从 2574.17 增至 2801.28 MiB，需保留这种阶段差异，不能只引用卸载或最终请求边界。

两次实际最大采样间隔为 0.310 / 0.315 秒，远大于请求的 10 ms，可能遗漏瞬态峰值。表中的采样最大值只是所观察到的下界；它与 OS 的全生命周期 peak RSS 有不同口径。当前保留这项显式优化用于降低请求后的常驻分配，不宣称它解决了声学临时空间或全程峰值。

## 退出问题与证据

前两次控制进程在完成 WAV 和卸载记录后，以退出码 134 中止。相关栈指向 ORT 遥测退出竞争，处理经过见 [退出诊断](2026-09-19-ort-teardown.md)。本表取随后设置相同遥测 opt-out 的控制与候选，两者都由标准库父进程记录到真实退出码 0。历史崩溃记录未覆盖。

路径相对于 `SakuraTTS-References/`：

| 证据 | 路径 |
|---|---|
| 正常退出控制组 | `runs/20260919T112504.624812Z-official-mps/` |
| 正常退出候选组 | `runs/20260919T112618.136695Z-official-mps/` |
| 8 份 WAV 对照 | `runs/20260919T112740.943057Z-audio-run-comparison/` |
| 热时间、资源与条件数组复核 | `runs/20260919T113242.268582Z-acoustic-lifecycle-comparison/` |
| 10 条候选输出 | `runs/20260919T112741.072548Z-official-mps/` |
| 10 条 WAV 对照 | `runs/20260919T113242.232788Z-audio-run-comparison/` |
| 先前退出异常的原始观察 | `runs/20260919T112420.880704Z-acoustic-lifetime-observation/` |
| 独立内存采样控制组 | `runs/20260919T115622.251165Z-official-mps/` |
| 独立内存采样候选组 | `runs/20260919T115750.404363Z-official-mps/` |
| CSV、波形哈希与采样极值复核 | `runs/20260919T120017.952252Z-sampled-memory-comparison/` |

复现正常请求，去掉最后的 `--prepare-acoustic` 即为控制条件：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-official-macos/bin/python" -u research/tools/run_reference_process.py \
  --references "$REF" --backend official --device mps --repeat 4 \
  --prepare-reference --prune-bert --prepare-acoustic
```

父进程保存完整 stdout/stderr 与 `process-result.json`，其中记录实际退出码和子进程报告的状态，不强制退出、不重试。测量脚本的源代码快照和实际命令随每个新运行保存。

复现内存采样时，在控制和候选命令末尾分别加 `--sample-memory`，两个进程串行运行；随后离线比较：

```sh
"$REF/.venv-mlx-macos/bin/python" research/tools/compare_memory_runs.py \
  --references "$REF" \
  --baseline "$REF/runs/20260919T115622.251165Z-official-mps" \
  --candidate "$REF/runs/20260919T115750.404363Z-official-mps"
```

该入口核对请求身份、正常退出、CSV 与清单极值以及所有 WAV 哈希，并保留最大值出现的阶段。下一项是通过独立声学算子对照继续调查临时分配，保留此处的官方资源基线。
