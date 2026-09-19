# FP32 运行权重的无损存储

2026-09-19，GPT、SoVITS 和中文 BERT 三个转换归档合计减少 840,298,756 字节，约 801.37 MiB。新包加载后的 1,303 个运行权重张量全部为 FP32，与旧包逐位一致。原包和原模型均保留。

## 存储规则

`scripts/repack_weights.py` 逐张量读取原 FP32 包，实际执行 FP32 → FP16 → FP32，再按 `uint32` 位模式比较。通过的张量以 FP16 存储，未通过的继续存 FP32。每个张量单独写入 NPZ 的一个未压缩 ZIP 成员；转换器不会同时持有整包的 FP32 与 FP16 副本。写完后再逐张量读取新旧包，复核展开结果。

GPT 的 `position_encoding[4000,512]` 未通过 FP16 往返，保留 FP32。其余 295 个 GPT 张量、全部 650 个 SoVITS 张量和 357 个 BERT 张量均通过。这改变了磁盘存储，运行时仍恢复原 FP32 权重，既有 FP64 Prefill 仍使用 FP64 运算。

新清单保留原 `source`、`tensor_sources`、配置和转换来源，新增 `parent_manifest_sha256`，并原样保存 `parent_manifest.json`。`weights.storage` 显式声明 `lossless-fp16-or-fp32-v1`、运行 dtype，以及每个张量的存储 dtype、展开 dtype、形状、存储字节 SHA-256 和展开 FP32 字节 SHA-256。原 `tensor_sources` 中的 FP32 dtype、大小和 hash 仍描述原转换权重；当前归档 payload 大小记录在 `weights.raw_tensor_bytes`，展开大小单独记录。

原包同目录中的许可证、转换脚本等普通附件也原样复制，记录并复核 SHA-256。GPT 和 SoVITS 包中的 `GPT-SoVITS-LICENSE` 引用因此能在新包内解析。BERT 原包只有清单和权重；其中 `source_files` 描述原模型来源，实际运行配置已保存在清单中。

`src/sakuratts/weight_storage.py` 是 NumPy 读取工具。五个 MLX 加载器读取新包时先检查声明、存储数组和展开 hash，再把 FP32 数组交给 MLX。未知存储格式或未声明的 FP16 不会被静默接受。GPT 和 BERT 的旧 FP32 包保留原 `mx.load` 路径。

## 实际大小与验证

| 包 | 原 NPZ 字节 | 新 NPZ 字节 | 减少字节 | 新包存储张量 |
|---|---:|---:|---:|---|
| GPT | 318,699,510 | 163,486,706 | 155,212,804 | 295 FP16 + 1 FP32 |
| SoVITS | 173,247,422 | 86,722,238 | 86,525,184 | 650 FP16 |
| 中文 BERT | 1,197,233,952 | 598,673,184 | 598,560,768 | 357 FP16 |

以上是归档文件的实际大小，未扣除新清单与实验来源记录的少量开销，也不代表原模型、环境和旧证据已从本机删除。运行时权重字节数保持不变：GPT 318,617,608，SoVITS 173,050,368，BERT 1,197,121,536。

验证分为三层：

- 14 项小数组检查覆盖正负零、精确 FP16 值、必须保留的 FP32 值、直接转 FP64 的一致性、未知存储和错误 hash 的拒绝，以及许可证和转换脚本的原样保留。
- `harness/lossless_storage_replay.py` 通过五个真实 MLX 加载器读取新包，将所有运行权重与原归档逐位比较。1,303 项全通过；每包释放后 MLX active 均为零；进程未导入 PyTorch。
- 日文、中文用户报告样例经过新旧 GPT 包的 FP64 Prefill 后，logits 和每条请求的 48 个 KV 张量逐位一致。小规模 GPT FP32/FP64 回归也通过。

SoVITS 扩展对照分开运行旧包和新包。两轮共 10 条固定输入的 120 个阶段数组逐位相同，10 个 PCM WAV 的 SHA-256 也相同。两轮的同一条 `ja-punctuation` 在 MRTE 阶段各有一个元素超过原容差，均保留 `numerical_mismatch`；所有最终 waveform 通过原容差。这个已有的中间结果问题仍需继续诊断，存储等价验证没有把它改写成通过。

转换器单进程峰值 RSS 分别为 GPT 88,915,968 字节、SoVITS 80,412,672 字节、BERT 325,615,616 字节。它们包含 Python、NumPy 和当前张量的往返/复核副本，是 OS 进程生命周期峰值，不能当作 GPU 占用。

首次紧凑归档实现有恢复与 hash 校验开销，当时 GPT FP64 Prefill 还会在每次请求中重读权重。对两条用户报告样例执行相同固定历史、1024 KV 容量、2 次预热和 5 次正常测量，得到以下结果：

| 样例 | 原 FP32 归档中位耗时 | 无损紧凑归档中位耗时 | 变化 |
|---|---:|---:|---:|
| 日文，121 步 | 0.83140 s | 0.96048 s | 慢 15.53% |
| 中文，147 步 | 0.95649 s | 1.10750 s | 慢 15.79% |

两轮所有步的 logits 逐位相同，也通过官方原容差。正式计时没有插入每步 CPU logits 复制、诊断计时器或内存采样。计时外的 Prefill 分段记录显示，日文的权重读取、恢复和校验由 0.05577 s 增至 0.19477 s，计算与其它处理为 0.17585 s、0.17535 s。

随后已按这个证据[改为复用加载时校验完成的 FP32 权重](2026-09-19-gpt-prefill-weight-reuse.md)，请求期间不再重读归档。相同紧凑包的两条中位耗时降为 0.79968 s、0.91575 s，保留了安装大小收益，也消除了这轮测得的重复读取代价。上表保留为优化前的实测记录。

磁盘归档减小不直接减少 FP32 运行权重或声学生成峰值；安装收益与上述请求耗时代价需要同时考虑。

## 文件与复现

以下路径相对于 `/Users/beyondpower/Documents/Projects/SakuraTTS-References`：

- GPT 新包：`models/converted/20260919T123226.764557Z-20260919T104221.749176Z-010197bfc30b-gpt-fp32-lossless-storage`。
- SoVITS 新包：`models/converted/20260919T123227.630582Z-20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32-lossless-storage`。
- BERT 新包：`models/converted/20260919T122406.949871Z-package-lossless-storage`。
- 小数组检查：`runs/20260919T123226.677645Z-lossless-storage-self-test`。
- 真实加载器及 GPT Prefill：`runs/20260919T122618.594757Z-lossless-storage-runtime`。
- 小规模 GPT 回归：`runs/20260919T122922.654721Z-mlx-gpt-self-test-cpu`。
- 原包 GPT 正常计时：`runs/20260919T123305.914822Z-gpt-benchmark-mlx-fp64-prefill-gpu-kv1024`。
- 紧凑包 GPT 正常计时：`runs/20260919T123403.098372Z-gpt-benchmark-mlx-fp64-prefill-gpu-kv1024`。
- SoVITS 旧包扩展：`runs/20260919T122652.881973Z-mlx-sovits-complete-gpu`。
- SoVITS 新包扩展：`runs/20260919T122739.465235Z-mlx-sovits-complete-gpu`；同目录保存 `storage-equivalence.json`。

每个新包保存 `repack_result.json`，其中记录完整 argv、逐张量结果、运行时间、原/新归档 SHA-256；`repack_source/` 保存转换器及读取工具的副本。运行 Harness 的证据目录保存实际执行源码和输入来源。

上述 GPT 和 SoVITS 包已补齐原包附件，与运行验证时所用的初版 compact 包具有完全相同的 `weights.npz` SHA-256：分别为 `9b03385158415665b9637ca303478a8628b76847cb43d04a122dc308f3c81070` 和 `d01cf1f09794c7e995766bab21d52da332289ef6fa8f0008c27db9905e2ab799`。早期实验目录没有被覆盖。

在 SakuraTTS 根目录执行，输出默认写入新的带时间戳目录：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" scripts/repack_weights.py \
  --references "$REF" --self-test
"$REF/.venv-mlx-macos/bin/python" scripts/repack_weights.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T104221.749176Z-010197bfc30b-gpt-fp32"
```

`harness/lossless_storage_replay.py` 接受多个 `--package` 和一个 `--official-gpt-run`。上述真实加载器运行目录的 `result.json` 保存了三包联合验证的完整命令。
