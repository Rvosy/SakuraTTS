# DirectML：显卡选择、解码调度与卸载占用

本轮在 Ryzen 7 7840HS / Radeon 780M 上验证 Session 所有的 GPU KV 缓存。配置保持 AMD FP16、容量 1280、GPT 主机线程 4、声学主机线程 2；CPU 配置未改动。完整请求没有测出稳定提速，明确改善的是卸载后的 GPU 提交量，以及非零显卡编号的缓存分配路径。

## 改动与设备范围

原实现通过独立 OrtValue 工厂分配 GPU KV。锁定的 ORT 1.24.4 内部会固定创建 DXGI 适配器 0 的分配器，传入其他 `device_id` 不能保证缓存归属。现由所选 Decode Session 在前两次正常解码中分配两组 KV，之后交替复用，不增加预热或模型导出步骤。根因与上游源码集中在 [ADR 0006](../../docs/adr/0006-device-precision-and-directml-kv.md#显卡选择与-kv-分配)，选卡用法见[设备指南](../../docs/cpu-amd.md#选择-amd-显卡)。

稳态每 token 的绑定调用从 99 次减为 51 次。输出固定绑定，logits 写入预分配的 CPU 数组；每次请求仅在两组 GPU 缓存初次产生时获取输出句柄。embedding 临时数组复用，mask 每步只更新新增位置。KV 输入仍逐步重绑，以刷新 ORT 图内 CPU 分区可能需要的副本。

初始化失败直接保留原因，不再先整体重试 CPU Session。合法模型输出顺序不再受“logits 必须排第一”的限制；名称、类型和形状契约仍校验。

实机只有 Radeon 780M。非零编号用配置转发、Session 分配替身及小型图回归验证；AMD 独显和多张物理显卡仍未实测。doctor 同时列出了 Microsoft Basic Render Driver，不能把这个软件适配器当作第二张物理显卡的验证。

## 测量方法与证据

[精简证据 JSON](../experiments/data/2026-09-27-directml-host-overhead.json)保留模型、参考音频、脚本和原始结果摘要。完整数据在被 Git 忽略的 `results/dml-overhead-20260927/`。

完整请求采用 N.A.V.I 中性参考及相同日文短句、长句，seed=1234。控制组冻结自 `2b16c37` 的静态 GPT 执行器，其余代码与候选共用。新进程串行运行，先测一次冷短句，每种句长再预热一次、热测三次；计时包括文本前端到完整 PCM 返回，不包括 WAV 写盘。主表合并控制组 `baseline-d` / `control-h` 和候选 `final-f` / `final-g`，每组每种句长各六个热请求。

更早的 a/b/c 是探索轮次。`final-e` 的九次请求也保留在证据中：短、长热测中位数分别为 1.316 / 8.579 秒。该轮加载与删除输出顺序检查重叠，结束时计算的源码摘要不能证明整轮实际加载版本，因此不纳入主表。固定回放的候选源码摘要为 `7df690…`，与最终 `c10469…` 只差这两行加载检查；本地 `.cache/directml_session_owned_before_order_check.py` 已按记录摘要精确复原，原测量摘要未改写。

未锁定频率、温度和桌面后台负载，也未随机轮换运行顺序。这些小样本不能证明统计意义上的提速。

## 完整请求

| 指标 | 控制组 | 本轮实现 | 变化 |
| --- | ---: | ---: | ---: |
| 短句热请求中位数 | 1.276 s | 1.283 s | +0.5% |
| 长句热请求中位数 | 8.250 s | 8.341 s | +1.1% |
| 短句热请求范围 | 1.264–1.289 s | 1.264–1.293 s | — |
| 长句热请求范围 | 8.214–8.327 s | 8.108–8.469 s | — |
| 两轮中最高采样进程树工作集 | 1806.2 MiB | 1818.5 MiB | 未下降 |

进程树工作集每 100 ms 采样，可能遗漏瞬时峰值或重复计入共享页；它不等于完整 GPU 占用。长句主进程累计 CPU 时间中位数从 7.648 CPU-s 降至 7.500 CPU-s，不能据此推算整机 CPU 百分比或功耗。

主表四轮共 36 个完整请求全部成功。同句的采样序列及 WAV 字节一致；短句 PCM 为 3.42 秒，长句为 20.70 秒，均包含产品附加的 0.3 秒尾静音。此处没有通过减少内容或改变音频长度获得加速。

## 固定 token 回放与执行位置

固定输入分别回放短句 78 步、长句 510 步。每个实现、句长各测三次热回放；首次回放单列。普通计时不启用 ORT profiling 或 Python 方法插桩，分段计时与 provider 检查各另跑一轮。

| GPT decode 热回放中位数 | 控制组 | 本轮实现 |
| --- | ---: | ---: |
| 短句 78 步 | 892.582 ms | 894.547 ms |
| 长句 510 步 | 5973.474 ms | 5734.098 ms |

短、长 logits 均有限，且控制组与候选逐字节一致。候选 decode profiling 分别记录 78 / 510 个 DirectML 融合节点事件，没有 CPU 节点事件；这只描述本机本图的 decode，不把它推广到其他驱动或声学图。

长句插桩中 `get_outputs()` 从 510 次减为 2 次，相关主机耗时大幅减少；`run_with_iobinding()` 仍占绝大部分时间。因此固定回放的局部改善没有转化为本轮完整请求提速。各段是主机观察到的墙钟时间，不是纯 GPU kernel 时间。短句候选使用早一版插桩，未单列初始 `bind_output()`，该开销包含在 residual，不能记为零。

## 卸载后的占用

单独的 GPT 回放进程在加载前、回放后、释放请求状态后、关闭模型后采样。每个边界先 GC，再等待 150 ms，全部在正常计时之外；ORT profiling 使用另一会话，安排在这些边界之后。

| 关闭 GPT 后的单 PID WDDM Total Committed | 控制组 | 本轮实现 |
| --- | ---: | ---: |
| 短句回放进程 | 281.98 MiB | 8.56 MiB |
| 长句回放进程 | 282.48 MiB | 8.93 MiB |

原独立分配器在模型关闭后仍保留的 GPU 提交量显著减少。仅调用 `release_request_state()` 时，两种实现的计数均没有下降：Session 内存池仍驻留。这里是 GPT 专项、单 PID 边界计数，不是完整 TTS 峰值、GPU 常驻量或独占显存；不能与 RSS 相加。原始 PDH 状态、缺失实例及专用／共享计数均保留在证据中。

## 复现与验证

使用已准备的静态 FP16 包和已有完整请求记录：

```powershell
python research/tools/directml_gpt_static.py --model models/navi-cpu-amd --result results/genie-comparison-20260927/sakura/amd-fp16-cap1280/directml/result.json --case long --precision fp16 --capacity 1280 --threads 4 --repeats 3 --warmups 0 --timing-breakdown --profile --memory-boundaries --output results/directml-replay
```

`--memory-boundaries` 默认关闭，仅在 Windows 下使用 PDH。设备编号另传 `--device-id`；工具不在计时前重新导出模型。完整语音基准仍使用 `research/tools/cpu_amd_benchmark.py`，每种实现单独启动进程。

本轮产品测试 489 项，488 通过、1 跳过；研究工具测试 151 项，149 通过、2 跳过。锁定 ORT 1.24.4 环境另跑 20 项 DirectML 回归，全部通过。小型真实 GPU ScatterND 探针输出正确。CPU / AMD 各两次 managed 唤醒、合成、休眠成功，所有自有进程退出；AMD 两次 staged 请求均成功，WAV 与 resident 一致。wheel、源码包和 preview ZIP 构建及源文件一致性检查通过。

试听保留在 `results/dml-overhead-20260927/final-f/hot-short-0.wav` 与 `hot-long-0.wav`，固定取该轮第一次热请求，没有按听感筛选或后处理。未新增人工听音、ASR、功耗或独显验收。

本轮未重跑 Genie。此前同机 Genie 原版 CPU 的短、长中位数为 1.489 / 14.540 秒；CPU INT8 为 1.521 / 10.855 秒，AMD FP16 为 1.272 / 8.278 秒。条件、音频长度差异及内存口径见[Genie 对照原记录](genie-comparison-20260927.md)，这些旧数据不作为本轮重测结果。
