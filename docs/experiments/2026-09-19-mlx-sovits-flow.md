# MLX V2Pro 反向 Flow 固定输入对照

2026-09-19，在 Apple M4、16 GB 统一内存的 Mac 上，独立 MLX 实现通过了两条固定官方输入的逐元素对照。实现只加载转换包中的 `flow.*` 参数，不导入 PyTorch、Lite 或官方推理类。

## 实现与边界

`src/sakuratts/mlx_sovits_flow.py` 实现官方 `ResidualCouplingBlock` 的反向路径。执行顺序是 `Flip 7 → Coupling 6 → Flip 5 → Coupling 4 → Flip 3 → Coupling 2 → Flip 1 → Coupling 0`。每个 Coupling 保留 WN 的条件切片、tanh/sigmoid 门控、残差、skip 累加和掩码位置；`mean_only=True` 的 log scale 恒为零。

转换包保留原始 `weight_g`、`weight_v`，运行时按元数据中的 `dim` 计算 `v * (g / sqrt(sum(v*v)))`，归约除该维度外的轴。卷积内部使用 NTC，公开接口使用 NCT，MLX 数组可直接衔接，无须复制到 CPU。

当前范围是已转换的朱雀院红叶 V2Pro FP32、batch 1，以及来自同一检查点的 `ge`。本轮固定 `flow_input`、`mask`、`ge`，验证到 `flow_output`；没有生成波形、执行 ASR 或试听，也没有扩大模型兼容声明。

## 验证结果

官方参考提交：`48b1a0169a28582a8984402f82cf438d3bfa6aca`。

| 固定样例 | 输出形状 | 最大绝对误差 | RMS 误差 | 超差元素 |
|---|---|---:|---:|---:|
| 日文用户报告样例 | `[1,192,240]` | 1.04904e-5 | 5.93143e-7 | 0 |
| 中文用户报告样例 | `[1,192,292]` | 4.76837e-6 | 5.16368e-7 | 0 |

判据保持 `abs(actual - expected) <= 1e-4 + 1e-5 * abs(expected)`，没有调整容差。两条输出均有限。

另有 15 项小规模 CPU 检查，以独立 NumPy FP64 标量卷积和 NCT flow 实现为对照，覆盖三个 weight norm 轴、分组与空洞卷积、反向顺序、完整掩码及带孔掩码。既检查 NumPy 输入，也检查 MLX 输入。带孔位置在完整反向流程结束后精确为零。两类检查均确认进程未导入 PyTorch 或上游推理模块。

## 资源数据

只加载 flow 的 124 个原始参数张量，共 53,623,296 字节（约 51.14 MiB）。MLX 加载后 active 为 53,688,832 字节；本轮带阶段捕获的 allocator peak 为 211,026,960 字节（约 201.25 MiB）。删除模型及输出并清理缓存后，active、cache 均为零。

这些是 MLX 分配器统计，不是进程 RSS，也不是 NVIDIA 独立显存。捕获保留了八个中间阶段，峰值包含诊断开销。两条诊断耗时分别约 66.2 ms、9.3 ms，受到冷启动、阶段求值及 CPU 复制影响，不能作为正常生成延迟或速度收益。

## 原始证据与复现

以下路径均相对于 `/Users/beyondpower/Documents/Projects/SakuraTTS-References`：

- 转换包：`models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32`，权重归档 SHA-256 为 `752e7ed40ec04014ec38b61dee4fba0310e075c89afb6e2ee9246abf7bd6996c`。
- 官方固定条件：`runs/20260919T111545.846735Z-sovits-fixed-official-mps`。
- GPU 逐元素结果：`runs/20260919T120204.632584Z-mlx-sovits-flow-gpu`。
- 最终 CPU 算子结果：`runs/20260919T120248.854719Z-mlx-sovits-flow-self-test-cpu`。

每个运行目录保存源码副本、依赖版本、复现 argv 和 SHA-256。GPU 目录还保存官方输入、官方与 MLX 输出，以及每次 Flip/Coupling 的输出。保存的中间阶段只用于后续定位，本轮官方证据只有完整 flow 的输出，不能声称八阶段已逐一与官方比较。

在 SakuraTTS 根目录运行：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/mlx_sovits_flow_replay.py \
  --references "$REF" --self-test
"$REF/.venv-mlx-macos/bin/python" harness/mlx_sovits_flow_replay.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T111545.846735Z-sovits-fixed-official-mps" \
  --device gpu
```

下一步把已通过独立对照的 encoder、flow 和 waveform decoder 接起来，检查累计误差，再在关闭阶段捕获的同条件请求中测量延迟和资源占用。
