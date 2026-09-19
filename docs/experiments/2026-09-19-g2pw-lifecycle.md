# G2PW 三轮加载与释放观察

日期：2026-09-19。同一个独立 CPU 进程连续创建三次 G2PW Session，每次处理相同短输入，再关闭 Session 并执行 GC。三轮的标签、置信度都与已保存的官方结果完全一致。关闭后的 RSS 在第二轮达到 380.14 MiB，第三轮没有继续增长，进程实际退出码为 0。

## 实验条件

本轮只调查上一轮发现的 Session 关闭后 RSS 未回到初始值。没有修改 G2PW 设置，也没有引入进程池或其他运行时策略。

- 原模型 SHA-256：`2eb3c71fd95117b2e1abef8d2d0cd78aae894bbe7f0fac105ddc9c32ce63cbd0`；执行前再次校验。
- 输入来自已验证的 `20260919T125451.773385Z-g2pw-session`，固定为 `zh-1-trace-segment-2` 的保存数组，包含三个多音字查询。
- 原文本为 `.你好,欢迎使用樱花语音.现在正在测试苹果电脑上的语音合成.`；目标结果直接读取同轮官方默认去重输出。
- 使用 `.venv-mlx-macos`，NumPy 2.4.6、ORT 1.30.0，ORT build commit `f2c39fe`。CPU provider、顺序执行、intra-op 线程数 2、默认句子去重均不变。
- `ORT_DISABLE_TELEMETRY=1` 在 ORT 导入前设置。进程没有导入 Torch 或 Transformers。

每轮记录创建前、加载后、一次预测后、`close()` 加 GC 后的进程 RSS，并单独记录 OS 提供的进程生命周期最大 RSS。内存读取位于计时区间外。

## 实际结果

ORT 导入后的初始 RSS 为 51.75 MiB。

| 轮次 | Session 加载后 RSS | 预测后 RSS | 关闭并 GC 后 RSS | 相对上一轮关闭值 | 加载耗时 | 单次预测耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1316.77 MiB | 1318.95 MiB | 379.45 MiB | — | 229.51 ms | 13.42 ms |
| 2 | 1328.08 MiB | 1328.52 MiB | 380.14 MiB | +0.69 MiB | 203.73 ms | 9.23 ms |
| 3 | 1328.52 MiB | 1328.52 MiB | 380.14 MiB | 0 MiB | 207.94 ms | 9.43 ms |

最终 OS 生命周期最大 RSS 为 1328.53 MiB。表中 RSS 是执行边界快照，生命周期最大值覆盖整个进程；两者都不是逐阶段采样峰值，也不是 GPU 显存。

三轮输出均为 `ㄏㄠ3`、`ㄆㄧㄥ2`、`ㄏㄜ2`，三项置信度逐项与官方保存值严格相等。这里验证的是固定 G2PW 数组到多音字预测的重复执行结果，不是完整中文前端或音频质量。

关闭后的占用仍比初始值高约 328 MiB，但本次三轮内趋于稳定。此前包含长文本诊断的进程关闭后为 512.67 MiB，本轮仅执行一个短输入，因此两个关闭值不宜直接比较成优化收益。当前证据不足以区分 ORT 全局资源、分配器保留或其他存活对象，也不能凭三轮证明或排除长期泄漏。

本轮保存加载和预测原始时长，用于复现实验条件。OS 文件缓存可能已热，三次样本也不是完整请求基准；不把这些时长当作应用冷启动或速度提升。

## 证据与复现

证据目录：`SakuraTTS-References/runs/20260919T130735.830870Z-g2pw-lifecycle/`。

- `result.json`：固定输入来源和哈希、版本、三轮输出、计时、RSS 边界及生命周期最大值。
- `exit-observation.json`、`stdout.log`、`stderr.log`：外层记录的命令、实际退出码和原始日志。
- `g2pw_lifecycle.py`、`g2pw_session.py`：本次执行源码快照。

从项目目录执行以下命令会在参考目录中新建一轮记录：

```sh
/Users/beyondpower/Documents/Projects/SakuraTTS-References/.venv-mlx-macos/bin/python \
  harness/g2pw_lifecycle.py \
  --equivalence-run /Users/beyondpower/Documents/Projects/SakuraTTS-References/runs/20260919T125451.773385Z-g2pw-session
```

现阶段保留显式加载、关闭接口，将高于初始值的进程占用列为运行时集成时需要复测的边界，不为三轮观察增加进程管理层。完整 G2PW 文本链路接通后，仍需用真实请求序列检查驻留成本和释放策略。本轮没有新增音频、ASR 或人工试听证据。
