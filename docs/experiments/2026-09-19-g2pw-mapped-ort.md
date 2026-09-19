# G2PW 使用内存映射的 ORT 模型

日期：2026-09-19。在固定 Mac CPU / ORT 1.30.0 环境，将原 ONNX 的图优化离线保存为 ORT 文件，再让初始化权重引用映射文件。自有运行时已经接入这个显式选项，原 ONNX 入口保留。完整拼音接口的 27 组结果及 353 个数组与官方逐位相同。

## 实现与适用范围

`scripts/prepare_g2pw_ort.py` 在新目录生成文件，不改原模型。`G2PWSession.from_ort_package()` 核对 ORT 版本、build、系统、架构、CPU provider 和模型 SHA-256，再启用 `session.use_memory_mapped_ort_model` 与 `session.use_ort_model_bytes_for_initializers`。重载使用 `ORT_DISABLE_ALL`，因为图优化已保存。Session 存活时不能改写映射文件。

G2PW 上层可显式传入 `ort_package`；它与 `model_path` 二选一。文本、去重、预测和拼音规则不变。ORT 包的原始转换清单仍标为 `converted_unvalidated`，验证证据另存，没有追写历史清单。优化图可能依赖硬件与运行库，Windows 需要重新准备和对照，不能直接承诺复用这个 Mac 二进制文件。

原模型为 635,212,732 字节，ORT 文件为 635,362,480 字节，多 149,748 字节。本轮没有磁盘压缩收益。离线转换约 0.889 秒，转换进程最高 RSS 约 2.44 GiB；该成本属于准备工具，不隐藏到运行测量之外。

## 同条件实测

迁入实际运行接口后的独立进程对照位于 `runs/20260919T134312.294827Z-g2pw-prepacking/`。两端固定相同 FP32 模型、输入与 ORT CPU 参数，17 次实际模型概率输出、标签和置信度逐位一致，进程退出码均为 0。

| 指标 | 原 ONNX | 映射 ORT |
|---|---:|---:|
| 加载后 RSS 边界 | 1318.64 MiB | 718.94 MiB |
| 请求后 RSS 边界 | 1357.00 MiB | 850.97 MiB |
| 关闭并 GC 后 RSS 边界 | 515.06 MiB | 442.64 MiB |
| OS 进程最高 RSS | 1357.02 MiB | 850.97 MiB |
| 构造耗时 | 0.258 秒 | 0.309 秒 |

进程最高 RSS 少约 506 MiB；这些都是 CPU / Apple 统一内存数据，不是 NVIDIA 显存。构造计时中，映射入口包含自己的完整文件校验，原入口不包含 Harness 预检，不能把这组时间当作同等校验的启动加速。

五组输入各预热两次、正常计时五次。原入口中位数为 9.842、25.964、12.519、7.349、7.464 ms；映射入口为 9.168、25.653、12.607、7.267、7.524 ms。计时不安装概率捕获，不含输出对照和写盘，尚未证明稳定的推理加速。

更早的三候选对照 `134025.316688Z-g2pw-prepacking` 说明，单纯保存 ORT 图并不能获得该内存收益：未启用映射的请求后 RSS 反而为 1725.95 MiB，未采用该路径。关闭预打包或关闭 arena 的先前实验也未采用，见 [分配策略实验](2026-09-19-g2pw-allocation.md)。

## 拼音与释放回归

`134355.174387Z-g2pw-pinyin` 通过实际 `G2PW(..., ort_package=...)` 执行全部 27 组样例，与独立官方进程比较，353 数组逐位一致。保留原有超长输入和域外异常行为，没有借此改变前端语义。

`134835.246749Z-g2pw-lifecycle` 在单进程依次创建、预测和关闭三个 Session；每轮标签、置信度完全相同，进程退出码为 0。三次关闭后的 RSS 为 362.39、363.13、363.14 MiB；最后一次预测后为 721.84 MiB。这里每轮只处理同一短输入，因此不能与上表全部输入后的数值直接比较，也不能据三轮观察排除长期泄漏。

没有生成新音频、运行 ASR 或新增人工试听。此处验证的是同源 G2PW 的输出和内存策略，不是完整 TTS 峰值。

## 证据与复现

路径均位于 `SakuraTTS-References`：

- `models/converted/20260919T133901.909143Z-g2pw-ort-cpu`：原模型身份、转换器快照、运行库 build、文件哈希、原始告警和转换进程记录。
- `runs/20260919T132635.973960Z-ort-prepacking-source`：当前 ORT build 的配置键源码。
- 上述四个运行目录：实际命令、源码快照、概率文件、正常计时、RSS 和退出码；完整拼音以 `comparison.json` 为准。

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" scripts/prepare_g2pw_ort.py \
  --source <原始g2pW.onnx> --output-root "$REF/models/converted"
"$REF/.venv-mlx-macos/bin/python" harness/g2pw_prepacking.py \
  --equivalence-run "$REF/runs/20260919T125451.773385Z-g2pw-session" \
  --policies default runtime-mmap --optimized-package <新ORT包>
"$REF/.venv-mlx-macos/bin/python" harness/g2pw_lifecycle.py \
  --equivalence-run "$REF/runs/20260919T125451.773385Z-g2pw-session" \
  --ort-package <新ORT包>
```

每次复现应新建运行目录。完整命令以对应原始 JSON 为准。
