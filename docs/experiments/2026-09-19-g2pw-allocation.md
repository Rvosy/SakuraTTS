# G2PW 的 ORT 分配选项对照

日期：2026-09-19。独立 G2PW Session 加载后约占 1.3 GiB RSS，关闭后仍有较高占用。本轮分别调查权重预打包和 CPU arena，两种候选都没有带来足以采用的整体资源收益，生产 Session 保持原设置。

## 相同条件

沿用 `20260919T125451.773385Z-g2pw-session` 的原 ONNX 模型、八组保存输入、完整标签与官方输出。ORT 1.30.0、CPU provider、顺序执行、intra-op 线程数 2、图优化 `ORT_ENABLE_ALL` 均不变，模型再次流式校验 SHA-256。默认句子去重和原始逐查询两种模式均执行。

`harness/g2pw_prepacking.py` 为每个策略创建独立进程，先保存概率与正确性比较，再关闭捕获，对五组实际规范化中文输入各预热两次、计时五次。计时区间只有实际 `predict`，包含去重、模型执行及标签 / 置信度读取；文件复制、哈希、数组比较和 RSS 查询在计时外。两个候选都只改 ORT 一个选项，自有文本或模型能力未削减。

RSS 是边界快照；OS 最大 RSS 覆盖导入、加载、诊断和正常请求的整个进程。这里没有 GPU 占用、完整前端延迟或语音质量结论。

## 关闭权重预打包：没有占用收益，耗时增加

候选只设 `session.disable_prepacking=1`。该开关语义来自与本机 ORT build 相同提交 `f2c39fe` 的 `onnxruntime_session_options_config_keys.h`，源文件与哈希已保存。

| 边界 | 默认 | 关闭预打包 |
|---|---:|---:|
| Session 加载后 RSS | 1316.83 MiB | 1316.56 MiB |
| 全部正常请求后 RSS | 1355.17 MiB | 1372.39 MiB |
| 关闭并 GC 后 RSS | 513.23 MiB | 903.25 MiB |

五组正常预测中位耗时分别由 `8.70 / 24.54 / 12.60 / 7.13 / 7.22 ms` 变为 `15.47 / 33.25 / 21.41 / 20.83 / 23.44 ms`。当前证据不支持“预打包保留了一份可以直接去掉的权重副本”这一收益假设，关闭后占用也没有改善。

全部标签相同，部分长边界输入的置信度不同；概率最大绝对差为 `2.68221e-7`，全部仍在既有 `atol=1e-6, rtol=1e-5` 内，但不再逐位一致。Harness 按本轮的严格保持条件返回 `mismatch`、退出码 1，记录中同时保留容差通过的事实。这不是模型报错。没有资源或速度收益，因此不采用该候选，也没有为了使状态变绿而改判定规则。

## 关闭 CPU arena：关闭值降低，运行峰值增加

第二轮保留预打包，仅设 `enable_cpu_mem_arena=False`，默认策略也在新进程中同条件重测。

| 边界 | 默认 | 关闭 CPU arena |
|---|---:|---:|
| Session 加载后 RSS | 1317.25 MiB | 1318.30 MiB |
| 全部正常请求后 RSS | 1355.91 MiB | 1439.48 MiB |
| 关闭并 GC 后 RSS | 513.97 MiB | 420.48 MiB |
| OS 进程生命周期最大 RSS | 1355.92 MiB | 1439.50 MiB |

五组正常预测中位耗时由 `8.63 / 25.26 / 12.26 / 7.07 / 8.55 ms` 变为 `9.63 / 25.51 / 12.98 / 7.31 / 7.26 ms`。这是一次顺序对照，没有足够证据把最后一组的下降当作稳定加速。

两侧全部概率逐位相同，标签、置信度和重复预测也完全保持，进程均退出 0。但候选运行后 RSS 和进程最大 RSS 增加约 84 MiB，只有关闭后的边界少约 93 MiB。当前优先级同时包含生成峰值和常驻，因而保留默认 arena；不把卸载一项改善描述成整体内存优化。上述数据也不足以把剩余 RSS 归因于某一类内部缓存或对象。

## 证据和复现

以下目录均在 `SakuraTTS-References/runs/`：

- `20260919T132635.973960Z-ort-prepacking-source`：ORT 配置开关的固定来源、原文件及哈希。
- `20260919T133120.035347Z-g2pw-prepacking`：默认与关闭预打包，含正常五次时长、原概率、内存、真实退出码和源码快照。
- `20260919T133230.207949Z-g2pw-prepacking`：默认与关闭 arena 的同条件记录。

每轮 `evidence-verification.json` 已核对源码快照和全部输出数组哈希。自有进程均未导入 Torch。

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/g2pw_prepacking.py \
  --equivalence-run "$REF/runs/20260919T125451.773385Z-g2pw-session" \
  --policies default no-prepack
"$REF/.venv-mlx-macos/bin/python" harness/g2pw_prepacking.py \
  --equivalence-run "$REF/runs/20260919T125451.773385Z-g2pw-session" \
  --policies default no-arena
```

运行会新建目录，不覆盖上述证据。下一项调查是将 ORT 图优化放到离线准备阶段，实测运行时是否因此减少加载成本或临时占用；不预设该转换一定有效，原模型保持可用。
