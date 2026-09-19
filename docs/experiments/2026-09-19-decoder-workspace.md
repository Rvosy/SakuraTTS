# 声码器计算边界与临时工作区

日期：2026-09-19。自有声学计算的活动峰值主要出现在声码器，限制空闲缓存没有明显减少这个峰值。本轮只调整 MLX 惰性图的执行边界，不改变卷积、权重、激活、残差顺序或输入长度。

## 对比三种执行位置

原实现每完成一个上采样阶段及其三个残差块，才调用 `mx.eval(x)`。每个残差块包含三个卷积对。未求值的图会同时保留更多依赖，部分卷积临时结果无法在下一对开始前释放。

`harness/decoder_workspace.py` 比较三种位置，每种使用单独进程：

- `stage`：保留原上采样阶段边界。
- `resblock`：额外在每个完整残差块后求值。
- `pair`：额外在每个卷积对及其残差相加后求值。

三种路径的算式、权重布局和求和顺序相同。测试使用相同官方 `decoder_input` 与 `ge`，不重跑编码器或 flow。调度候选先保存在 Harness 中，完成十例逐位核验后才把 `pair` 方式移入自有声码器。

## 两条原始样例的正常测量

每条输入先核对波形，再热身两次、测量五次。计时包括 CPU 输入转为 MLX、完整声码器执行及输出求值，不包括诊断捕获、CPU 输出复制或波形比较。分配器 peak 在热身后重置，只覆盖该例计时请求；cache 是请求后边界值。它们不是 NVIDIA 显存或完整 TTS 的峰值。

| 调度 | 日文中位耗时 | 中文中位耗时 | 日文 allocator peak | 中文 allocator peak |
|---|---:|---:|---:|---:|
| stage | 0.18813 秒 | 0.22705 秒 | 790.52 MiB | 943.18 MiB |
| resblock | 0.19600 秒 | 0.23524 秒 | 519.49 MiB | 617.72 MiB |
| pair | 0.21454 秒 | 0.24974 秒 | 432.86 MiB | 512.78 MiB |

三个版本的两条波形逐位相同，并通过原官方 `atol=1e-4, rtol=1e-5`。`pair` 的分配器峰值降低约 45%，声码器耗时增加约 10%–14%。`resblock` 的峰值少约 34%，耗时增加约 4%。原权重 active 约 56.98 MiB 未变；更早求值减少的是临时活动数组，并不是模型被删减。

中文请求后 cache 分别为 2456.72、1927.51、1579.09 MiB，仍然明显高于模型权重。更小的活动峰值不会自动保证低空闲缓存，缓存策略需要独立测量。三个进程释放后 active/cache 均为零。

## 十例完整波形回归

随后对十条固定条件做原 stage 与 pair 的独立进程比较，覆盖原中日文输入、短句、长句、标点和混合文本。十条完整浮点波形全部逐位相同，且都通过官方原容差。这组仅各运行一次用于扩展回归，不据此给出长句速度结论。

日文长句 peak 从 1556.19 降为 1210.31 MiB，中文长句从 1801.86 降为 1477.59 MiB，降幅约 22% 和 18%。不同长度的工作区和调度成本不同，短句的约 45% 降幅不能推广到所有输入；约 0.5 GiB 的短句声码器峰值也不是整个引擎的显存承诺。

本轮只验证声码器计算调度。编码器既有的日文标点 MRTE 超差不受影响，也没有采纳会造成长句波形退化的 FP64 LayerNorm 候选。声码器使用的是固定官方 latent；完整生成链路的回归与成本另行记录。

## 移入运行时后的整链结果

生产候选 `MLXSoVITSDecoder.resblock` 只增加每个卷积对后的 `mx.eval(x)`。Harness 的 `--policy runtime` 直接调用这个方法；其他策略继续明确覆盖残差块，因此仍能复现旧 stage 基线。迁移后的十例输出与独立 pair 候选再次逐位一致，证据为 `20260919T130228.689264Z-decoder-workspace-runtime`。

再运行两条原始报告样例的自有 GPT → 声学 → PCM，每例首次运行、2 次热身、5 次计时，默认常驻与按请求释放两种策略分别和修改前的同配置比较：

| 配置 | 调度 | 日文 / 中文完整准备条件请求 | MLX allocator peak | 中文结束 active / cache |
|---|---|---:|---:|---:|
| GPT 常驻，默认缓存 | stage | 1.053 / 1.288 秒 | 1452.41 MiB | 565.05 / 2674.53 MiB |
| GPT 常驻，默认缓存 | pair | 1.068 / 1.270 秒 | 1021.96 MiB | 565.05 / 1793.21 MiB |
| 按请求释放 GPT，cache limit 256 MiB | stage | 1.344 / 1.512 秒 | 1052.45 MiB | 165.15 / 257.16 MiB |
| 按请求释放 GPT，cache limit 256 MiB | pair | 1.314 / 1.517 秒 | 622.05 MiB | 165.15 / 162.23 MiB |

这张表的 peak 覆盖加载及全部请求，口径不同于上面的 decoder-only 热请求峰值。两种生命周期分别降低约 430.4 MiB，完整请求时间接近；不能据单次顺序实验的小差异宣称端到端加速。常驻 pair 的 RSS 生命周期最大值为 747.83 MiB，按请求 pair 为 663.16 MiB，二者都另含 CPU 数据和运行环境。

每次 token、停止、语义和波形检查通过。与修改前同配置的 token、history、semantic、完整浮点波形及 native/official-fixed WAV 全部逐位相同，共 24 项；模型和输入来源也核对一致。它们是用户已经确认的同一份音频字节，本轮无需把新的听感推断为通过。释放模型和缓存后，两进程的 active/cache 均为零。

上述数字来自同一 V2Pro 的两条短请求，文本特征和参考条件已经准备好；没有加载完整中文、日文前端。622 MiB 不能当作完整 TTS、所有长度或 NVIDIA 显存的指标。此改动保留在自有声码器中，按请求释放 GPT 仍为可选实验策略。

## 证据与复现

以下目录位于 `/Users/beyondpower/Documents/Projects/SakuraTTS-References/runs/`：

- 两例 stage：`20260919T125103.953179Z-decoder-workspace-stage`。
- 两例 resblock：`20260919T125128.362354Z-decoder-workspace-resblock`。
- 两例 pair：`20260919T125148.045676Z-decoder-workspace-pair`。
- 十例 stage：`20260919T125457.817693Z-decoder-workspace-stage`。
- 十例 pair：`20260919T125528.508824Z-decoder-workspace-pair`。
- 迁移后的十例 runtime：`20260919T130228.689264Z-decoder-workspace-runtime`。
- 迁移后的常驻整链：`20260919T130321.665569Z-native-prepared-speech`。
- 迁移后的按请求整链：`20260919T130443.682641Z-native-prepared-speech`；同目录的 `compare-decoder-integration.py` 和 `decoder-integration-equivalence.json` 保存完整逐位复核。

每个运行保存源码、模型包和输入哈希、完整浮点波形、逐次耗时及内存计数。候选通过 `--equivalence-reference` 检查基线的模型、条件和波形哈希，再要求逐位相同；不一致时不会进入计时。

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/decoder_workspace.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T111545.846735Z-sovits-fixed-official-mps" \
  --policy stage
```

使用相同参数改为 `--policy resblock` 或 `pair`，再把 `--equivalence-reference` 指向刚生成的 stage 目录。十例回归使用 `20260919T122232.246358Z-sovits-fixed-official-mps` 条件目录、`--warmup 0 --repeat 1` 和语料中的十个 `--cases` ID，完整命令已保存在对应结果文件。
