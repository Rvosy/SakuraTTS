# MLX 完整声学链路的累积数值误差

日期：2026-09-19。本文记录独立声学模块连接后的一次失败、误差定位和一个通过固定样例的执行候选。所有比较使用“朱雀院红叶”V2Pro、同一组官方 codes/phones/ge/ge512/noise，保留 `atol=1e-4, rtol=1e-5`。这里没有 ASR、人工试听或正常性能验收。

## 组合失败及坐标

编码器、reverse flow、Generator 各自使用官方固定输入时已通过局部检查。但在 `runs/20260919T120834.391918Z-mlx-sovits-complete-gpu/` 中连接三个 MLX GPU 模块后，日文波形通过，中文 186,880 个采样点中有 1 个超差。原 run 保持 `numerical_mismatch`，没有修改阈值或丢弃该采样点。

中文超差位置是从 0 开始的第 140032 个采样点，即 32 kHz 下的 4.376 秒：

| 项目 | 数值 |
|---|---:|
| 官方波形 | -0.09815556556 |
| 完整 MLX GPU 波形 | -0.09804629534 |
| 误差 | 1.09270215e-4 |
| 该点允许误差 | 1.00981556e-4 |
| 官方 latent 输入 MLX decoder 后的误差 | 2.90796161e-5 |
| 改为 MLX latent 后追加的波形差异 | 8.01905990e-5 |

这两部分在该点同向相加。完整中文 latent 与官方的最大差异为 `1.33514e-5`、RMS 为 `1.11318e-6`，虽然 latent 本身通过阶段容差，仍会在波形解码中被局部放大。邻近第 140031 个采样点的误差也接近上限。这些数据不能单独说明差异是否可闻。

## 高精度权重归一化没有解决失败

`harness/mlx_sovits_numerical_diagnosis.py` 在独立进程里，用 NumPy FP64 对原始 `g/v` 归一化，再一次转换成 FP32。分别替换 decoder、flow 和两者的归一化结果，计算过程仍为 MLX FP32。诊断保留原始 `g/v`，不会改写转换包或默认运行模块，也没有按输入或采样点作特殊处理。

| 归一化路径 | 日文波形最大绝对误差 | 中文波形最大绝对误差 | 中文超差数量 |
|---|---:|---:|---:|
| 原 GPU FP32 | 7.17342e-5 | 1.09270e-4 | 1 |
| 仅 decoder 使用 FP64 归一化结果 | 6.65784e-5 | 1.08093e-4 | 1 |
| 仅 flow 使用 FP64 归一化结果 | 6.67721e-5 | 1.07370e-4 | 1 |
| flow、decoder 都使用 FP64 归一化结果 | 6.91712e-5 | 1.09248e-4 | 1 |

四条路径都比较了完整波形。这个实验没有解决中文超差，因此没有采用为默认实现。不能根据该结果把失败归因于 weight normalization。

证据为 `runs/20260919T121235.071244Z-mlx-sovits-numerical-diagnosis/`。其中 baseline 对原组合波形逐位复现；其余路径保存了各归一化权重的差异、官方/原 MLX latent 下的波形和重新连接后的完整波形。

## 从阶段边界替换输入

下一轮保持运行模块不变，只把某一阶段输入替换为已经保存的官方数组，再沿原图继续计算。这些替换只用于定位，不能作为生产运行时的答案缓存。

| 使用官方结果的边界 | 日文最终波形最大绝对误差 | 中文最终波形最大绝对误差 | 两例波形是否通过 |
|---|---:|---:|---|
| decoder 输入 | 5.05298e-5 | 4.78234e-5 | 是 |
| flow 输入 | 4.14401e-5 | 5.04870e-5 | 是 |
| mean 和 log_scale | 4.14401e-5 | 5.04870e-5 | 是 |
| 仅 mean | 3.39895e-5 | 4.91571e-5 | 是 |
| 仅 log_scale | 6.64443e-5 | 1.11222e-4 | 否，中文 2 点 |
| encoder_hidden | 4.14401e-5 | 5.04870e-5 | 是 |
| MRTE 输出 | 4.55379e-5 | 5.41322e-5 | 是 |
| SSL、text 编码器输出 | 7.16448e-5 | 6.67460e-5 | 是 |
| 仅 SSL 编码器输出 | 1.02460e-4 | 4.06411e-5 | 否，日文 1 点 |
| 仅 text 编码器输出 | 4.47482e-5 | 9.76697e-5 | 是 |

官方 `encoder_hidden` 经过 MLX 投影后，两例 `mean/log_scale` 都与官方逐位相同，接着得到的 `flow_input` 也逐位相同。因此在这两例中，末尾投影和显式噪声构造没有增加差异。误差已经在更早的声学编码器计算中累积。

单独替换 SSL 或 text 会改变不同位置误差的抵消情况，不能据此只修一个分支。这里只把调查范围缩小到编码器的 FP32 执行路径，没有确认某一个注意力、归一化或卷积算子存在公式错误。

证据为 `runs/20260919T121451.955691Z-mlx-sovits-numerical-diagnosis/`，复用原 MLX flow_input 的控制项再次逐位复现完整组合失败。

## 显式 CPU 编码器候选

第三轮在 `with mx.stream(mx.cpu)` 内执行同一个 MLX FP32 编码器，随后在 `with mx.stream(mx.gpu)` 内执行 latent 构造、flow 和 decoder。权重、公式、输入、噪声和容差全部保持不变；没有调用 PyTorch，也没有用默认设备状态来推测执行位置。

在 `runs/20260919T121612.404229Z-mlx-sovits-numerical-diagnosis/` 中，两例全部 12 个阶段通过，完整波形没有超差元素：

| 样例 | 波形最大绝对误差 | RMS 误差 | 编码器诊断耗时 | 完整声学诊断耗时 |
|---|---:|---:|---:|---:|
| 日文 | 7.40141e-5 | 1.74400e-6 | 0.0232 秒 | 0.247 秒 |
| 中文 | 4.92409e-5 | 1.68282e-6 | 0.0294 秒 | 0.290 秒 |

日文的误差比原全 GPU 组合略大，仍在原定容差内；中文超差消失。这是通用、明确选择的执行路径，不随句子或坐标切换。它目前只是通过两条固定样例的候选，需要再用正常计时和更多条件验证成本与覆盖范围，不能据此称 CPU 算法在所有输入上更接近官方。

上述时间来自带阶段保留的单次诊断，不代表热运行延迟或端到端 TTS 速度。三个诊断 run 均未导入 PyTorch/上游，退出码 0，源码快照和数组 SHA-256 已核验，最后释放引用并清理 cache 后 MLX active/cache 都为 0。诊断 run 的 `completed` 表示矩阵实验执行完整，各变体是否通过仍以对应的逐项比较为准。

首次启动时命令中的包路径误写为 `...-f8bd92161X-sovits-decode-fp32`，在模型加载前触发 `FileNotFoundError`。该目录和退出后补记的 `startup-error.json` 保留于 `runs/20260919T121221.541213Z-mlx-sovits-numerical-diagnosis/`。后续成功运行使用下面列出的正确包路径。

复现时使用 `--stage-attribution` 运行阶段替换矩阵，使用 `--cpu-encoder` 运行显式 CPU 编码器候选；不加这两个选项则运行归一化矩阵：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/mlx_sovits_numerical_diagnosis.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T111545.846735Z-sovits-fixed-official-mps" \
  --baseline-complete "$REF/runs/20260919T120834.391918Z-mlx-sovits-complete-gpu" \
  --cpu-encoder
```
