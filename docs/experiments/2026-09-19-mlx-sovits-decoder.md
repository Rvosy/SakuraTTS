# MLX SoVITS 波形解码器

日期：2026-09-19。范围是“朱雀院红叶”V2Pro、FP32、单请求、`speed=1` 下的 `Generator`，输入为已经施加 mask 的 reverse-flow 输出和同一模型准备的 `ge`。这一模块独立读取转换包中的 `dec.*` 权重，不导入 PyTorch、官方实现或 Lite。

## 实现与边界

`src/sakuratts/mlx_sovits_decoder.py` 保留固定官方提交 `48b1a0169a28582a8984402f82cf438d3bfa6aca` 的 Generator/ResBlock1 顺序。普通卷积权重从 OIK 转为 MLX 的 OKI，转置卷积从 IOK 转为 OKI。weight normalization 在原始布局上按 manifest 的 `dim` 计算，保留原始 FP32 `g/v`，没有折叠或量化权重。

每一级先执行斜率 0.1 的 leaky ReLU 和转置卷积，再让三个不同 kernel 的残差分支读取同一个输入，按官方顺序求和、取平均。每个分支保留三对膨胀卷积和普通卷积。最后一处 leaky ReLU 使用官方未显式指定参数时的默认斜率 0.01，然后普通卷积和 tanh。这个末尾斜率与前面各处不同。

对外输入和输出使用 NCT。`MLXSoVITSDecoder.load(package)` 加载后，`decode(latent, ge)` 可直接接收 FP32 MLX 数组，返回 MLX 波形，不经 CPU 转换；`capture=True` 额外返回每级中间结果。每个完整上采样阶段结束时求值，限制懒执行图的跨度。这里还没有声码器分块或临时工作区复用优化。

当前只接受转换包声明的 V2Pro、三对卷积的 ResBlock1 和 groups=1。新的模型结构需要先补充验证，不能根据当前模型的通过情况宣布支持整个官方模型家族。该模块不负责参考条件准备、语义生成、响度处理或静音拼接。

## 独立 CPU 小算例

`harness/mlx_sovits_decoder_replay.py --self-test` 完成 16 项检查。NumPy FP64 对照使用输入/输出位置循环，转置卷积逐位置散加；没有调用 PyTorch，也没有复用 MLX 的卷积实现。

- weight normalization 检查三个维度，覆盖原始权重布局上的归约。
- 普通卷积覆盖 stride、padding、dilation；转置卷积覆盖 stride、padding、dilation、output padding。
- 小型两级 Generator 使用两种 kernel 和每个分支三对残差卷积，逐级检查条件相加、上采样、残差平均及最后波形。专门保留了负值输入，末尾错误地使用 0.1 会令最大波形差异达到 0.004883，因此该样例能检出默认斜率被误改的问题。

16 项全部通过预设的小算例容差 `atol=2e-6, rtol=2e-5`；最大的绝对误差约 `9.84e-8`。证据目录为参考目录下的 `runs/20260919T120759.135342Z-mlx-sovits-decoder-self-test-cpu/`。

## 完整模型 GPU 对照

运行读取 `runs/20260919T111545.846735Z-sovits-fixed-official-mps/` 中保存的官方 `decoder_input`、`ge` 和完整 `waveform`。先核对 checkpoint、官方源码提交及数组 SHA-256，再用 MLX 生成波形。输入没有改写、重新采样或裁剪。使用的转换包为 `models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32/`。

完整模型保持原定容差 `atol=1e-4, rtol=1e-5`。两例波形全部有限、长度相同，没有超差元素：

| 样例 | 波形样本数 | 时长 | 最大绝对误差 | RMS 误差 | 相对 L2 |
|---|---:|---:|---:|---:|---:|
| 日文 | 153,600 | 4.80 秒 | 5.0530e-5 | 1.3049e-6 | 8.5892e-6 |
| 中文 | 186,880 | 5.84 秒 | 4.7823e-5 | 1.4175e-6 | 9.3758e-6 |

证据目录为 `runs/20260919T120809.612645Z-mlx-sovits-decoder-gpu/`，进程退出码 0。目录内保存官方和 MLX 波形数组、MLX 每一级中间结果、输入、误差、执行命令及源码快照。两个 run 的源码与数组哈希均已复核，运行进程没有导入 PyTorch 或上游模块。

这些结果验证的是固定声学条件下的数值对齐。没有在本轮进行 ASR 或人工试听，也没有据此重新判定漏句、助词或音色问题已经通过验收。完整编码器、flow 和 decoder 连接后的误差传播仍由单独的声学链路实验验证。

## 资源与计时口径

该模块加载 290 个张量，原始 FP32 数据为 59,697,152 字节。加载后 MLX active 为 59,746,304 字节。保存各级中间结果的两例诊断运行中，MLX allocator peak 为 992,616,604 字节，最后一例结束时 cache 为 2,571,968,540 字节。释放模型、波形、中间数组引用并清理 cache 后，active/cache 均为 0。

诊断运行的日文/中文耗时约 0.397/0.294 秒，包含阶段求值与 CPU 拷贝，不能直接作为正常请求延迟。上述 peak 是 MLX allocator 记录，cache 另列；它们都不是进程 RSS 或 NVIDIA 独立显存。大 cache 表明正常性能实验需要明确记录缓存策略，并用同条件运行判断清理缓存的速度代价。

本轮没有与同条件官方正常运行进行内存或速度比较，没有安装体积收益结论。原模型和历史运行目录保持原样。

复现命令：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/mlx_sovits_decoder_replay.py \
  --references "$REF" --self-test
"$REF/.venv-mlx-macos/bin/python" harness/mlx_sovits_decoder_replay.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T111545.846735Z-sovits-fixed-official-mps" \
  --device gpu
```
