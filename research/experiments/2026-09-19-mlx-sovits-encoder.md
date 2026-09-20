# MLX 声学码本与编码器候选

日期：2026-09-19。范围是“朱雀院红叶”V2Pro 的 FP32、单请求、`speed=1` 路径，输出到 `mean/log_scale/mask`。Flow、声码器和完整语音生成尚未实现。

## 实现范围

`src/sakuratts/backends/mlx/encoder.py` 从已验证的 [SoVITS 解码包](2026-09-19-sovits-package.md) 中读取 235 个 `enc_p` 张量和一个量化码本，共 59,729,920 字节 FP32 数据。运行端只依赖 MLX、NumPy 和标准库，没有导入 PyTorch、官方工程或 Lite。

包内的原始键名、精度与布局保持不变。MLX Conv1d 接口使用 NTC 激活，卷积计算前将原 OIK 权重转为 OKI；对外输出和诊断阶段仍保存为官方 NCT 布局。码本查表后按官方 25 Hz 路径重复两倍时间轴，随后执行 SSL 编码、文本编码、MRTE、第二段编码器和均值/对数尺度投影。

自注意力保留官方计算顺序：先将 query 除以 `sqrt(head_features)`，计算普通与相对 key 分数，加上相对项后施加 `-1e4` 掩码，再 softmax，最后累加普通 value 和相对 value 输出。MRTE 保留交叉注意力结果、SSL 残差和调用方提供的 `ge512` 的相加顺序。通道 LayerNorm 的 epsilon 来自转换 manifest，推理时跳过 dropout。

这里只接受已经与同一 SoVITS checkpoint 绑定的 `[1,512,1]` FP32 `ge512`。准备参考条件仍由原模型负责。CPU/GPU 是同一 MLX 算子图的验证设备选项，没有新增多版本后端框架。

## CPU 算子检查

`research/tools/mlx_sovits_encoder_replay.py --self-test` 完成 17 项检查，证据目录为 `runs/20260919T114713.192724Z-mlx-sovits-encoder-self-test-cpu/`，相对于参考目录。

- 长度 1、3、8 的相对/绝对位置转换与相对 embedding 裁剪、补零，逐元素精确一致。NumPy 对照直接按 `j-i` 索引，不复用实现中的 pad/reshape。
- 带相对 key/value 的自注意力和普通交叉注意力，对照独立 NumPy FP64 点积与加权求和；包含被遮蔽的 key 和全遮蔽 query 行。
- kernel 为 1、3、4 的 Conv1d，使用独立循环检查原权重轴顺序和奇偶 padding。
- 通道 LayerNorm，使用独立 FP64 均值、方差、缩放与偏置计算。

全部通过。精确索引检查的误差为 0；浮点算子最大绝对误差约 `1.515e-7`，低于小算例预设的 `atol=2e-6, rtol=2e-5`。检查进程没有导入 PyTorch。这些检查用于验证算子语义，不能替代完整权重下的逐阶段对照。

## 固定模型对照方法

Harness 读取已经保存的官方固定声学 NPZ，核对 checkpoint、源码提交和数组哈希，使用相同 codes、phones 和 `ge512`，比较 `quantized`、`ssl_encoded`、`text_encoded`、`mrte`、`encoder_hidden`、`mean`、`log_scale`、`mask`。完整模型预设容差保持 `atol=1e-4, rtol=1e-5`，不会用小算例的容差或 top-1 指标代替它。

每次运行保存官方/MLX 阶段数组、逐项误差、执行源码快照和命令。诊断时间包含求值和 CPU 数组复制；MLX allocator 的 active、cache、peak 单独记录，不能作为正常声学生成延迟、进程 RSS 或 NVIDIA 显存。

## 完整编码器的 GPU 结果

固定两例的八个阶段全部通过原设容差，没有超差元素。码本输出和 mask 精确一致，其他阶段的最大绝对误差如下：

| 阶段 | 日文 | 中文 |
|---|---:|---:|
| SSL 编码器 | 5.7220e-6 | 6.5230e-6 |
| 文本编码器 | 1.1027e-6 | 8.9407e-7 |
| MRTE | 9.7275e-5 | 8.1778e-5 |
| 第二段编码器 | 2.0266e-6 | 1.9073e-6 |
| `mean` | 6.1989e-6 | 7.8678e-6 |
| `log_scale` | 8.3447e-7 | 8.3447e-7 |

证据为 `runs/20260919T115443.126195Z-mlx-sovits-encoder-gpu/`，对照官方固定条件 `runs/20260919T111545.846735Z-sovits-fixed-official-mps/`。运行进程未导入 PyTorch 或上游实现，退出码为 0；阶段数组、输入和源码快照哈希均已核验。这些结果只覆盖当前两条样例，长句、混合语言及其他参考条件仍需扩展。

加载 236 个张量后，MLX active 为 59,741,184 字节；本次带阶段保存的诊断运行 allocator peak 为 163,017,104 字节。释放模型、阶段引用并清理 cache 后，active/cache 均为 0。这些是单个编码器的统一内存计数，没有与同条件官方正常运行比较，不能据此宣布完整 TTS 节省了多少内存或时间。

首轮运行 `runs/20260919T115322.740044Z-mlx-sovits-encoder-gpu/` 在建立图时因 `KeyError: 'enc_p.mrte.cross_attention'` 失败。原转换器只保存有直属张量的模块信息，遗漏了没有相对 embedding 的 MRTE cross-attention 的头数配置。补齐该模块描述后，新包为 `models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32/`；新旧 `weights.npz` 的 SHA-256 完全相同，650 个张量的来源描述也相同，只增加所需元数据。旧包和失败记录均未修改。

本轮没有生成新音频，也没有 ASR、人工试听或正常性能对照。后续继续实现 reverse flow 和声码器，并保留逐阶段检查。

复现 CPU 小算例：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" research/tools/mlx_sovits_encoder_replay.py \
  --references "$REF" --self-test
"$REF/.venv-mlx-macos/bin/python" research/tools/mlx_sovits_encoder_replay.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T111545.846735Z-sovits-fixed-official-mps" \
  --device gpu
```

实现依据为固定官方提交 `48b1a0169a28582a8984402f82cf438d3bfa6aca` 的 `TextEncoder`、`Encoder`、`MultiHeadAttention` 和 `MRTE`。源码文件保留 MIT 出处，并链接项目中的官方许可副本。
