# Radeon 780M：声码器混合 FP16

N.A.V.I 的声学图通过了 DirectML 混合精度工程筛查：编码器与 flow 保持原始 FP32，声码器卷积使用 FP16，末端 Tanh 和公共输入输出保留 FP32。这里的 `fp16` 配置是混合精度，GPT 仍使用 FP32。四组保存输入和两组固定长短输入均通过预先设定的幅度、误差和频谱门槛；原始 FP32 等价容差仍有失败，人工听音和 ASR 尚未验收。

本机为 AMD Radeon 780M，DXGI 适配器索引为 `0`，PCI vendor/device 为 `1002:15bf`，驱动 `31.0.14005.11001`，ONNX Runtime DirectML `1.24.4`。适配器身份通过 DXGI 枚举读取，驱动来自匹配 PCI 标识的 CIM 记录。其他显卡和驱动没有在这轮实验中验证；运行时要求显式开启实验模式，并检查 provider 与已筛查的会话设置，不按显卡型号锁定模型包。

完整数值摘要、硬件记录、包哈希和原始文件位置保存在[实验数据](../experiments/data/2026-09-27-directml-acoustic-mixed.json)。通过的候选位于 `models/navi-cpu-amd/sovits-mixed-directml-vocoder-isolated`；独立模型配置是 `models/navi-cpu-amd/model-directml-mixed.json`。原 FP32 包及三个失败候选均保留。

## 输入与测量

四个原始 `validation-*.npz` 包含 1、2、7、19 个语义 token。另两组固定输入把 19-token 样本的语义和显式噪声平铺到 43、256 token，保留其音素、参考条件和 noise scale。这两组用于测量随长度增长的计算开销，不是自然语音质量样本。

每个 Session 先执行一次，再测三次热执行，表中取中位数。计时从 CPU 输入到 CPU 波形返回，不包含写盘；profile 在单独 Session 中采集。生产图、诊断图与 profile 图的波形一致，每次重复的所有已采集阶段一致。FP32 基线的四组波形满足原保存输出的 `atol=1e-4, rtol=1e-5`。

| 固定输入 | FP32 ms | 混合精度 ms | 耗时减少 |
| --- | ---: | ---: | ---: |
| 43 token / 1.72 s 波形 | 479.21 | 413.24 | 13.77% |
| 256 token / 10.24 s 波形 | 1513.63 | 1267.83 | 16.24% |

1、2、7、19-token 样本的混合精度耗时分别为 70.44、214.66、237.68、297.01 ms；2-token 和 19-token 样本比同轮 FP32 慢约 5.1% 和 8.3%。这组结果支持较长固定输入的声学加速，不能推导所有句长或完整请求都会加速。

最终权重文件从 248,441,344 字节变为 180,801,279 字节，减少约 27.2%。初始化张量包含 113,282,560 字节 FP32 和 67,639,488 字节 FP16。文件和张量容量不是进程或 GPU 峰值内存，本轮声学计时没有测量这两项峰值。

## 从失败候选到子图隔离

直接把整图可转换部分设为 FP16 时，256-token 输入从约 1638 ms 降到 1168 ms，但最大波形误差约 0.150，SNR 约 19.0 dB，工程筛查失败。低幅度样本还出现明显量化误差。

保留 Tanh、Sigmoid、Exp 为 FP32 后，1、2、7-token 样本通过；19、43、256-token 样本仍失败。长输入 SNR 约 19.2 dB。非线性算子精度改善了小信号输出，但没有消除编码器、MRTE 和 flow 中累积的误差。

第三个候选通过转换器的 `node_block_list` 排除上游节点。图中 Cast 数从 157 增至 2754，保存的中间张量仍出现精度损失；19-token 的 decoder input 最大误差达到约 0.150。该候选既未达到预期的 FP32 上游边界，也没有得到稳定的速度收益。

最终转换先按 `decoder_input → waveform` 的数据依赖提取声码器子图，再只转换该子图并合回原图。上游节点、权重和张量连接保持原样，声码器入口只在需要处转换为 FP16。小图测试逐字节确认上游节点和初始化权重没有改变；实际保存阶段输出和完整波形另由 DirectML 筛查验证。

## 工程门槛与准入

门槛沿用现有声学 FP16 工程实验，并在 GPU 执行前写入 `thresholds-before-run.json`：形状相同且全部有限，最大绝对误差不超过 0.05，RMSE 不超过 0.005，SNR 至少 25 dB，峰值不超过基线的 1.05 倍加 1e-4，频谱收敛误差不超过 5%，活跃频点 log 幅度 RMS 差不超过 1 dB。频谱使用 1024 点 Hann 窗、256 点 hop，活跃门槛为峰值以下 60 dB。

| 输入 token | 最大绝对误差 | RMSE | SNR dB | 活跃频点 log RMS 差 dB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0000268 | 0.00000705 | 29.77 | 0.468 |
| 2 | 0.0000378 | 0.00000935 | 34.67 | 0.379 |
| 7 | 0.000850 | 0.000133 | 40.24 | 0.134 |
| 19 | 0.004776 | 0.000431 | 42.14 | 0.105 |
| 43 | 0.003580 | 0.000368 | 43.18 | 0.087 |
| 256 | 0.005673 | 0.000470 | 41.19 | 0.094 |

profile 记录了 `DmlExecutionProvider` 上的 FP16 卷积，没有 CPU 神经网络计算事件；CPU 仍处理部分形状运算。输入语义和音素为 int64，参考条件、噪声和波形输出为 FP32。

候选使用独立的 `fp16-directml-engineering-screen` 准入记录，绑定生产图、诊断图、权重哈希、优化设置、真实 GPU 事件和会话设置。当前筛查设置为适配器 `0`、2 个 intra-op 线程、1 个 inter-op 线程、顺序执行、关闭 memory pattern、CPU arena 和线程自旋。CUDA 的既有筛查不能替代 DirectML 筛查，CPU 也不能沿用该结果。

## 复现

两个输出目录必须尚不存在：

```powershell
.venv-amd/Scripts/python.exe tools/convert_sovits_onnx_fp16.py --source models/navi-cpu-amd/sovits --output models/navi-cpu-amd/sovits-mixed-directml-rerun --fp16-scope vocoder
.venv-amd/Scripts/python.exe research/tools/directml_acoustic_precision.py --baseline models/navi-cpu-amd/sovits --candidate models/navi-cpu-amd/sovits-mixed-directml-rerun --output .cache/directml-mixed-rerun --repeats 3 --publish-screen
```

转换后的包先标记为未筛查。第二条命令只有在数值、重复性、公共 I/O 和 GPU 执行门槛全部通过后才写入可加载的筛查记录。公开 Engine 使用匹配的模型配置和 `backend="directml", profile="fp16"`；选择配置不会自动转换模型。低层 ORT 接口需要 `allow_experimental_fp16=True` 与上述会话设置。

DirectML 的[运行约束](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)和 ORT 的[混合精度转换说明](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)提供接口背景；本机的数值和速度结论来自上述保存实验。
