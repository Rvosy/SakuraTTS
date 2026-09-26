# CPU 与 DirectML 声学 FP16 实验准入

本轮允许将超出原有误差门槛的声学候选用于试听。默认 `screened` 准入保持原样；实验必须同时设置 `allow_experimental_acoustic_fp16=True` 和 `acoustic_fp16_acceptance="finite"`。`finite` 只放宽数值误差，不放宽文件哈希、公开 FP32 输入输出、形状、有限值、重复运行和算子执行位置检查。

准入记录按设备保存在 `manifest.experimental_validations.cpu` 与 `.directml`。两条路径共用权重，但互不借用验证结果。原 `manifest.validation.passed=false` 不改写。记录中的 `engineering_screen.passed` 与 `quality_accepted` 也不因允许试听而改为通过。加载边界见 `sakuratts.backends.onnx.precision_experiment.validate_experiment`；记录发布工具是 `research/tools/directml_acoustic_precision.py`。

## 实测事实

候选为 `models/navi-cpu-amd/sovits-fp16-directml-candidate`，模型入口为同目录上一层的 `model-fp16-experimental.json`。它将全声学网络中可转换的权重和算子改为 FP16；公开输入输出、LayerNorm 累加及转换器列明的阻止算子仍保留 FP32。这里的“全 FP16”不表示每个浮点张量都为半精度，也不包含 GPT。

DirectML 使用此前在 Radeon 780M、ORT 1.24.4 上保存的四个验证输入及 43 / 256 token 固定输入。存档波形与诊断中间量均有限，重复与生产图对比通过；profile 确认卷积及矩阵乘法由 `DmlExecutionProvider` 执行，输出类型为 `float16`，没有 CPU 神经算子。256 token 的最大波形绝对误差为 0.149994，RMSE 为 0.006028，原工程筛查失败。本轮没有为恢复证据重新执行 GPU；图哈希、公开类型、存档 NPZ 与 profile 重新核对，原始失败报告保留。

CPU 使用同一候选独立执行四个保存输入（1 / 2 / 7 / 19 token）。输出和诊断量均有限，重复检查通过，这四个输入的工程数值指标通过；最大绝对误差最高为 0.019469，RMSE 最高为 0.000893。profile 中 Conv、FusedConv、ConvTranspose、MatMul、FusedMatMul 全部是 `CPUExecutionProvider/.../float`。因此这台机器上的 CPU 路径是 FP16 存储、FP32 神经计算，不能据此宣称原生 CPU FP16 加速或内存减半。CPU 本轮未测 43 / 256 token 长度，也未做听感验收。

紧凑证据、报告 SHA-256、每组误差和原始计时见 [实验数据](../experiments/data/2026-09-27-acoustic-fp16-finite.json)。GPU 原始失败记录为 `.cache/directml-fp16-20260927/result.json`，恢复记录为该目录中的 `finite-experiment-recovered.json`；CPU 原始记录为 `.cache/cpu-fp16-acoustic-20260927/result.json`。原混合声码器通过记录见 [混合精度实验](directml-mixed-vocoder-780m-20260927.md)，其阈值与准入没有改变。

## 复现与诊断

准备全可转换 FP16 候选仍使用原转换器，输出到独立目录：

```powershell
.venv-amd/Scripts/python.exe tools/convert_sovits_onnx_fp16.py --source models/navi-cpu-amd/sovits --output models/navi-cpu-amd/sovits-fp16-new --fp16-scope all
```

CPU 独立执行与有限准入：

```powershell
.venv-amd/Scripts/python.exe research/tools/directml_acoustic_precision.py --backend cpu --saved-only --baseline models/navi-cpu-amd/sovits --candidate models/navi-cpu-amd/sovits-fp16-new --output .cache/cpu-fp16-new --repeats 1 --publish-finite-experiment
```

DirectML 使用 `--backend directml`，去掉 `--saved-only` 可包含两个固定长输入。该工具记录数值筛查实际结果；`--publish-finite-experiment` 不等同于 `--publish-screen`。

只读诊断需要显式指定选择有限准入的执行配置：

```powershell
.venv-amd/Scripts/python.exe -m sakuratts doctor models/navi-cpu-amd/model-fp16-experimental.json --backend cpu --profile fp16
.venv-amd/Scripts/python.exe -m sakuratts doctor models/navi-cpu-amd/model-fp16-experimental.json --backend directml --profile fp16
```

Doctor 校验已有文件与证据并探测依赖，不执行推理或确认听感。未指定有限准入配置时，未通过原工程筛查的候选仍会被拒绝。
