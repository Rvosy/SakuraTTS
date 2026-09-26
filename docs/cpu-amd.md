# CPU 与 AMD / DirectML 推理

CPU 和 AMD 各提供一套配置。只指定 `backend` 即使用对应默认值，也可以显式写出 `profile`。

| 后端 | 默认档位 | GPT | 声学 | 默认线程与容量 |
| --- | --- | --- | --- | --- |
| `cpu` | `int8` | ONNX Runtime CPU 动态 INT8 | ONNX Runtime CPU FP32 | GPT 4 线程，声学 8 线程，KV 容量 2048 |
| `directml` | `fp16` | DirectML FP16，Decode KV 保留在 GPU | DirectML 全图 FP16 | GPT 的 CPU 部分 4 线程，声学的 CPU 部分 2 线程，KV 容量 1280 |

CPU 模式不创建 GPU 会话。AMD 模式让 GPT Transformer 与声学模型使用 GPU，文本前端、embedding 和采样仍使用 CPU。线程数不表示 GPU 并行度。具体默认值与可用组合由 [profiles.py](../src/sakuratts/profiles.py) 定义，CUDA 和 MLX 的档位保持各自行为。

Radeon 780M 的 Genie 前后复测、完整请求、内存和试听见 [Genie 对比](../research/notes/genie-comparison-20260927.md)。此前候选、失败结果和数值检查保存在 [精度实验](../research/notes/cpu-amd-precision-listening-20260927.md)；其中其他 CPU / AMD 档位不再作为当前公开配置。选择这两条路径的原因见 [ADR 0006](adr/0006-device-precision-and-directml-kv.md)，硬件与质量验证范围见[兼容矩阵](specs/compatibility-matrix.md)。

上述 CPU 与 AMD 性能测量均来自 `onnxruntime-directml 1.24.4`，CPU 使用其中的 `CPUExecutionProvider`。单独安装 `cpu` extra 使用 `onnxruntime 1.30.0`；这些已保存的延迟数据不能直接视为 1.30.0 的实测结果。

## 安装运行环境

以下命令在仓库根目录执行，以 Windows x64、Python 3.12 为例。使用独立环境，保留已有 Genie 或 CUDA 安装。

仅使用 CPU：

```powershell
python -m venv .venv-cpu
.venv-cpu\Scripts\python.exe -m pip install -e ".[cpu,japanese-text,server]"
.venv-cpu\Scripts\python.exe -m pip check
```

在同一环境中使用 CPU 与 AMD：

```powershell
python -m venv .venv-directml
.venv-directml\Scripts\python.exe -m pip install -e ".[directml,japanese-text,server]"
.venv-directml\Scripts\python.exe -m pip check
```

后文的 `sakuratts` 指所选环境的 `Scripts\sakuratts.exe`。不提供 HTTP 服务时可以省略 `server`。DirectML 需要 Windows、支持 DirectX 12 的 GPU 与可用驱动。

DirectML 版 ORT 同时提供 `CPUExecutionProvider`。同一解释器中只安装一种 ORT 发行包；`directml` 不与 `cpu` 或 `japanese` extra 混装，它们分别引入共用模块目录的 `onnxruntime-directml` 和 `onnxruntime`。日文依赖使用 `japanese-text`，版本见 [pyproject.toml](../pyproject.toml)。

普通推理不需要 PyTorch。原始检查点转换和新参考编码使用独立准备环境，参见[模型准备流程](setup-windows-nvidia.md#转换原始模型)和[参考 API](python-api.md)。

## 准备模型

当前公共路径使用 V2ProPlus 模型与准备好的参考条件。基础转换输出原始 FP32 GPT 包和 `sakuratts-sovits-onnx-v1` 整图声学包；它是后续精度导出的输入，不是完成所有运行资源准备的标志。

已有完整官方源码、辅助模型和准备解释器时：

```powershell
sakuratts convert --backend cpu --gpt voices/model.ckpt --sovits voices/model.pth --reference voices/reference.wav --reference-text "参考音声です。" --official-source tools/GPT-SoVITS --python runtime/preparation/python.exe --output models/cpu-amd
```

示例路径替换为实际输入。CPU / AMD 基础转换不需要 CUDA 声学解释器；辅助模型必须事先存在，转换器不会自动下载。`--backend` 写入模型建议的执行设备，精度资源仍须按下述步骤准备。普通加载不会转换或覆写权重。

先用包含 ONNX 与转换依赖的准备解释器导出共享的 FP32 ONNX 中间资源：

```powershell
PREPARATION_PYTHON -m sakuratts._internal.conversion.export_gpt_onnx --gpt models/cpu-amd/gpt
```

将 `PREPARATION_PYTHON` 换成准备环境的 Python 路径。导出新增 `gpt/onnx/`，保留原始权重。各精度资源通过 manifest 绑定原模型与转换来源；ONNX 运行时校验实际使用的图和 embedding 文件，不再扫描未使用的原始 GPT 权重归档。原始归档由准备工具或读取它的 NumPy 执行器校验。

### CPU 资源

```powershell
PREPARATION_PYTHON -m sakuratts._internal.conversion.export_gpt_onnx --gpt models/cpu-amd/gpt --precision int8
```

此步骤生成 `gpt/onnx-int8/`。CPU 的 GPT 常量线性权重按通道 INT8 动态量化，图 I/O、KV、embedding 和采样 logits 保持 FP32；声学继续使用基础 FP32 包。量化可能改变生成序列和音频长度，需要结合试听检查发音与长句完整性。

### AMD 资源

```powershell
PREPARATION_PYTHON -m sakuratts._internal.conversion.export_gpt_onnx --gpt models/cpu-amd/gpt --precision fp16
PREPARATION_PYTHON -m sakuratts._internal.conversion.export_gpt_directml --gpt models/cpu-amd/gpt --precision fp16 --capacity 1280
```

这两步生成 `gpt/onnx-fp16/` 和 `gpt/directml-fp16-cap1280/`。GPT 的浮点图 I/O 与 KV 使用 FP16；embedding 和采样使用 FP32。Prefill 与 Decode 使用独立 Session，Decode 的两组 GPU KV 缓冲区交替读写。

声学另需有独立 DirectML 执行记录的全图 FP16 包，不能用仅转换声码器的包代替。转换、执行检查和已有候选的来源见[独立精度实验](../research/notes/cpu-amd-precision-listening-20260927.md)。模型描述中的 `acoustic` 指向模型目录内的该候选子目录；保留指向原 FP32 声学包的 CPU 模型描述。模型路径规则见[模型格式](model-format.md)。

AMD 使用 `finite` 准入：发布实验记录时检查有限输出、I/O、重复性和 GPU 执行，原 FP32 误差筛查结果继续保留。加载时校验已发布记录、对应后端及实际图和权重的文件身份；CPU 或 CUDA 的执行记录不能代替 DirectML 记录。历史测试的线程数、CPU arena 和适配器序号是测量条件，不限制后续资源选择。FP16 的模型范围不要求整数索引、公共声学 I/O、文本处理和采样全部改成半精度；执行成功也不等于音质验收。

## 生成音频

```powershell
sakuratts tts MODEL_CPU --backend cpu --text "こんにちは。" --output outputs/cpu.wav
sakuratts tts MODEL_AMD --backend directml --text "こんにちは。" --output outputs/amd.wav
```

`MODEL_CPU` 和 `MODEL_AMD` 分别指向匹配声学资源的模型目录或描述文件。输出文件必须尚不存在。上述命令分别等同于显式指定 `--profile int8` 和 `--profile fp16`。

需要查看或保存默认执行选项时，使用 [cpu.json](../examples/cpu.json) 和 [directml.json](../examples/directml.json)：

```powershell
sakuratts tts MODEL_CPU --backend cpu --profile int8 --experimental examples/cpu.json --text "こんにちは。" --output outputs/cpu-explicit.wav
sakuratts tts MODEL_AMD --backend directml --profile fp16 --experimental examples/directml.json --text "こんにちは。" --output outputs/amd-explicit.wav
```

Python 使用同一套选择：

```python
from sakuratts import Engine

with Engine.load("models/cpu-amd/model-amd.json", backend="directml") as engine:
    audio = engine.synthesize("こんにちは。")
    audio.save("outputs/amd-python.wav")
    print(audio.report["gpt_device"], audio.report["acoustic_device"])
```

省略 `backend` 时使用模型的 `backend.preferred`；模型未声明时仍默认 CUDA。CPU / AMD 的其他旧预设会明确拒绝，不能通过精度或 GPT 设备覆盖重新选择历史候选。

## 服务配置

在 YAML 中写入设备与模型即可使用该设备默认档位：

```yaml
sakuratts:
  model: models/cpu-amd/model-amd.json
  backend: directml
  profile: fp16
```

可将 [AMD 示例](../examples/tts-amd.example.yaml)复制为 `configs/tts_infer.amd.yaml`，从仓库根目录启动：

```powershell
sakuratts serve -c configs/tts_infer.amd.yaml
```

CPU 服务可将 [CPU 示例](../examples/tts-cpu.example.yaml)复制为 `configs/tts_infer.cpu.yaml`，填写对应 CPU 模型路径。省略 `profile` 也会选择各自默认档位。YAML 中的相对模型路径按服务启动目录解析，搬迁后须保持目录关系或填写新的路径。

命令行 `--backend` 覆盖配置中的后端；`--profile` 覆盖配置中的预设名。`sakuratts.runtime_options` 覆盖预设中的资源参数，显式 `--experimental FILE` 再覆盖同名值。设备、GPT 精度与声学范围仍须满足对应路径要求。配置合并由 [read_inference_configuration 与 Inference](../src/sakuratts/engine.py) 定义。

CPU / AMD 在主解释器中执行 GPT 与声学，不启动 `acoustic_python` 声学 worker。经典日文前端仍使用 `frontend_python`；旧配置省略时继续沿用 `acoustic_python`。安装路径需要保持有效，见[日文运行资源](japanese-runtime.md)。

## 调整资源与休眠

默认资源参数由 [profiles.py](../src/sakuratts/profiles.py) 与 [CPUEngine](../src/sakuratts/backends/cpu/engine.py) 定义。可以显式调整 `threads`、`gpt_threads`、`capacity` 和 `policy`，但应重新测量完整请求与占用。

`capacity` 包含文本、参考语义和已生成语义。AMD 修改容量后必须重新导出匹配容量的静态 decode 资源；缺包或超过容量时明确报错，不截断输入、不自动换设备。更长文本也可以通过公共分句选项减少每片长度。

当前两条路径都使用完整 Prefill 注意力矩阵，`gpt_prefill_query_chunk_size` 必须为 `0`。CPU arena 默认关闭，可以显式调整；它不限制 GPU 分配。`device_id` 是 DirectML 的 DXGI 适配器索引，默认 `0`；CPU 要求为 `0`。不能用 `Win32_VideoController` 返回顺序代替 DXGI 索引。AMD 可调整声学线程、CPU arena 和显卡选择，DirectML 所需的顺序执行与禁用 mem pattern 由执行器设置。

`policy=resident` 保留权重和请求缓冲区；`release-state` 在每片语义生成后释放 GPT KV；`staged` 交替加载 GPT 和声学，减少同时驻留的模型，也增加等待。后者用于 Python Engine、`tts` 或显式启用的 `managed` HTTP，默认 `direct` HTTP 拒绝错峰策略。

长期待机使用进程休眠：

```powershell
sakuratts serve -c configs/tts_infer.amd.yaml --runtime-mode managed --idle-sleep-seconds 60
```

控制服务继续运行，空闲后退出推理进程树；下一次请求或唤醒重新加载。准备完成不等于执行预热。提前唤醒、取消和实际验收范围见[后台运行指南](background-runtime.md)。

## 检查与比较

```powershell
sakuratts doctor MODEL_CPU --backend cpu
sakuratts doctor MODEL_AMD --backend directml
```

诊断检查所选默认档位实际使用的 GPT 精度资源、静态图、声学准入、文件哈希、参考身份、解释器依赖与所需 provider。`gpt_resources` 记录图路径、精度和 KV 存储方式。普通诊断不加载模型 Session、不执行 GPU 计算，不能代替真实生成或音质验收。

运行时由各模型 loader 在消费资源时校验文件；同次装配复用检查结果，重新加载时重新检查。Engine 只核对所选档位与包内实际声学精度、范围，不重复扫描文件或复查配置副本。

DirectML 初始化失败时明确报错，不会把 GPT 或声学整体静默切到 CPU。ORT 可将图内部分算子分配给 CPU；实际 GPU 工作量须结合 profile 或设备计量确认。合成报告分别记录 `gpt_device`、`acoustic_device`、`gpt_precision` 和 `acoustic_precision`。

比较配置时同时记录完整请求时间、首 PCM 时间、实际波形长度、进程树工作集与 GPU 内存。核显共享系统资源，不能将工作集与 GPU 共享内存直接相加，也不能将生成更短音频解释为固定工作量加速。计量方法见[基准协议](specs/benchmark-protocol.md)。
