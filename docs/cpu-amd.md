# CPU 与 AMD / DirectML 推理

只指定 `backend` 即使用对应默认值，也可以显式写出 `profile`。CPU 另有两个省内存档位，保持相同计算精度。

| 后端 | 默认档位 | GPT | 声学 | 默认线程与容量 |
| --- | --- | --- | --- | --- |
| `cpu` | `balanced` | ONNX Runtime CPU 动态 INT8 | ONNX Runtime CPU FP32 | GPT 4 线程，声学 8 线程，KV 容量 2048 |
| `directml` | `fp16` | DirectML FP16，Decode KV 保留在 GPU | DirectML 全图 FP16 | GPT 的 CPU 部分 4 线程，声学的 CPU 部分 2 线程，KV 容量 1280 |

CPU 模式不创建 GPU 会话。AMD 模式让 GPT Transformer 与声学模型使用 GPU，文本前端、embedding 和采样仍使用 CPU。线程数不表示 GPU 并行度。具体默认值与可用组合由 [profiles.py](../sakuratts/profiles.py) 定义，CUDA 和 MLX 的档位保持各自行为。

Radeon 780M 的 Genie 前后复测、完整请求、内存和试听见 [Genie 对比](../research/notes/genie-comparison-20260927.md)。此前候选、失败结果和数值检查保存在 [精度实验](../research/notes/cpu-amd-precision-listening-20260927.md)；其中其他 CPU / AMD 档位不再作为当前公开配置。选择这两条路径的原因见 [ADR 0006](adr/0006-device-precision-and-directml-kv.md)，硬件与质量验证范围见[兼容矩阵](specs/compatibility-matrix.md)。

上述九月精度与 Genie 对比使用 `onnxruntime-directml 1.24.4`，CPU 使用其中的 `CPUExecutionProvider`。单独安装 `cpu` extra 使用 `onnxruntime 1.30.0`；九月保存的延迟数据不能直接视为 1.30.0 的实测结果。本文下方的 CPU 内存档位另以 ORT 1.30.0 测量。

CPU ORT 1.30.0 与 DirectML ORT 1.24.4 的原始权重转换、短长句生成和切换失败恢复已另行实测，见[准备与恢复验证](../research/notes/backend-preparation-recovery-20260927.md)。该轮验证不作为新的性能对照。

Windows 整合包由同一个 [CPU/AMD recipe](../packaging/recipes/windows-cpu-amd-ja.toml) 构建，通过配置或启动参数切换；无需分发独立 CPU 包和 AMD 包。使用、离线构建及验收见[整合包指南](portable-bundle.md)。HTTP 后端在服务启动时确定，修改配置后重启，不在单个 `/tts` 请求中切换。

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

当前公共路径使用 V2ProPlus。`convert --backend` 会准备所选后端的完整资源，检查成功后才发布模型目录。普通推理加载不会转换或覆写权重。

已有官方源码、辅助模型和准备解释器时，按目标后端转换：

```powershell
sakuratts convert --backend cpu --gpt voices/model.ckpt --sovits voices/model.pth --reference voices/reference.wav --reference-text "参考音声です。" --official-source tools/GPT-SoVITS --python runtime/preparation/python.exe --output models/voice-cpu
sakuratts convert --backend directml --gpt voices/model.ckpt --sovits voices/model.pth --reference voices/reference.wav --reference-text "参考音声です。" --official-source tools/GPT-SoVITS --python runtime/preparation/python.exe --output models/voice-amd
```

将示例路径替换为实际输入。准备环境需要转换依赖、ONNX Runtime 和辅助模型；辅助模型不会自动下载。可省略参考音频与转写，之后通过参考 API 准备。

CPU 转换生成 INT8 GPT sidecar 和 FP32 声学包。AMD 转换生成 FP16 Prefill、匹配容量的静态 Decode 图和全图 FP16 声学包。图导出在 `--python` 指定的准备解释器中完成，无需执行目标显卡上的筛查实验。

AMD 声学加载不要求 `experimental-directml-finite.json` 或其他筛查报告。需要比较精度、重复性或设备执行时，可单独运行实验工具；报告保留测量条件和失败结果，供评估与试听参考。

需要指定其他静态容量或非零适配器时，转换命令接受与推理相同的 `--experimental FILE`。例如文件内容为 `{"capacity": 2048, "device_id": 1}`，转换和推理均传入该文件。线程、驻留策略等参数只影响执行，不改变转换产物身份。默认仍使用本页开头的两套配置。

CPU 与 AMD 的声学精度不同，当前分别发布模型目录。使用同一对原始权重启动 HTTP 服务时，转换缓存按后端和 AMD 静态容量区分，后续启动复用对应产物；不需要手工修改模型 JSON 或调用内部导出工具。切换原始 GPT / SoVITS 权重也会先补齐当前后端的资源。

## 生成音频

```powershell
sakuratts tts MODEL_CPU --backend cpu --text "こんにちは。" --output outputs/cpu.wav
sakuratts tts MODEL_AMD --backend directml --text "こんにちは。" --output outputs/amd.wav
```

`MODEL_CPU` 和 `MODEL_AMD` 分别指向匹配声学资源的模型目录或描述文件。输出文件必须尚不存在。上述命令分别等同于显式指定 `--profile balanced` 和 `--profile fp16`。

需要查看或保存默认执行选项时，使用 [cpu.json](../examples/cpu.json) 和 [directml.json](../examples/directml.json)：

```powershell
sakuratts tts MODEL_CPU --backend cpu --profile balanced --experimental examples/cpu.json --text "こんにちは。" --output outputs/cpu-explicit.wav
sakuratts tts MODEL_AMD --backend directml --profile fp16 --experimental examples/directml.json --text "こんにちは。" --output outputs/amd-explicit.wav
```

Python 使用同一套选择：

```python
from sakuratts import Engine

with Engine.load("models/voice-amd", backend="directml") as engine:
    audio = engine.synthesize("こんにちは。")
    audio.save("outputs/amd-python.wav")
    print(audio.report["gpt_device"], audio.report["acoustic_device"])
```

省略 `backend` 时使用模型的 `backend.preferred`，模型未声明时默认 CUDA。CPU / AMD 默认分别选择 `balanced` / `fp16`；显式执行选项可以覆盖预设默认值。

### CPU 省内存档位

三个档位都使用 INT8 GPT、FP32 声学，默认容量仍为 2048，采样和输出长度规则相同。`balanced`（均衡）保留模型和 KV，优先连续请求速度；`low-memory`（省内存）在语义生成后释放 KV，使用 256 帧声码器分块；`minimum-memory`（极限省内存）使用 128 帧分块，并让 GPT / 声学、latent / vocoder 两层都错峰加载。极限档会增加每片段的加载和重叠计算时间。

256 帧兼顾重复计算和工作区大小；128 帧用于极限档。64 帧降低局部声码器工作区，但本机完整峰值基本不变，Linux 连续长句高水位也未见实质下降；重叠计算和请求耗时增加，因此不作为默认值。测量条件、数据与未解决的 GPT 峰值见 [CPU 档位测量](../research/notes/cpu-profiles-20261026.md)。

```sh
sakuratts tts MODEL_CPU --backend cpu --profile low-memory --text "こんにちは。" --output outputs/low-memory.wav
sakuratts serve MODEL_CPU --backend cpu --profile minimum-memory --runtime-mode managed
```

Python 使用 `Engine.load(model, backend="cpu", profile="low-memory")`；服务 YAML 使用 `sakuratts.profile`。线程数通过 `sakuratts.runtime_options` 的 `threads` 和 `gpt_threads` 调整，默认值由 [profiles.py](../sakuratts/profiles.py) 定义，与部署服务器无关。旧名 `int8` 对应 `balanced`，`compact` 对应 `low-memory`；现有 JSON / YAML 参数仍覆盖档位默认值。

### CPU 声码器分块

encoder / flow 处理完整上下文，声码器按感受野保留重叠输入，拼回完整波形后统一转换 PCM。分块不改变文本切分、采样、长度上限或语速；不同算子形状可能产生浮点舍入差异。

两个省内存档的 `acoustic_chunk_frames="auto"` 根据模型包选择执行方式：分块包使用上面的默认块长；旧整图包沿用原有 KV 释放或 GPT / 声学错峰，声学内部仍整图执行。均衡档加载分块包时一次执行完整声码器。音频报告和 `doctor` 显示实际块长与 Session 策略；只有分块包才能获得分块与声学内部错峰的收益。

显式设置非负整数块长要求分块包，配错整图包会报错；`0` 表示整段双图执行。覆盖 `acoustic_session_policy` 为 `resident` 时，`auto` 使用 256 帧；设为 `staged` 时使用 128 帧。CPU 不调用 CUDA arena 收缩，不新增强制 GC。

原始权重转换时，在前文 CPU 转换命令上增加 `--experimental examples/cpu-low-memory.json`，输出到新的模型目录。HTTP 使用原始权重和准备环境时，两个省内存档自动准备并共用 FP32 分块缓存；普通模型加载不做转换。已有 FP32 ONNX 声学包可以复用本地带 ONNX 的准备解释器：

```sh
python -m sakuratts.prepare.export_sovits_chunks --source MODEL_CPU/acoustic --output models/acoustic-cpu-chunked --precision fp32
```

在新的模型描述文件中，将 `acoustic`（旧式配置为 `sovits`）指向新目录，GPT、前端和参考继续引用原有资源。相对路径以描述文件所在目录为准。保留原描述文件和整图声学包，便于对照。

```sh
sakuratts tts MODEL_CPU_CHUNKED --backend cpu --profile low-memory --text "こんにちは。" --output outputs/chunked.wav
sakuratts serve MODEL_CPU_CHUNKED --backend cpu --profile minimum-memory --runtime-mode managed
```

执行选项示例为 [cpu-low-memory.json](../examples/cpu-low-memory.json) 和 [cpu-minimum-memory.json](../examples/cpu-minimum-memory.json)。`examples/cpu-compact.json` 继续保留，供旧命令复用。

## 服务配置

在 YAML 中写入设备与模型即可使用该设备默认档位：

```yaml
sakuratts:
  model: models/voice-amd
  backend: directml
  profile: fp16
```

可将 [AMD 示例](../examples/tts-amd.example.yaml)复制为 `configs/tts_infer.amd.yaml`，从仓库根目录启动：

```powershell
sakuratts serve -c configs/tts_infer.amd.yaml
```

CPU 服务可将 [CPU 示例](../examples/tts-cpu.example.yaml)复制为 `configs/tts_infer.cpu.yaml`，填写对应 CPU 模型路径。省略 `profile` 也会选择各自默认档位。YAML 中的相对模型路径按服务启动目录解析，搬迁后须保持目录关系或填写新的路径。

命令行 `--backend` 覆盖配置中的后端，`--profile` 覆盖预设名。`sakuratts.runtime_options` 覆盖预设，显式 `--experimental FILE` 再覆盖同名值。所选设备、实现与实际图精度需要相容。配置合并见 [read_inference_configuration 与 Inference](../sakuratts/engine.py)。

CPU / AMD 在主解释器中执行 GPT 与声学，不启动 `acoustic_python` 声学 worker。经典日文前端仍使用 `frontend_python`；旧配置省略时继续沿用 `acoustic_python`。安装路径需要保持有效，见[日文运行资源](japanese-runtime.md)。

DirectML 不按 GPU 厂商过滤设备；AMD、Intel 等硬件仍需分别通过实际执行验收，当前实测范围不能外推。便携包的 Windows 与 CPU 指令集要求见[整合包指南](portable-bundle.md)。

## 选择 AMD 显卡

先查看当前机器的 DXGI 适配器列表：

```powershell
sakuratts doctor --backend directml
```

输出中的 `directml.adapters` 按 DXGI 顺序列出 `device_id`、`description`、专用显存与共享系统内存字节数，以及可用于对照 WDDM 计量的 `luid`。实现见 [list_adapters](../sakuratts/backends/directml/devices.py)。列表可能包含软件适配器；枚举结果只说明设备身份，不代表它已通过模型执行测试。硬件查询失败时，诊断保留原因，推理仍由实际 Session 初始化决定能否执行。

`device_id` 默认是 `0`，不会自动选择显存最多或速度最快的显卡，也不能用 `Win32_VideoController` 的返回顺序代替。核显与独显共存时，将查到的目标编号写入配置。例如目标编号为 `1`：

```yaml
sakuratts:
  model: models/voice-amd
  backend: directml
  runtime_options:
    device_id: 1
```

CLI 可在 `--experimental FILE` 指定的 JSON 中写入 `{"device_id": 1}`；Python 使用 `Engine.load(..., backend="directml", experimental={"device_id": 1})`。同一个编号传给 GPT Prefill、静态 Decode 与声学 Session，重新加载及 managed 唤醒继续使用该选择。应用层不会因初始化失败而改用另一张显卡或整体重试 CPU。

非零适配器的参数传递与 Session 缓存分配已接通，并有无 GPU 计算的回归测试；当前真实设备记录仍以 Radeon 780M 为限。独显及多张物理显卡上的执行、内存占用和音频结果尚未验收，详见[兼容矩阵](specs/compatibility-matrix.md)。

## 调整资源与休眠

默认资源参数由 [profiles.py](../sakuratts/profiles.py) 与 [ORTEngine](../sakuratts/backends/ort.py) 定义。可以显式调整 `threads`、`gpt_threads`、`capacity` 和 `policy`，但应重新测量完整请求与占用。

`capacity` 包含文本、参考语义和已生成语义。AMD 修改容量后必须重新导出匹配容量的静态 decode 资源；缺包或超过容量时明确报错，不截断输入、不自动换设备。更长文本也可以通过公共分句选项减少每片长度。

CPU ONNX GPT 保留完整容量的连续地址，底层使用私有匿名映射，物理页随写入进入内存。它避免 NumPy 大页提示造成的额外常驻，不复制历史来扩容，也不修改系统分配策略。虚拟地址容量与 RSS 是不同指标；实际收益依赖操作系统，见 [CPU KV 的分配依据](adr/0006-device-precision-and-directml-kv.md#cpu-kv-的物理内存分配)。

当前两条路径都使用完整 Prefill 注意力矩阵，`gpt_prefill_query_chunk_size` 必须为 `0`。CPU arena 默认关闭，可以显式调整；它不限制 GPU 分配。CPU 的 `device_id` 要求为 `0`；AMD 的显卡选择见[上节](#选择-amd-显卡)。AMD 可调整声学线程与 CPU arena，DirectML 所需的顺序执行与禁用 mem pattern 由执行器设置。

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

诊断检查所选 GPT 图、声学资源路径、参考数组、解释器依赖与 provider，不比对文件哈希或实验报告。普通诊断不加载模型 Session、不执行 GPU 计算，不能代替真实生成或音质验收。

模型加载器直接读取所选文件，检查计算需要的格式、张量形状与容量。Engine 核对显式精度预设与实际声学精度，不按来源哈希、字典 MD5、固定前端版本或历史筛查结论拒绝使用。

DirectML 初始化失败时明确报错，不会把 GPT 或声学整体静默切到 CPU。ORT 可将图内部分算子分配给 CPU；实际 GPU 工作量须结合 profile 或设备计量确认。合成报告分别记录 `gpt_device`、`acoustic_device`、`gpt_precision` 和 `acoustic_precision`。

比较配置时同时记录完整请求时间、首 PCM 时间、实际波形长度、进程树工作集与 GPU 内存。核显共享系统资源，不能将工作集与 GPU 共享内存直接相加，也不能将生成更短音频解释为固定工作量加速。计量方法见[基准协议](specs/benchmark-protocol.md)。
