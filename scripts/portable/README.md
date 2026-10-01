# SakuraTTS Windows 整合包

本包提供后台语音服务、CLI 和私有 Python。模型目录为空，请自行准备有权使用的 GPT-SoVITS V2ProPlus 权重和参考音频。当前首次准备流程以日文为验收范围；硬件和音质范围见[兼容矩阵](https://github.com/Rvosy/SakuraTTS/blob/main/docs/specs/compatibility-matrix.md)。

发行组合记录在 `runtime/portable.json`：CPU/AMD 包共用 DirectML ONNX Runtime，由配置或启动参数选择 CPU INT8 或 AMD FP16；NVIDIA 包使用独立 CUDA 运行环境。无需预装 Python、Git、编译器或 CUDA Toolkit。GPU 模式需要兼容的系统驱动。NVIDIA 包的 CuPy/NVRTC 要求英文安装目录（可含空格）；CPU/AMD 包允许中文和空格目录，具体路径仍需通过随包检查。CPU 首次 INT8 转换支持安装目录和 TEMP/TMP 同时包含中文，无需手动修改系统临时目录。完整安装路径为 ASCII 时，启动器沿用包内 cache/tmp；中文安装路径保留外部 TEMP/TMP。

本包面向 Windows 10/11 x64；随包 NumPy 2.4.6 要求 x86-64-v2 指令集，不能视为所有 x64 CPU 通用包。旧 CPU 的完整模型转换链尚未实测。DirectML 的官方基础条件是 Windows 10 1903 及 DirectX 12，这不代表整包已在该最低系统版本验收。代码不按 AMD/Intel 厂商限制硬件；目前真实 GPU 验收仅覆盖 Radeon 780M，Intel 与其他显卡仍需设备测试。

## 检查与启动

双击 `check-runtime.bat` 检查本包默认设备：CPU/AMD 包默认检查 CPU，NVIDIA 包检查 CUDA。AMD 用户另运行：

```bat
sakuratts.bat check-runtime --backend directml
```

多显卡可追加 `--device-id 1`。检查按 DXGI 原编号选择设备，输出适配器名称和软件渲染器标记；DirectML 分别执行 FP32、FP16 矩阵运算并核对实际执行 provider，软件渲染器不能作为 GPU 通过。结果写入 `cache/runtime-check.json`；失败不会改用 CPU 伪装通过。它不验证声音模型或音质。检查失败时报告保留原始异常、所选设备和 Python 路径；可先用 `sakuratts.bat doctor --backend directml` 查看设备列表。若 Python 无法启动，检查是否完整解压、运行库 DLL 是否被隔离；无需另外安装 Python 或 Git。

带 HTTP 的包可双击 `start-server.bat`，默认监听 `127.0.0.1:9880`。没有配置模型时服务保持未加载状态。查看接口说明可访问 `/docs`。

## 原始模型与配置

带 `runtime/preparation/` 的完整包可离线转换原始权重和编码新参考：

1. 将自己的 GPT `.ckpt` 和 SoVITS `.pth` 放进 `models/`。
2. 复制 `configs/tts_infer.example.yaml` 为 `configs/tts_infer.yaml`，填写两项权重路径。CPU/AMD 包的这个模板默认 CPU；也提供 `tts_infer.directml.example.yaml`。
3. 启动 `start-server.bat`，首次转换会显示进度；结果保存在 `cache/`，后续复用。

同一套原始权重可通过启动参数选择模式：

```bat
start-server.bat --backend cpu --profile int8
start-server.bat --backend directml --profile fp16
```

也可在 YAML 的 `sakuratts` 段设置 `backend: cpu` 或 `backend: directml`，省略 `profile` 会使用对应默认值。后台应用可追加 `--runtime-mode managed`；首次转换较慢时可加 `--wake-timeout-seconds 900`。managed 模式先调用 `POST /runtime/wake`，空闲时退出推理进程。切换后端后重启服务；`/tts` 不逐请求切换设备。

CPU 和 AMD 使用不同精度的转换资源，缓存会分别保留。已有转换包时，需选择与设备匹配的模型路径：

```bat
sakuratts.bat doctor models/voice-cpu --backend cpu
sakuratts.bat tts models/voice-cpu --backend cpu --text "こんにちは。" --output hello.wav
start-server.bat models/voice-amd --backend directml
```

外部程序调用 Python 时使用 `Engine.load(model, backend="cpu")` 或 `backend="directml"`；HTTP 和 CLI 共用同一配置契约。线程、显卡编号等参数见[CPU/AMD 指南](https://github.com/Rvosy/SakuraTTS/blob/main/docs/cpu-amd.md)。

## HTTP 请求

接口对齐 GPT-SoVITS API v2 的已支持字段，必须显式传 `parallel_infer=false`。以自己的参考录音和对应转写发起请求：

```powershell
curl.exe -X POST http://127.0.0.1:9880/tts -H "Content-Type: application/json" --data-raw '{"text":"こんにちは。","text_lang":"ja","ref_audio_path":"D:/Voices/reference.wav","prompt_text":"参考音声です。","prompt_lang":"ja","parallel_infer":false}' --output hello.wav
```

首次参考编码由独立 CPU 准备进程完成，之后复用缓存；普通推理不导入 PyTorch。完整字段和限制见 [API v2 指南](https://github.com/Rvosy/SakuraTTS/blob/main/docs/api-v2-guide.md)。

省略准备组件的精简包只使用已有转换模型和参考条件；需要转换原始权重或新参考时会明确提示缺少准备环境。构建和运行不会自动下载缺失模型或依赖。

## 更新与许可证

更新时解压到新目录，再复制自己的模型、配置、参考音频及所需缓存。不要用模板覆盖个人配置。日志在 `logs/`，版本、源码提交和文件哈希在 `bundle-manifest.json`。

各依赖的许可证保留在 `.dist-info`、组件目录和 `licenses/`。准备组件的 `licenses.json` 与 `auxiliary-model-sources.json` 记录辅助模型来源和仍待核对的材料；用户角色和个人录音不属于发行内容。FFmpeg 的实际构建许可见 `licenses/FFmpeg-build-and-license.txt`，不能一概视为 LGPL。
