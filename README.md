# SakuraTTS

SakuraTTS 是兼容已验证 GPT-SoVITS 权重的独立推理库，用于 Sakura 的语音生成。Windows 路径使用自有 CuPy CUDA GPT 和独立 ONNX Runtime 声学工作进程，普通推理不导入 PyTorch。

当前准备发布 `0.1.0a1` 开发者预览版：GitHub 保存源码，wheel 和源码包由维护者手动构建、压缩后上传 ModelScope。项目源码采用 MIT；模型、词典及第三方组件各自的许可保持不变。

## 开始使用

先读[预览版安装与发布说明](docs/preview-release.md)。本次交付的是 Python 包和源码，不含 Python、模型、CUDA/cuDNN、词典或预备好的声学运行环境。

在 Windows x64 / Python 3.11 环境中，从源码安装主运行环境。当前 CuPy / NVRTC 要求 Python 环境及 CUDA 头文件所在路径只含 ASCII 字符；建议放在 `D:/SakuraTTS/`，不要放在中文或日文目录中。模型与文本中的非 ASCII 字符不受这项安装限制影响。

```powershell
uv venv --python 3.11 .venv-windows-runtime
uv pip install --python .venv-windows-runtime/Scripts/python.exe -r requirements-windows-runtime.txt
uv pip install --python .venv-windows-runtime/Scripts/python.exe --no-deps .
.venv-windows-runtime/Scripts/sakuratts.exe --version
```

上述命令允许联网获取依赖；缓存完整时可加 `--offline`。下载预览 ZIP 后，也可以直接安装其中的 wheel，具体命令见预览指南。转换器使用单独的开发环境，训练和导出依赖不会进入普通运行包。

按 [Windows 资源准备指南](docs/setup-windows-nvidia.md)准备 GPT、声学模型、日文前端、参考条件和独立声学组件。配置示例在 [examples/runtime.windows.example.json](examples/runtime.windows.example.json)。原 FP32 整图配置可以执行：

```powershell
.venv-windows-runtime/Scripts/sakuratts.exe doctor --nvidia --config models/voice/runtime.json
.venv-windows-runtime/Scripts/sakuratts.exe synthesize --config models/voice/runtime.json --text "おはよう。今日もよろしくね。" --output outputs/hello.wav
```

输出为完整 WAV 和同名 JSON。退出码 `0` 表示正常停止，`2` 表示达到生成上限；未生成过的输出路径才会被接受。`doctor` 检查依赖和资源，不能替代真实推理或音质验收。

## Python 接口

```python
from sakuratts.nvidia import NVIDIAEngine, write_wav

engine = NVIDIAEngine("runtime.json")
try:
    engine.load()
    pcm, report = engine.synthesize("おはよう。今日もよろしくね。", seed=1234)
    write_wav("hello.wav", pcm, report["sample_rate"])
finally:
    engine.close()
```

连续请求可复用同一实例；每个实例只处理一个活动请求。模型转换、参考音频编码在准备阶段完成，普通请求只加载独立产物，不读取官方、Lite 或 Genie 的源码目录。

## 当前支持范围

Windows / NVIDIA 是本轮交付重点。实测对象为 RTX 5060 8 GB、Sakura V2ProPlus、日文单请求、非流式完整 WAV。Mac 的 MLX 路径及原 V2Pro 证据保留，使用方式见[日文运行指南](docs/japanese-runtime.md)。

默认使用 GPT / 声学 FP32 与 baseline attention。已测的低显存配置使用 GPT FP16 与声学分块；速度配置使用 GPT FP32 split-KV 与声学分块。它们都需要显式参数和通过筛查的模型包：

| 自然生成配置 | 热长句完整请求 | 返回 PCM | 独立资源轮全卡峰值增量 |
| --- | ---: | ---: | ---: |
| GPT FP16 / baseline，声学 FP16 分块 256 | 2373.28 ms | 25.46 s | 799 MiB |
| GPT FP32 / split-KV 256，声学 FP16 分块 256 | 1557.08 ms | 25.90 s | 1108 MiB |

计时包括文本前端至完整 PCM 返回；资源采样另跑，不是进程独占显存，也不是流式首包。两组生成工作量不同，不能把最快时间和最低显存拼成同一配置。详细条件见[分块公开入口](docs/experiments/2026-09-20-windows-vocoder-public.md)与[自然生成补测](docs/experiments/2026-09-20-windows-asr.md)。

声学 FP16 保留相对官方 FP32 的严格数值失败。已完成 24 条音频的辅助 ASR，部分内容仍待复听，Windows 新样音的音色与自然度尚未完成人工验收。其他 GPU / 模型、最低显存、中文整链、流式输出及宿主集成还没有正式支持承诺。

## 源码和手动打包

```text
src/sakuratts/   推理、模型包、文本前端、CLI 和 Python 接口
scripts/        模型转换、参考准备、环境准备和手动发布工具
tests/          自动化回归
harness/        数值、资源和性能验证；不进入运行 wheel
examples/       配置示例
docs/           使用说明、契约、研究与原始实验记录
```

运行 `python scripts/build_preview.py --output dist/preview-0.1.0a1` 生成开发者 ZIP 和校验文件；缓存齐全时加 `--offline`。脚本不上传文件。维护者核验后手动上传 ModelScope，`dist/`、模型、虚拟环境和运行输出不入 Git。

本轮封装取舍对照了 [Genie / Lite 的实际发布方式](docs/research/windows-preview-packaging.md)，没有把它们的 Python 包大小或社区性能数字当作本项目的完整安装成本。

## 文档

| 文档 | 内容 |
| --- | --- |
| [开发者预览版](docs/preview-release.md) | wheel / 源码安装、手动构建、ModelScope 分发边界 |
| [Windows 安装与模型准备](docs/setup-windows-nvidia.md) | 独立工作进程、前端、参考和模型转换 |
| [Windows 实测](docs/experiments/2026-09-20-windows-nvidia-backend.md) | 官方对照、速度、显存、生命周期和已知问题 |
| [推理契约](docs/specs/inference-contract.md) / [兼容矩阵](docs/specs/compatibility-matrix.md) | 必须保持的行为与实测范围 |
| [运行时方案](docs/adr/0001-native-gpu-runtime.md) / [实施路线](docs/roadmap.md) | 架构取舍、当前发布优先级和后续候选 |
| [开发环境](docs/setup.md) / [贡献约定](AGENTS.md) | 工具、目录和验证方式 |

历史 Mac 和 Windows 实验继续保存在 `docs/experiments/`，包含未通过项。新结果不会把旧的失败改写成通过。

## 许可

项目自身代码采用 [MIT](LICENSE)。第三方代码、字典及衍生实现的许可和声明位于 [docs/third-party](docs/third-party)，随 wheel 和源码包保留。源码许可不授予任何角色模型、参考音频或 NVIDIA 运行库的再分发权。
