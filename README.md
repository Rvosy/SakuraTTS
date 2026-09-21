# SakuraTTS

GPT-SoVITS 的轻量推理运行时，以原版 API 和推理行为为兼容目标。HTTP 接受原版的文本、参考音频路径和采样参数；权重转换和参考编码在独立环境完成，主推理进程不导入 PyTorch。

当前接通 Windows / NVIDIA、V2ProPlus 日文单请求，以及原版 `streaming_mode=1` 的按句返回。仍有未实现的原版功能，详见[HTTP 兼容清单](docs/http-api.md)。项目处于开发者预览阶段，不能视为完整替代原版。

## 安装

在 Python 3.11 环境中，从仓库根目录安装：

```powershell
python -m pip install ".[nvidia,japanese,server]"
sakuratts doctor
```

`server` 是可选依赖。不启动 HTTP 服务时可省略。声学 CUDA 和经典日文前端仍使用[单独准备的 Python 工作进程](docs/setup-ort-worker-offline.md)，不能只装上述依赖就开始推理。模型、字典和参考条件需事先准备，运行时不自动下载。

## 准备模型

已有预览版 `runtime.json` 可以直接传给新入口，也可以整理成一个模型目录：

```powershell
sakuratts convert --config models/windows-sakura/runtime.json --output models/mika
sakuratts doctor models/mika
```

转换会复制 GPT、声学、前端和参考包，生成 `model.json`；共享的声学解释器使用明确路径引用。旧资源不被覆盖。从原始检查点转换，需要独立开发环境和受支持的官方源码；完整命令见[快速开始](docs/quickstart.md)。

## 使用

下面的低级 Python 和 `tts` 示例用于已经带参考条件的旧模型包。HTTP 请求使用原版的音频路径和转写，无需参考名称。

```python
from sakuratts import Engine

with Engine.load("models/mika") as tts:
    audio = tts.synthesize("こんにちは。今日はどんな一日でしたか。")
    audio.save("hello.wav")
    print(audio.report["status"])
```

```powershell
sakuratts tts models/mika --text "こんにちは。" --output hello.wav
sakuratts serve models/mika --host 127.0.0.1 --port 9880
```

推荐从 [tts_infer.example.yaml](examples/tts_infer.example.yaml) 配置模型和准备环境，再运行 `start-server.bat -c configs/tts_infer.yaml`。也支持 `python api.py -a 127.0.0.1 -p 9880 -c configs/tts_infer.yaml`。终端显示加载和推理日志，按 `Ctrl+C` 退出并释放模型。

无参数启动只读取明确保存的 `configs/tts_infer.yaml`；没有该文件时服务保持未配置状态，不自动选择示例角色。显式传入现有模型目录仍可使用，新的原始参考音频需要配置独立准备环境。

`serve` 提供原版 `GET/POST /tts`、权重切换、参考音频设置和进程控制接口，另有 `/health`、`/models` 和 `/docs`。此版本没有前端页面。

```powershell
curl.exe -X POST http://127.0.0.1:9880/tts -H "Content-Type: application/json" --data-raw '{"text":"こんにちは。","text_lang":"ja","ref_audio_path":"D:/Voices/reference.wav","prompt_text":"参考音声です。","prompt_lang":"ja"}' --output hello.wav
```

各入口共用 Engine。一次只处理一条请求；服务忙碌时返回 HTTP 409。默认保留 FP32、baseline attention 和原采样参数。达到生成上限时，CLI 写出音频和报告后退出 `2`；HTTP 的 `X-SakuraTTS-Status` 返回 `stopped_at_limit`。

推理配置收敛为 **FP32、FP16 标准、FP16 低显存、FP16 极限**四档，配置文件、显存与耗时见[推理档位](docs/inference-profiles.md)。

## 支持与实测

Windows 历史实测使用 RTX 5060 8 GB、Sakura V2ProPlus、日文非流式请求。声学 FP16 对官方 FP32 的严格数值对照仍有失败。ASR 辅助检查不能替代人工听感验收；中文整链、其他 GPU / 模型、语义 token 流式模式及宿主集成仍待实现或验证。性能条件和失败项见[兼容范围](docs/compatibility.md)与[研究总结](docs/research/reference-parity.md)。

## 仓库导航

```text
src/sakuratts/  Engine、模型目录、转换器、CLI、HTTP 服务
  frontend/    日文、中文、G2PW；第三方衍生代码在 _vendor/
  backends/    CUDA、ONNX 和实验 MLX 实现
  _internal/   推理步骤、模型校验、私有工作进程、离线转换
start-server.bat  Windows 终端服务启动脚本
examples/      Python、服务和模型描述示例
benchmarks/    完整请求、官方对照入口及固定测试文本
tests/         自动化回归
docs/          使用说明、Spec、ADR 和研究结论
tools/         资源准备、模型实验及高级维护工具
scripts/       环境准备和预览版发布
research/      历史诊断工具、实验记录与已保存证据
requirements/  已验证环境的冻结依赖快照
```

依赖声明以 `pyproject.toml` 为准。研究工具与原始证据不进入运行 wheel；新的输出保存在被 Git 忽略的目录。开发与迁移细节见[开发指南](docs/development.md)。

| 文档 | 内容 |
| --- | --- |
| [快速开始](docs/quickstart.md) | 环境、模型转换、首次生成 |
| [推理档位](docs/inference-profiles.md) | 四档配置、显存、耗时与旧配置迁移 |
| [Python API](docs/python-api.md) / [HTTP API](docs/http-api.md) | 调用、输出、错误与生命周期 |
| [模型目录](docs/model-format.md) | model.json、路径及身份校验 |
| [架构](docs/architecture.md) / [开发](docs/development.md) | 模块职责、迁移和验证 |
| [推理契约](docs/specs/inference-contract.md) / [兼容矩阵](docs/specs/compatibility-matrix.md) | 行为不变量与实测边界 |

项目代码采用 [MIT](LICENSE)。第三方代码和字典声明见 [docs/third-party](docs/third-party)，随 wheel 和源码包保留。源码许可不授予角色模型、参考音频或 NVIDIA 运行库的再分发权。
