# SakuraTTS

面向 GPT-SoVITS 模型的独立推理后端，提供 Python 引擎、命令行和兼容原版 `api_v2.py` 的 HTTP 服务。权重转换和新参考音频编码由独立准备进程完成，主推理进程不导入 PyTorch。

项目处于开发者预览阶段，公共入口支持 Windows / NVIDIA、V2ProPlus 日文单请求及按句流式返回。完整的接口能力见 [API V2 使用说明](docs/api-v2-guide.md)，设备、模型与音质的实测范围见[兼容矩阵](docs/specs/compatibility-matrix.md)。

## 快速开始

[Windows / NVIDIA 整合包](docs/portable-bundle.md)自带 Python 与运行依赖，完整包还带有模型转换和参考准备组件。解压后复制 `configs/tts_infer.example.yaml` 为 `configs/tts_infer.yaml`，填写自己的 GPT / SoVITS 权重路径，再运行 `start-server.bat`。模型与参考音频由使用者提供。

源码安装使用 Python 3.11：

```powershell
python -m pip install ".[nvidia,japanese,server]"
sakuratts doctor
```

然后按[快速开始](docs/quickstart.md)准备模型、日文资源和独立工作进程，复制 [HTTP 配置示例](examples/tts_infer.example.yaml)并填写路径，启动服务：

```powershell
.\start-server.bat -c configs/tts_infer.yaml
```

将参考音频路径和转写替换为自己的内容：

```powershell
curl.exe -X POST http://127.0.0.1:9880/tts -H "Content-Type: application/json" --data-raw '{"text":"こんにちは。","text_lang":"ja","ref_audio_path":"D:/Voices/reference.wav","prompt_text":"参考音声です。","prompt_lang":"ja","parallel_infer":false}' --output hello.wav
```

请求须显式传 `parallel_infer=false`。默认服务地址为 `127.0.0.1:9880`，浏览器打开 `/docs` 查看接口；终端按 `Ctrl+C` 退出。客户端接入、流式读取与错误处理见 [API V2 使用说明](docs/api-v2-guide.md)。

已有带参考条件的模型包也可直接调用 Python：

```python
from sakuratts import Engine

with Engine.load("models/mika") as tts:
    audio = tts.synthesize("こんにちは。")
    audio.save("hello.wav")
```

## 文档与源码

| 需求 | 入口 |
| --- | --- |
| 安装、模型准备与首次生成 | [快速开始](docs/quickstart.md) |
| HTTP 客户端接入 | [API V2 使用说明](docs/api-v2-guide.md) |
| 服务配置、休眠与提前唤醒 | [HTTP API](docs/http-api.md)、[后台运行](docs/background-runtime.md) |
| 精度、显存与延迟取舍 | [推理档位](docs/inference-profiles.md) |
| 嵌入 Python 或了解模型格式 | [Python API](docs/python-api.md)、[模型目录](docs/model-format.md) |
| 修改代码与运行验证 | [开发指南](docs/development.md)、[架构](docs/architecture.md) |
| 查找其他指南与规范 | [文档索引](docs/README.md) |

运行代码在 `src/sakuratts/`，测试在 `tests/`，离线维护工具在 `tools/`，构建脚本在 `scripts/`。`research/` 保存实验工具和历史证据，仅随 Git 仓库提供；普通基准入口在 `benchmarks/`。

项目代码采用 [MIT](LICENSE)。第三方代码和字典声明见 [docs/third-party](docs/third-party)。角色模型、参考音频和 NVIDIA 运行库适用各自的许可。
