# 快速开始

当前公共 Engine 面向 Windows / NVIDIA、V2ProPlus，支持日文；加入[英文依赖与资源](english-frontend.md)后支持英文与日英混合。建议用 Python 3.11；模型转换环境与日常运行环境分开。

## 安装运行环境

从仓库根目录执行 `python -m pip install ".[nvidia,japanese,server]"`，或者安装构建好的 wheel 及对应 extras。HTTP 服务需要 `server`，普通 Python / CLI 推理可以省略。

按照[Windows 准备指南](setup-windows-nvidia.md)准备模型、语言资源和[独立声学解释器](setup-ort-worker-offline.md)。经典 pyopenjtalk 的二进制 ABI、字典和来源身份不能用 pyopenjtalk-plus 直接替换。

## 已有转换结果

```powershell
sakuratts convert --config models/windows-sakura/runtime.json --output models/mika
sakuratts doctor models/mika
sakuratts tts models/mika --text "こんにちは。" --output outputs/hello.wav
```

`convert --config` 会复制资源，输出目录必须不存在；声学解释器和可选主字典保留为共享资源的绝对路径。迁移到另一台机器时，需要重新准备它们并更新 `model.json`。也可以直接把旧 `runtime.json` 传给新入口。

`doctor` 检查依赖、资源哈希与身份，不执行 GPU 推理，不能代表音质通过。

需要选择精度与显存档位时，使用[四档配置](inference-profiles.md)：FP32、FP16 标准、FP16 低显存、FP16 极限。通过 `--experimental examples/fp16.json` 等文件选择；FP16 档要求已经准备好的 FP16 chunk256 模型包。

## 从检查点转换

在开发环境安装 `python -m pip install ".[convert,japanese]"`。官方源码须包含参考准备需要的 HuBERT、说话人编码模型、日文字典等资源；转换过程不会补下载。转换使用 V2ProPlus 准备工具，具体权重范围见[兼容矩阵](specs/compatibility-matrix.md)。

```powershell
sakuratts convert --gpt character.ckpt --sovits character.pth --reference reference.wav --reference-text "参考音频的日文转写" --official-source D:/Models/GPT-SoVITS --acoustic-python D:/Runtime/python.exe --output models/mika
```

`--reference` 和 `--reference-text` 可以同时省略，输出只包含模型和前端。可用 `--python` 指定已有的官方开发解释器，`--language-model` 指定已有的 `lid.176.bin`。转换依次准备参考和前端、导出 GPT、导出并验证 ONNX 声学图、校验模型目录。只有全部成功才发布目标目录。输入检查点不被覆盖。

## Python 与服务

```python
import sakuratts

with sakuratts.load("models/mika") as engine:
    audio = engine.synthesize("おはようございます。")
    audio.save("outputs/greeting.wav")
```

复制 [tts_infer.example.yaml](../examples/tts_infer.example.yaml) 为本地 `configs/tts_infer.yaml` 并填写路径，运行 `start-server.bat`。也可显式执行 `start-server.bat -c PATH` 或 `python api.py -c PATH`。已有部署包可用 `sakuratts serve models/mika` 启动，但准备新的原始参考音频仍需配置准备环境。终端持续显示加载、请求和推理日志，按 `Ctrl+C` 停止。服务默认绑定 `127.0.0.1:9880`，当前没有前端页面。首次调用见 [API v2 使用指南](api-v2-guide.md)。当前只兼容 API v2，合成请求须显式传 `parallel_infer=false`，未实现的 V2 功能返回 400；服务配置和缓存细节见 [HTTP API](http-api.md)。

终端按请求显示进度和完成摘要，完整诊断保存到 `logs/sakuratts.log`。需要展开细节时使用 `start-server.bat --log-level debug`；`--log-file PATH` 可以更换日志文件。

桌宠等间歇使用语音的应用，可在启动时加入 `--runtime-mode managed`，并在发起 LLM 请求时提前唤醒。接入流程与可选试合成见[后台驻留与提前唤醒](background-runtime.md)。

服务没有账户、认证或多租户隔离；远程访问应由宿主提供访问控制。
