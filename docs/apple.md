# Apple M 系列推理

`backend="mlx"` 使用 Metal 执行 GPT decode 与声学 flow / decoder，支持原生 V2Pro、FP32 和日文。公共 Engine、CLI 和 HTTP 共用这条路径。原始 `.ckpt` / `.pth` 转换、新参考编码使用独立的 CPU 准备环境；日常推理不导入 PyTorch。

## 整合包

Apple silicon 整合包自带 Python、MLX、日文依赖和 FFmpeg，完整包另带准备组件。模型和参考音频由使用者提供。解压到可写目录后：

1. 运行 `check-runtime.command`，检查包内解释器和 Metal。
2. 将自己的 V2Pro 权重放入 `models/`，复制 `configs/tts_infer.example.yaml` 为 `configs/tts_infer.yaml`，填写权重路径。
3. 运行 `start-server.command`。首次加载转换模型；首次 `/tts` 请求编码参考，之后复用包内缓存。

请求格式见 [API V2](api-v2-guide.md)。显式启用休眠使用 `./start-server.command --runtime-mode managed`；`minimum-memory` 的 HTTP 使用要求与其他后端相同，须启用 managed 模式。

构建目标、依赖选择、打包和离线验收见 [Mac 整合包指南](portable-macos.md)。当前构建目标为 macOS 14 或更高版本的 Apple silicon；实际真机结果见[验证记录](../research/notes/macos-portable-20261004.md)。最低系统版本由包内清单声明，Intel Mac 不使用这个 arm64 包。

## 源码安装与转换

使用原生 arm64 Python，安装主环境：

```sh
python -m pip install -e '.[mlx,japanese,server]'
sakuratts doctor --backend mlx --japanese
```

原始权重转换复用固定 GPT-SoVITS 源码和日文准备工具。准备解释器的依赖、辅助权重与词典须齐全，不能仅安装主运行依赖。

```sh
sakuratts convert --backend mlx \
  --gpt /path/to/voice.ckpt --sovits /path/to/voice.pth \
  --official-source /path/to/GPT-SoVITS --python /path/to/preparation/bin/python3 \
  --language-model /path/to/lid.176.bin \
  --reference /path/to/reference.wav --reference-text '参考音声です。' \
  --output models/voice
sakuratts tts models/voice --backend mlx --text 'こんにちは。' --output outputs/hello.wav
```

已有预制资源仍可用 `sakuratts convert --config OLD --output NEW` 打包。配置须声明 `backend.preferred: mlx`；声学格式为 `sakuratts-sovits-decode-fp32-v1`，不能使用 Windows 的 ONNX 声学包。

```python
from sakuratts import Engine

with Engine.load("models/voice", backend="mlx", profile="low-memory") as engine:
    audio = engine.synthesize("こんにちは。")
```

HTTP 可以读取预制模型，也可通过 `--tts-config` 指定原始权重自动转换。源码安装时在 YAML 的 `sakuratts` 中填写 `backend: mlx`、`official_source`、`python` 和需要的 `language_model`；完整整合包自动绑定准备组件。新参考需保留原始权重，预制参考只作为匹配音频的缓存。

## 精度与模型限制

GPT Prefill 在 CPU 使用 FP64，decode 在 Metal 使用 FP32；声学 encoder 在 CPU 使用 FP32，flow / decoder 在 Metal 使用 FP32。这些阶段写入合成报告。采用混合执行路径的数值依据见[历史真机实验](../research/notes/compatibility-evidence-20260920.md)。

`fp32` 保留模型和请求缓存；`low-memory` 在语义生成后释放 GPT 请求状态；`minimum-memory` 让 GPT 与声学模型错峰加载。定义集中在[推理档位](inference-profiles.md)。选择 MLX 不会自动切换其他后端。

FP16、V2ProPlus 和其他模型家族尚未接入 MLX。当前流程可转换和运行 V2Pro，不代表所有同系列权重均已通过音质验收。

## 验证

`tests/test_mlx_runtime.py` 验证公共请求、切换失败恢复、取消与回收，计算部分使用替身。真机生命周期检查使用包内解释器运行 `scripts/verify_mlx_runtime.py`，检查三种内存策略、连续请求、处理中取消及恢复，输出音频和报告。整合包首次使用与搬迁用 `scripts/verify_portable_first_use.py` 验证。

本次 M4 实机证据与未验收项统一保存在[验证记录](../research/notes/macos-portable-20261004.md)；功能通过不代替人工听音、最低系统和其他设备测试。
