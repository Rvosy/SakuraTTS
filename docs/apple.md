# Apple M 系列推理

`backend="mlx"` 将已有 MLX 组件接入公共 Engine，使用 Metal 执行 GPT decode 与声学 flow / decoder。当前是实验性原生 V2Pro 路径，只接受预制的 `sakuratts-sovits-decode-fp32-v1` 声学包、GPT 包、前端资源和参考条件。Windows 的 V2ProPlus ONNX 包不能直接用于此后端。

## 安装与调用

已有预制资源配置可以用 `sakuratts convert --config OLD --output NEW` 打包；配置须声明 `backend.preferred: mlx`，打包时按原生 V2Pro 格式核对资源。这与读取原始 `.ckpt` / `.pth` 的转换不同，后者尚未接入 MLX。

在 Apple silicon 原生 arm64 Python 环境安装：

```sh
python -m pip install -e '.[mlx,japanese]'
sakuratts doctor --backend mlx
sakuratts tts MODEL --backend mlx --profile fp32 --text 'こんにちは。' --output outputs/hello.wav
```

```python
from sakuratts import Engine

with Engine.load("MODEL", backend="mlx", profile="low-memory") as engine:
    audio = engine.synthesize("こんにちは。")
```

`doctor` 检查 macOS / arm64、MLX 与 Metal 可用性。传入模型时另核对资源身份和格式，不执行语音生成。选择 `mlx` 不会回退到其他设备；Windows、Intel Mac、缺少 MLX 或 Metal 不可用时会给出具体错误。

## 精度与模型限制

GPT Prefill 在 CPU 使用 FP64，decode 在 Metal 使用 FP32；声学 encoder 在 CPU 使用 FP32，flow / decoder 在 Metal 使用 FP32。这些阶段和设备写入合成报告。选用这条混合执行路径，是因为[历史真机实验](../research/notes/compatibility-evidence-20260920.md)中的全 Metal FP32 路线曾出现数值超差，不能将其结果视为已经通过。

`fp32` 保留模型和请求缓存；`low-memory` 在语义生成后释放 GPT 请求状态；`minimum-memory` 让 GPT 与声学模型错峰加载。后两种策略会改变资源生命周期，需在目标 Mac 上测量延迟与内存。预设定义集中在[推理档位](inference-profiles.md)。

FP16、V2ProPlus、公共原始权重转换、HTTP 入口和整合包均未完成。此后端的当前入口是低级 Engine / `tts` 配合预制参考包；`serve` 会提前拒绝 MLX。

## 验证范围

[公共入口测试](../tests/test_mlx_runtime.py)使用计算替身检查模型装配、取消、资源释放、错峰和格式拒绝。它们验证软件边界，不验证 Metal 数值或性能。当前这次适配在 Windows 上开发，尚未完成 Apple 真机复验；历史组件通过项、失败项与听音反馈保留在原始研究记录中。
