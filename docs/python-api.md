# Python API

```python
from sakuratts import Engine, Model

model = Model.load("models/mika")
print(model.info())
with Engine.load(model) as engine:
    audio = engine.synthesize("こんにちは。", reference=model.default_reference, seed=1234)
    audio.save("hello.wav")
```

`Model.load` 读取描述和检查路径，不导入 GPU 库。`Engine.load` 准备前端、核对资源身份，GPU 权重在第一次请求中按需加载。包顶层导入不加载 NumPy、CuPy、ORT、MLX、FastAPI 或 PyTorch。

`Audio` 包含单声道 int16 `pcm`、`sample_rate` 和 `report`。`wav_bytes()` 返回 WAV 字节；`save(path)` 写入新文件，拒绝覆盖。`report["status"]` 为 `completed` 或 `stopped_at_limit`，停止原因、片段、参数和计时继续保存在报告中。

这里是已有参考包的低级 Python 接口，字段和默认值保留历史行为；原版兼容入口是 [HTTP API](http-api.md)。

`synthesize` 可指定 `reference`、`seed`、`language`（`ja` / `all_ja`）、`split_method`（`cut0` 至 `cut5`，默认 `cut0`）、`top_k`、`temperature`、`repetition_penalty`、`early_stop_num` 和 `cancel_requested`。默认值沿用原推理契约；相同 NumPy seed 不表示与官方 Torch seed 等价。

一个 Engine 同时只处理一个请求，重入会抛出 `BusyError`。使用 `with` 或显式 `close()` 释放资源；关闭后不能继续合成。失败会沿原链路释放请求状态。宿主应在同一个线程中拥有、调用和关闭引擎，HTTP 服务已执行这一约束。

实验后端选项通过 `Engine.load(path, experimental={...})` 显式传入。支持的键为 `policy`、`use_graph`、`capacity`、`gpt_precision`、`gpt_attention`、`gpt_attention_chunk_size`、`allow_experimental_acoustic_fp16`、`acoustic_arena_shrink`、`acoustic_chunk_frames`。它们不是新的兼容性承诺。

旧版 `from sakuratts.nvidia import NVIDIAEngine, write_wav` 继续可用。其他内部模块已迁移；新的业务代码应使用包顶层 API。
