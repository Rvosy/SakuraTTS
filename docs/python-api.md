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

`sakuratts.start_server(model, host="127.0.0.1", port=9880)` 启动 HTTP 服务，默认 `runtime_mode="direct"`，仍在启动时加载配置的模型。需要休眠和提前唤醒时显式选择控制模式：

```python
from sakuratts import start_server

start_server("models/sakura", runtime_mode="managed", idle_sleep_seconds=60,
             wake_timeout_seconds=120, operation_timeout_seconds=300)
```

三个秒数必须是有限正数，只在 `managed` 模式使用。该模式提供 `/runtime`、`/runtime/wake` 和 `/runtime/sleep`；普通 `/tts` 也会自动唤醒。它不会改变进程内 `Engine.load` 或 `Engine.close()` 的生命周期，准备就绪也不等于完成首次执行预热。

HTTP 的总体 `policy="staged"` 仅允许显式启用的 `managed` 模式。可读取 `examples/minimum-vram.json`，通过 `start_server("MODEL", experimental=options, runtime_mode="managed")` 使用 H 档，模型须匹配 FP16 chunk256 声学包。此时 `/runtime.preparation="runtime_init"`，`awake` 表示推理进程、前端和配置已准备；GPU 权重在执行时交替加载，`model_loaded` 始终为 `false`。其他档位仍为 `preparation="model_load"`，加载步骤完成后 `model_loaded=true`。`direct` 的加载时机和策略限制保持不变。

`synthesize` 可指定 `reference`、`seed`、`language`（`ja` / `all_ja`）、`split_method`（`cut0` 至 `cut5`，默认 `cut0`）、`top_k`、`temperature`、`repetition_penalty`、`early_stop_num` 和 `cancel_requested`。完整签名与默认值见 [Engine.synthesize](../src/sakuratts/engine.py)。NumPy 与官方 Torch 使用不同随机数实现。

`tts` CLI 可通过 `--split-method cut5` 按标点分句，默认仍为 `cut0`。当前只提供 FP32、FP16 标准、FP16 低显存、FP16 极限[四档配置](inference-profiles.md)，分句作为独立选项。历史实验和实测范围见[低显存报告](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/low-vram-20260921.md)。

低级 `Engine.synthesize` 默认 `split_bucket=False`，保留原文片段的推理顺序。显式设为 `True` 时，按规范化文本长度稳定排序推理，再恢复完整 PCM 和片段报告的原文顺序；报告用 `execution_order` 记录实际顺序。HTTP 沿用原版默认 `True`。传入 `on_fragment` 时关闭分桶，以便逐片按原文输出；排序会改变各句的随机数消耗顺序。

一个 Engine 同时只处理一个请求，重入会抛出 `BusyError`。使用 `with` 或显式 `close()` 释放资源；关闭后不能继续合成。失败会沿原链路释放请求状态。宿主应在同一个线程中拥有、调用和关闭引擎，HTTP 服务已执行这一约束。

CUDA 引擎默认在每片声学计算结束后收缩 ORT 显存池，释放不再使用的工作区，避免长文本将后续空闲占用一直抬高。权重和仍在使用的分配继续保留；收缩影响请求后的空闲分配，活动峰值仍由计算过程决定。FP32 精度和采样规则保持不变。

需要对照旧策略时，可用 `Engine.load(path, experimental={"acoustic_arena_shrink": False})` 关闭收缩。CLI 的 `tts`、`serve`、`benchmark` 可通过 `--experimental` 读取同样的 JSON；旧 `synthesize` 命令可用 `--no-acoustic-arena-shrink`。实际策略记录在合成报告的 `acoustic_arena_shrink` 字段中。内部 `ORTSoVITS` / `ORTProcessSoVITS` 的直接调用仍需显式选择收缩，研究脚本应记录自己的选项。

实验后端选项通过 `Engine.load(path, experimental={...})` 显式传入。支持的键与默认值由 [CUDA 后端的 create_runtime](../src/sakuratts/backends/cuda/__init__.py) 和 [NVIDIAEngine](../src/sakuratts/backends/cuda/engine.py) 定义；常用组合见[推理档位](inference-profiles.md)。

`acoustic_session_policy` 默认 `"resident"`，保留 latent 和 vocoder 两个 Session。设为 `"staged"` 时，先加载 latent Session 并取回完整 CPU latent，再释放它、加载一次 vocoder Session 处理该片段的所有块，最后释放 vocoder。它只支持已经验证的分块声学包，必须同时指定 `acoustic_chunk_frames` 并开启 `acoustic_arena_shrink`；每个片段会重建 Session，增加加载时间。模型图、块长、完整上下文和输入噪声保持不变。此选项控制声学内部的驻留方式，与控制 GPT／声学模型驻留的 `policy` 分开设置。私有 worker 的 `session_initialization="deferred"` 表示模型包与进程已准备，CUDA Session 将在执行时创建并检查。

`gpt_prefill_query_chunk_size` 默认 `0`，使用完整 Prefill 注意力矩阵；正整数指定每次计算的 query 行数，例如 `128`。每块仍读取全部 key，保留文本双向、音频因果的注意力关系。分块减少长前缀的临时分数矩阵，但 GEMM 形状变化可能改变浮点舍入和后续采样结果，需要单独验收；它不限制整条推理链的显存，也不改变 KV 容量。实际选项写入音频报告同名字段。

旧版 `from sakuratts.nvidia import NVIDIAEngine, write_wav` 继续可用。其他内部模块已迁移；新的业务代码应使用包顶层 API。
