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

`tts` CLI 可通过 `--split-method cut5` 按标点分句，默认仍为 `cut0`。长文本的低显存实验配置与实测范围见[低显存报告](research/low-vram-20260921.md)。

一个 Engine 同时只处理一个请求，重入会抛出 `BusyError`。使用 `with` 或显式 `close()` 释放资源；关闭后不能继续合成。失败会沿原链路释放请求状态。宿主应在同一个线程中拥有、调用和关闭引擎，HTTP 服务已执行这一约束。

CUDA 引擎默认在每片声学计算结束后收缩 ORT 显存池，释放不再使用的工作区，避免长文本将后续空闲占用一直抬高。权重和仍在使用的分配继续保留；这不保证显存回到加载时的数值，也不会降低计算过程所需的峰值。FP32 精度和采样规则保持不变。

需要对照旧策略时，可用 `Engine.load(path, experimental={"acoustic_arena_shrink": False})` 关闭收缩。CLI 的 `tts`、`serve`、`benchmark` 可通过 `--experimental` 读取同样的 JSON；旧 `synthesize` 命令可用 `--no-acoustic-arena-shrink`。实际策略记录在合成报告的 `acoustic_arena_shrink` 字段中。内部 `ORTSoVITS` / `ORTProcessSoVITS` 的直接调用仍需显式选择收缩，研究脚本应记录自己的选项。

实验后端选项通过 `Engine.load(path, experimental={...})` 显式传入。支持的键为 `policy`、`use_graph`、`capacity`、`gpt_precision`、`gpt_attention`、`gpt_attention_chunk_size`、`gpt_prefill_query_chunk_size`、`allow_experimental_acoustic_fp16`、`acoustic_arena_shrink`、`acoustic_chunk_frames`。它们不是新的兼容性承诺。

`gpt_prefill_query_chunk_size` 默认 `0`，使用完整 Prefill 注意力矩阵；正整数指定每次计算的 query 行数，例如 `128`。每块仍读取全部 key，保留文本双向、音频因果的注意力关系。分块减少长前缀的临时分数矩阵，但 GEMM 形状变化可能改变浮点舍入和后续采样结果，需要单独验收；它不限制整条推理链的显存，也不改变 KV 容量。实际选项写入音频报告同名字段。

旧版 `from sakuratts.nvidia import NVIDIAEngine, write_wav` 继续可用。其他内部模块已迁移；新的业务代码应使用包顶层 API。
