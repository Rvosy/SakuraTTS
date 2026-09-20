# Windows CUDA 结论

已接通 CuPy GPT、独立 ORT CUDA SoVITS 和日文前端的完整 WAV 请求。主要实测为 RTX 5060 8 GB、Sakura V2ProPlus，公共默认仍为 FP32 / baseline。

split-KV、GPT FP16、声学 FP16 和分块各自保留显式选择。速度和显存数据只对各自的模型、请求长度、精度、工作量及采样方法有效。

详见[Windows 后端记录](../../research/experiments/2026-09-20-windows-nvidia-backend.md)、[split-KV](../../research/experiments/2026-09-20-windows-split-kv.md)和[GPT FP16](../../research/experiments/2026-09-20-windows-gpt-fp16.md)。
