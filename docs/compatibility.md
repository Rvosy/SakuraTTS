# 当前兼容范围

公共 Engine、CLI 和服务接入 Windows / NVIDIA 的日文 V2ProPlus 路径，HTTP 提供完整 WAV / RAW / OGG / AAC，以及按句流式返回。默认 GPT / 声学 FP32、baseline attention，不自动启用实验优化。HTTP 沿用原版的 `seed=-1`、`text_split_method=cut5`；历史低级 Python 接口仍默认 `seed=1234`、`cut0`。

具体接口差异与缺失功能见 [HTTP 兼容清单](http-api.md)。声学解释器、经典 pyopenjtalk 及字典版本属于兼容条件，见[推理契约](specs/inference-contract.md)和[兼容矩阵](specs/compatibility-matrix.md)。结构调整不会扩大已验证的模型、设备或语言范围。

MLX 和中文模块供历史回放及后续开发使用，当前 `model.json` 不接受选择它们。其他模型家族、中文整链、语义 token 流式模式、批量并行计算、最低显存与宿主集成没有正式支持承诺。

FP16 声学仍有严格数值失败。人工音色、自然度和部分长句完整性未完成验收，不能把哈希检查、ASR 或接口测试当作音质通过。见[参考对齐总结](research/reference-parity.md)。
