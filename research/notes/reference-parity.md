# 参考对齐与已知失败

参考身份包括 GPT / SoVITS 检查点、官方源码、参考音频、转写和语言。经典 pyopenjtalk 与 pyopenjtalk-plus 不能任意互换；目录迁移必须保留原 profile。

固定历史 logits、中间张量、相同随机输入回放、自然生成和人工复听回答不同问题。种子相同不保证 NumPy 与 Torch 生成同一音频；ASR 识别结果也不能确认音色相同。

已保存的声学 FP16 严格数值失败及待复听项继续有效。重构只调整入口和组织，不宣称解决这些问题。详见[参考绑定](../experiments/2026-09-20-bound-reference-runtime.md)、[前端隔离](../experiments/2026-09-20-frontend-process-isolation.md)、[ASR 补测](../experiments/2026-09-20-windows-asr.md)和[兼容矩阵](../../docs/specs/compatibility-matrix.md)。
