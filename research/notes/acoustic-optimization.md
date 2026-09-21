# 声学优化结论

声学分块通过已验证的边界和重叠依赖组织完整波形，不构成流式输出。调用者需要对应的 split package，并显式指定实验开关。

FP16 包对官方 FP32 的严格数值验收仍有失败。工程误差筛查和辅助 ASR 不能代替音色与自然度复听。

详见[FP16 记录](../experiments/2026-09-20-windows-acoustic-fp16.md)、[分块边界](../experiments/2026-09-20-windows-vocoder-chunks.md)和[公开入口实测](../experiments/2026-09-20-windows-vocoder-public.md)。
