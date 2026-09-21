# 内存与显存结论

模型常驻、请求状态释放和分阶段加载分别影响下一次请求的耗时。ORT 工作进程、CUDA arena 收缩和声学分块改变不同的生命周期或工作区，不能只按包大小推断运行成本。

当前 CUDA 引擎默认回收声学 arena 的空闲区域。相同 84 次混合请求后，SakuraTTS FP32 的空闲显存比原版 API FP16 低 25.0%，比自身旧策略低 48.7%；测量条件、峰值、延迟和输出误差见[原版显存对照报告](inference-memory-comparison.md)。

Windows WDDM 与全卡采样可能包含其他进程；资源采样轮与计时轮分开，不能将全卡峰值增量写成独占显存或最低硬件要求。

证据见[WDDM 测量](../experiments/2026-09-20-windows-wddm-memory.md)、[声学 arena](../experiments/2026-09-20-windows-acoustic-arena.md)及[运行环境清单](../experiments/2026-09-20-windows-runtime-inventory.md)。
