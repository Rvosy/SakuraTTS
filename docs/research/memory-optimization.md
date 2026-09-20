# 内存与显存结论

模型常驻、请求状态释放和分阶段加载分别影响下一次请求的耗时。ORT 工作进程、CUDA arena 收缩和声学分块改变不同的生命周期或工作区，不能只按包大小推断运行成本。

Windows WDDM 与全卡采样可能包含其他进程；资源采样轮与计时轮分开，不能将全卡峰值增量写成独占显存或最低硬件要求。

证据见[WDDM 测量](../../research/experiments/2026-09-20-windows-wddm-memory.md)、[声学 arena](../../research/experiments/2026-09-20-windows-acoustic-arena.md)及[运行环境清单](../../research/experiments/2026-09-20-windows-runtime-inventory.md)。
