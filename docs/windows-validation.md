# Windows 验证入口

Windows / NVIDIA 的公共用法见[部署指南](setup-windows-nvidia.md)，已有实测与剩余验收项见[兼容矩阵](specs/compatibility-matrix.md)。运行前准备对应模型、参考和独立 ORT 组件；`doctor` 的依赖检查与实际 CUDA 合成分别执行。

完整请求对照从 `python research/compare_official.py --help` 进入，需要包含 `research/` 的 Git checkout。固定语料在 [benchmarks/cases/](../benchmarks/cases)，比较方式和证据要求见[基准协议](specs/benchmark-protocol.md)。

CUDA 后端接通前的四例 NumPy 便携验收格式、比较容差及 Mac 已知失败保存在[2026-09-20 接续记录](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/windows-handoff-20260920.md)。它用于复现那组固定模型的诊断；后续 Windows 运行结果见[Windows 后端实测](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-nvidia-backend.md)。
