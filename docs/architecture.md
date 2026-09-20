# 架构与目录

公共入口是 Engine、Model 和 Audio。CLI 和 HTTP 调用相同 Engine，Engine 将请求交给现有 CUDA 编排；GPT、采样、声学和随机数消耗顺序没有因目录调整而改变。

| 位置 | 职责 |
| --- | --- |
| `engine.py` / `model.py` | 公共生命周期、输出和模型目录 |
| `converter.py` | 模型转换与独立参考准备，成功后发布目录 |
| `reference.py` | 将原版音频路径和文本解析成条件，透明复用音频缓存 |
| `frontend/` | 文本分段、日文和中文前端；派生代码保留许可边界 |
| `backends/cuda/` | CuPy GPT 与 Windows 推理编排 |
| `backends/onnx/` | SoVITS 图、分块验证及独立进程适配 |
| `backends/mlx/` | Mac 实验实现，未接入公共 Engine |
| `_internal/` | 采样、生成、参考、权重、诊断和私有 IPC |
| `_internal/conversion/` | 仅离线转换时使用的导出与参考准备 |

声学和经典前端 worker 通过 `_internal/worker.py` 只绑定指定的 SakuraTTS 包，不向私有 Python 暴露主环境整个 `site-packages`。包顶层使用延迟导入，兼容私有 Python 3.9 的加载过程。

HTTP 以原版 api_v2 的字段和默认值接入，在单一线程中创建、调用、切换和关闭引擎，不维护另一套采样流程。参考条件在请求时解析；模型目录不再要求绑定参考。按句流式在每片完成后发送 PCM，通过有界队列施加背压，客户端断开后在计算边界取消。模型内存策略和实验参数仍由后端负责；没有引入插件注册层或自动后端回退。

`benchmarks/` 提供常用入口。`research/` 保存历史工具、报告和原始证据，不进入运行 wheel，源码包仍保留。保存过的失败项不能因归档而改写为通过。
