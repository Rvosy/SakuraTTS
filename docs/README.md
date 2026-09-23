# 文档

首次部署从[快速开始](quickstart.md)进入；接入桌宠或其他 HTTP 客户端先看 [API V2 使用说明](api-v2-guide.md)，其中列出当前支持的调用、全部参数和错误处理。服务配置与生命周期见 [HTTP API](http-api.md)，在 Python 中直接使用自有 Engine 则看 [Python API](python-api.md)。

| 需求 | 文档 |
| --- | --- |
| 安装与首次生成 | [快速开始](quickstart.md) |
| 桌宠的 LLM 并行接入、休眠与预热 | [后台驻留与提前唤醒](background-runtime.md) |
| 服务启动、状态与生命周期接口 | [HTTP API](http-api.md) |
| HTTP 客户端接入与当前能力表 | [API V2 使用说明](api-v2-guide.md) |
| 英文与日英混合资源准备 | [英文前端](english-frontend.md) |
| Windows 模型与运行资源准备 | [Windows 指南](setup-windows-nvidia.md)、[独立 ORT 组件](setup-ort-worker-offline.md) |
| 精度、显存和延迟取舍 | [推理档位](inference-profiles.md) |
| 模型目录与路径校验 | [模型格式](model-format.md) |
| 支持范围与已知限制 | [兼容矩阵](specs/compatibility-matrix.md) |
| 行为约束 | [推理契约](specs/inference-contract.md)、[基准协议](specs/benchmark-protocol.md) |
| 修改代码与运行测试 | [开发指南](development.md)、[架构](architecture.md) |
| 打包与安装发布产物 | [预览版说明](preview-release.md) |
| 整合包使用、离线构建与验收 | [Windows / NVIDIA 整合包](portable-bundle.md) |
| 后续开发顺序 | [路线图](roadmap.md) |

历史实验、失败记录和原始测量保存在 Git 仓库的 [research](https://github.com/Rvosy/SakuraTTS/tree/main/research) 中，不随 wheel 或源码包发布。Mac 实验入口见[日文运行记录](japanese-runtime.md)，不属于当前公共 Engine 的支持范围。
