# 产品路线

SakuraTTS 提供可嵌入的 Python 引擎、命令行和 HTTP 推理服务。当前公共入口支持 Windows / NVIDIA、已验证的 V2ProPlus 日文链路和按句流式返回。支持范围以[兼容矩阵](specs/compatibility-matrix.md)为准。

后续优先完成以下工作：

- 交付 [Windows / NVIDIA 整合包](portable-bundle.md)：自带 Python 与推理依赖，处理运行资源搬迁、CUDA 环境隔离、非 ASCII 路径和实际 GPU 启动检查。
- 补齐部署验证：在第二台干净机器验证模型转换、独立 ORT 组件、安装与首次请求，明确驱动和硬件范围。
- 完成质量验收：对四档配置检查长句内容、音色和自然度，保留 FP16 相对 FP32 的数值失败记录。
- 验证服务长期运行：覆盖重复请求、失败恢复、取消、模型切换和资源释放，并测量实际宿主的播放与取消行为。
- 简化资源准备：减少首次部署的手工路径配置，保持转换依赖与日常推理依赖分开。

中文整链、其他模型家族、语义 token 流式、并行批量和 MLX 公共接入没有完成承诺。新增后端或优化要有具体使用需求及回归证据，再决定是否纳入产品。

早期研究计划保留在[历史路线记录](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/roadmap-20260920.md)，其中的阶段状态和默认参数属于当时记录。
