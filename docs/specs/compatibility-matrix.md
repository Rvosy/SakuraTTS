# 模型与功能验证矩阵

本页集中记录产品支持范围与验证依据。请求字段和限制见 [API V2 使用说明](../api-v2-guide.md)，默认值由 [SpeechRequest](../../src/sakuratts/server.py) 与 [Engine.synthesize](../../src/sakuratts/engine.py) 定义。

## 实现范围

| 项目 | 公共入口的范围 |
| --- | --- |
| 设备与模型 | Windows / NVIDIA CUDA，V2ProPlus |
| 语言 | `ja`、`all_ja`、`en`、`auto`；英文段需要[英文依赖与资源包](../english-frontend.md)，`auto` 检测到其他语言时明确报错 |
| HTTP | GPT-SoVITS 固定版本 API V2；单活动请求，完整音频及按句流式 |
| 精度 | 默认 GPT / 声学 FP32；FP16 为显式实验档位，选择条件见[推理档位](../inference-profiles.md) |
| 生命周期 | 默认 `direct`；可选 `managed` 提供提前唤醒、保活与空闲休眠 |
| 分发 | wheel、源码包和可离线构建的 Windows 整合包；构建、验收与发布分别记录 |

后端实现表在 [backends._IMPLEMENTATIONS](../../src/sakuratts/backends/__init__.py)，语言模式在 [frontend.profiles](../../src/sakuratts/frontend/profiles.py)。`Model` 允许读取结构有效的其他语言或后端声明；执行时仍由已实现的适配器判断是否可用。MLX 与中文研究模块未接入公共 Engine。

## 已有验证与未验收项

以下是保存记录的索引，详细条件和数值由原报告维护。

| 范围 | 证据 | 仍需验证 |
| --- | --- | --- |
| Windows 日文整链 | [RTX 5060 / Sakura V2ProPlus 实测](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-nvidia-backend.md)：原文到 WAV、长文、五参考及异常恢复 | 其他 GPU、其他权重、人工音色与自然度 |
| FP16 与显存策略 | [低显存对照](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/low-vram-20260921.md)、[声学 Session 错峰](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/acoustic-session-staging-20260921.md) | FP16 对官方 FP32 的严格波形容差仍有失败，长句完整性和听感未完成验收 |
| HTTP 字段与调度 | [固定上游契约测试](../../tests/test_upstream_api_contract.py)及服务测试覆盖字段、错误、按句流式和分桶编排 | 分桶变更后的 CUDA 音频复验、实际客户端播放与取消 |
| 后台控制模式 | [首次睡醒验证](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-runtime-20260922.md)、[资源精简与 100 次睡醒](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-refinement-20260922.md) | H 档长期反复睡醒、数小时待机与更多设备 |
| 完整整合包 | [2026-09-22 构建与验收](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/portable-compact-20260922.md)：原始权重、新参考、缓存、搬迁和解压校验 | 第二台干净机器、最低驱动、峰值显存、音质及分发材料 |
| 内容检查 | [24 条 Windows 音频的 ASR 记录](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-asr.md) | 报告中的内容疑点与人工复听 |

硬件实测目前集中于 RTX 5060 8 GB。其他模型家族、CPU / AMD / Apple 公共后端、语义 Token 流式、并行批量与宿主集成尚未完成。

## 历史证据与更新方式

Mac / MPS / MLX 的固定样例、Lite 语音问题、数值超差和逐项实验进度保存在[2026-09-19 至 20 日证据汇总](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/compatibility-evidence-20260920.md)。这份历史记录保留当时的通过项、失败项和试听反馈，Windows 的后续结果单列在上表。

新增结果应链接带设备、模型、源码与配置身份的运行报告。自动检查、ASR 和人工试听分别记录；未执行的检查标为未验证。具体数值判定和资源计量遵循[基准协议](benchmark-protocol.md)，行为要求见[推理契约](inference-contract.md)。
