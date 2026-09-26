# 模型与功能验证矩阵

本页集中记录产品支持范围与验证依据。请求字段和限制见 [API V2 使用说明](../api-v2-guide.md)，默认值由 [SpeechRequest](../../src/sakuratts/server.py) 与 [Engine.synthesize](../../src/sakuratts/engine.py) 定义。

## 实现范围

| 项目 | 公共入口的范围 |
| --- | --- |
| 设备与模型 | Windows CUDA、CPU、DirectML，V2ProPlus；CPU 的 GPT 与声学均在 CPU，DirectML 的 GPT Transformer 与声学均使用 GPU。Apple MLX 提供实验性原生 V2Pro 适配器，限预制模型与参考包 |
| 语言 | `ja`、`all_ja`、`en`、`auto`；英文段需要[英文依赖与资源包](../english-frontend.md)，`auto` 检测到其他语言时明确报错 |
| HTTP | CPU / CUDA / DirectML 支持 GPT-SoVITS 固定版本 API V2；单活动请求，完整音频及按句流式；MLX 暂不接入 |
| 精度 | CPU 仅提供并默认选择 INT8 GPT + FP32 声学；DirectML 仅提供并默认选择 FP16，KV 容量 1280。CUDA / MLX 保持原有档位，资源与验证要求见[推理档位](../inference-profiles.md) |
| 生命周期 | 默认 `direct`；可选 `managed` 提供提前唤醒、保活与空闲休眠 |
| 分发 | wheel、源码包和可离线构建的 Windows 整合包；构建、验收与发布分别记录 |

后端实现表在 [backends._IMPLEMENTATIONS](../../src/sakuratts/backends/__init__.py)，语言模式在 [frontend.profiles](../../src/sakuratts/frontend/profiles.py)。`Model` 允许读取结构有效的其他语言或后端声明；执行时仍由已实现的适配器判断是否可用。中文研究模块未接入公共 Engine。MLX 的模型限制和安装方式见 [Apple 指南](../apple.md)。

## 已有验证与未验收项

以下是保存记录的索引，详细条件和数值由原报告维护。

| 范围 | 证据 | 仍需验证 |
| --- | --- | --- |
| Windows 日文整链 | [RTX 5060 / Sakura V2ProPlus 实测](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-nvidia-backend.md)：原文到 WAV、长文、五参考及异常恢复 | 其他 GPU、其他权重、人工音色与自然度 |
| 当前 CPU / DirectML | [Genie 前后复测与试听](../../research/notes/genie-comparison-20260927.md)：CPU INT8、8 线程声学；AMD FP16、容量 1280；N.A.V.I 短句、长句与进程树资源 | 更多硬件与权重、人工听音、ASR、长期睡醒及共存负载；CPU 性能取自 DirectML 1.24.4 的 CPU provider，不代表独立 CPU ORT 1.30.0 的速度 |
| DirectML 显卡选择 | [设备转发与初始化失败测试](../../tests/test_directml_devices.py)、[配置覆盖与重新加载](../../tests/test_cpu_runtime.py)、[Session KV 分配与回收](../../tests/test_directml_static_gpt.py)：用替身覆盖非零适配器编号；[780M 回放与卸载占用](../../research/notes/directml-host-overhead-20260927.md)保留本轮实机证据 | AMD 独显及多张物理显卡上的真实执行、KV 归属、内存占用与音频结果尚未验收；非零编号测试不代表独显实测 |
| CPU / AMD 候选历史 | [独立精度与容量实验](../../research/notes/cpu-amd-precision-listening-20260927.md)、[首轮设备实测](../../research/notes/cpu-directml-780m-20260927.md)：保留未选候选、实际 GPU 算子、误差与 HTTP 睡醒 | 历史候选不再作为公开配置；降精度可能改变采样、语速与完整性 |
| CPU GPT / AMD 混合设备历史 | [完整请求对照](../../research/notes/cpu-amd-performance-20260927.md)、[GPT 线程与数值](../../research/notes/cpu-gpt-ort-20260927.md)、[声学精度筛查](../../research/notes/directml-mixed-vocoder-780m-20260927.md) | 严格 FP32 波形等价与音质是不同验收项；历史设备组合不能代替当前独立 GPU 路径 |
| Apple MLX 适配器 | [公共入口测试](../../tests/test_mlx_runtime.py)：使用计算替身检查装配、取消、错峰、精度拒绝与回收；组件的历史真机结果见本页末尾证据汇总 | 当前公共入口尚未在 Apple 真机复验；FP16、V2ProPlus、统一转换与 HTTP 新参考未实现 |
| FP16 与显存策略 | [低显存对照](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/low-vram-20260921.md)、[声学 Session 错峰](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/acoustic-session-staging-20260921.md) | FP16 对官方 FP32 的严格波形容差仍有失败，长句完整性和听感未完成验收 |
| HTTP 字段与调度 | [固定上游契约测试](../../tests/test_upstream_api_contract.py)及服务测试覆盖字段、错误、按句流式和分桶编排 | 分桶变更后的 CUDA 音频复验、实际客户端播放与取消 |
| 后台控制模式 | [首次睡醒验证](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-runtime-20260922.md)、[资源精简与 100 次睡醒](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-refinement-20260922.md) | H 档长期反复睡醒、数小时待机与更多设备 |
| 完整整合包 | [2026-09-22 构建与验收](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/portable-compact-20260922.md)：原始权重、新参考、缓存、搬迁和解压校验 | 第二台干净机器、最低驱动、峰值显存、音质及分发材料 |
| 内容检查 | [24 条 Windows 音频的 ASR 记录](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-asr.md) | 报告中的内容疑点与人工复听 |

CUDA 实测设备为 RTX 5060 8 GB；CPU / DirectML 已在 Ryzen 7 7840HS / Radeon 780M 上执行 N.A.V.I V2ProPlus 日文模型，配置与限制见 [CPU / AMD 指南](../cpu-amd.md)。Apple 公共入口仍待真机验收；其他模型家族、语义 Token 流式、并行批量与宿主集成尚未完成。CPU / DirectML / MLX 目前通过源码安装，尚未提供对应整合包。

## 历史证据与更新方式

Mac / MPS / MLX 的固定样例、Lite 语音问题、数值超差和逐项实验进度保存在[2026-09-19 至 20 日证据汇总](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/compatibility-evidence-20260920.md)。这份历史记录保留当时的通过项、失败项和试听反馈，Windows 的后续结果单列在上表。

新增结果应链接带设备、模型、源码与配置身份的运行报告。自动检查、ASR 和人工试听分别记录；未执行的检查标为未验证。具体数值判定和资源计量遵循[基准协议](benchmark-protocol.md)，行为要求见[推理契约](inference-contract.md)。
