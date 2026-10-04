# 模型与功能验证矩阵

本页集中记录产品支持范围与验证依据。请求字段和限制见 [API V2 使用说明](../api-v2-guide.md)，默认值由 [SpeechRequest](../../sakuratts/server.py) 与 [Engine.synthesize](../../sakuratts/engine.py) 定义。

## 完整兼容目标与当前差距

兼容基准为 GPT-SoVITS [`48b1a016`](https://github.com/RVC-Boss/GPT-SoVITS/tree/48b1a0169a28582a8984402f82cf438d3bfa6aca)。目标覆盖该版本的模型、语言、配置、调用入口与行为。版本升级时先核对差异，再更新基准和回归样例；目录相似、字段齐全或单句出声都不能代替兼容验收。

| 维度 | 当前差距与验收要求 |
| --- | --- |
| 模型与资源 | Windows 公共转换和推理集中在 V2ProPlus，MLX 为原生 V2Pro；V1、V2、V3、V4 及其他组合仍须逐一完成权重识别、加载、转换、参考编码和音频对照 |
| 语言 | 日文、英文及日英混合已接入；中文组件尚未贯通公共入口，粤语、韩语及其混合模式待补齐；同时验证目标文本与参考转写 |
| 推理参数 | 多参考、无转写、语速、真正的批量/并行、语义 Token 流式 2/3 与超采样仍有缺口；Top-p 的 HTTP 字段与采样函数已存在，但生成循环仍要求 `top_p=1`，不能标成已贯通 |
| 配置 | 当前读取部分上游 YAML 字段；上游 `is_half`、模型版本分支、设备切换与全部资源字段尚未完全映射；不能用 SakuraTTS 的独立 profile 宣称上游配置已兼容 |
| 调用入口 | 当前 HTTP 为 API V2 子集；根目录 `api.py` 启动同一 V2 服务，尚未实现旧版协议。原版 Python 类/导入接口、Gradio、WebUI 和其他启动流程仍需适配与回归 |
| 输出与错误 | 逐项对照分句、顺序、采样和停止规则、PCM、流式时机、错误响应及取消；同种子不保证不同计算库逐样本一致，数值和语音质量分别验收 |
| 训练与数据工具联动 | 当前交付推理与准备组件；训练、标注、数据制作所用的路径、配置、产物与调用流程尚未全面核对。按独立组件确定交付方式，不把训练依赖直接并入日常推理环境 |
| 分发与设备 | 每种设备分别验证首次准备、缓存、搬迁、切换失败恢复和退出回收；已测的一种设备或模型不能代表所有组合 |

现有默认值和未实现项的具体错误保持当前行为，新增兼容能力通过对应回归验收后更新下表。重构只调整代码归属，不提升任何未验证项的状态。

## 实现范围

| 项目 | 公共入口的范围 |
| --- | --- |
| 设备与模型 | Windows CUDA、CPU、DirectML，V2ProPlus；CPU 的 GPT 与声学均在 CPU，DirectML 的 GPT Transformer 与声学均使用 GPU。Apple MLX 提供实验性原生 V2Pro 适配器，支持原始权重转换与新参考准备 |
| 语言 | `ja`、`all_ja`、`en`、`auto`；英文段需要[英文依赖与资源包](../english-frontend.md)，`auto` 检测到其他语言时明确报错 |
| HTTP | CPU / CUDA / DirectML / MLX 共用 GPT-SoVITS 固定版本 API V2；单活动请求，完整音频及按句流式 |
| 精度 | CPU 默认选择 INT8 GPT + FP32 声学；DirectML 默认选择 FP16，KV 容量 1280。CUDA / MLX 保持原有档位，资源与验证要求见[推理档位](../inference-profiles.md) |
| 生命周期 | 默认 `direct`；可选 `managed` 提供提前唤醒、保活与空闲休眠 |
| 分发 | wheel、源码包和可离线构建的 Windows / Apple silicon 整合包；Linux 当前采用 [CPU 源码安装](../linux.md)，整合包尚未接入；构建、验收与发布分别记录 |

后端实现表在 [backends._IMPLEMENTATIONS](../../sakuratts/backends/__init__.py)，语言模式在 [frontend.profiles](../../sakuratts/text/profiles.py)。`Model` 允许读取结构有效的其他语言或后端声明；执行时仍由已实现的适配器判断是否可用。中文研究模块未接入公共 Engine。MLX 的模型限制和安装方式见 [Apple 指南](../apple.md)。

## 已有验证与未验收项

以下是保存记录的索引，详细条件和数值由原报告维护。

| 范围 | 证据 | 仍需验证 |
| --- | --- | --- |
| Windows 日文整链 | [RTX 5060 / Sakura V2ProPlus 实测](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-nvidia-backend.md)：原文到 WAV、长文、五参考及异常恢复 | 其他 GPU、其他权重、人工音色与自然度 |
| 当前 CPU / DirectML | [Genie 前后复测与试听](../../research/notes/genie-comparison-20260927.md)：CPU INT8、8 线程声学；AMD FP16、容量 1280；N.A.V.I 短句、长句与进程树资源 | 更多硬件与权重、人工听音、ASR、长期睡醒及共存负载；CPU 性能取自 DirectML 1.24.4 的 CPU provider，不代表独立 CPU ORT 1.30.0 的速度 |
| DirectML 显卡选择 | [设备转发与初始化失败测试](../../tests/test_directml_devices.py)、[配置覆盖与重新加载](../../tests/test_cpu_runtime.py)、[Session KV 分配与回收](../../tests/test_directml_static_gpt.py)：用替身覆盖非零适配器编号；[780M 回放与卸载占用](../../research/notes/directml-host-overhead-20260927.md)保留本轮实机证据 | AMD 独显及多张物理显卡上的真实执行、KV 归属、内存占用与音频结果尚未验收；非零编号测试不代表独显实测 |
| CPU / AMD 准备与恢复 | [原始权重到生成及切换恢复](../../research/notes/backend-preparation-recovery-20260927.md)：CPU ORT 1.30.0、DirectML ORT 1.24.4；公开转换自动补齐资源，缺少 GPT sidecar 的切换失败后旧模型恢复，PCM 一致 | AMD 独显、多卡、真机显存不足和设备丢失；CPU / AMD 仍分别发布模型目录，未制作对应整合包 |
| CPU / AMD 候选历史 | [独立精度与容量实验](../../research/notes/cpu-amd-precision-listening-20260927.md)、[首轮设备实测](../../research/notes/cpu-directml-780m-20260927.md)：保留未选候选、实际 GPU 算子、误差与 HTTP 睡醒 | 默认预设保持当前配置，显式选项可覆盖；降精度可能改变采样、语速与完整性 |
| CPU GPT / AMD 混合设备历史 | [完整请求对照](../../research/notes/cpu-amd-performance-20260927.md)、[GPT 线程与数值](../../research/notes/cpu-gpt-ort-20260927.md)、[声学精度筛查](../../research/notes/directml-mixed-vocoder-780m-20260927.md) | 严格 FP32 波形等价与音质是不同验收项；历史设备组合不能代替当前独立 GPU 路径 |
| Apple MLX 适配器与完整包 | [M4 原始权重、HTTP 与生命周期验收](../../research/notes/macos-portable-20261004.md)；[公共入口测试](../../tests/test_mlx_runtime.py)覆盖装配、取消、切换失败恢复、错峰与精度拒绝 | macOS 14 / 15、其他 Apple 芯片、长期运行与人工听音；FP16、V2ProPlus 尚未实现 |
| Windows / Linux 跨平台回归 | [2026-10-04 检查记录](../../research/notes/cross-platform-portability-20261004.md)：Windows 10 x64 与 Linux x86_64 运行依赖、CPU ONNX 运算和产品测试；Windows Unicode 启动与包组件检查 | Linux 完整模型、原始权重准备与整合包；当前 Windows 包的完整模型、新参考与更多设备；小图运算不代表语音验收 |
| FP16 与显存策略 | [低显存对照](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/low-vram-20260921.md)、[声学 Session 错峰](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/acoustic-session-staging-20260921.md) | FP16 对官方 FP32 的严格波形容差仍有失败，长句完整性和听感未完成验收 |
| HTTP 字段与调度 | [服务测试](../../tests/test_server.py)覆盖音频响应、错误、按句流式和分桶编排 | 分桶变更后的 CUDA 音频复验、实际客户端播放与取消 |
| 后台控制模式 | [首次睡醒验证](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-runtime-20260922.md)、[资源精简与 100 次睡醒](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-refinement-20260922.md) | H 档长期反复睡醒、数小时待机与更多设备 |
| 完整整合包 | [2026-09-22 构建与验收](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/portable-compact-20260922.md)：原始权重、新参考、缓存、搬迁和解压校验 | 第二台干净机器、最低驱动、峰值显存、音质及分发材料 |
| 内容检查 | [24 条 Windows 音频的 ASR 记录](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-asr.md) | 报告中的内容疑点与人工复听 |

CUDA 实测设备为 RTX 5060 8 GB；CPU / DirectML 已在 Ryzen 7 7840HS / Radeon 780M 上执行 N.A.V.I V2ProPlus 日文模型，配置与限制见 [CPU / AMD 指南](../cpu-amd.md)。Apple M4 的当前验证见上表；其他模型家族、语义 Token 流式、并行批量与宿主集成尚未完成。CPU / DirectML 的 Windows 共包构建入口见[整合包指南](../portable-bundle.md)，代码支持与最终包验收分别记录；MLX 可通过源码安装或构建 [Mac 整合包](../portable-macos.md)。

## 历史证据与更新方式

Mac / MPS / MLX 的固定样例、Lite 语音问题、数值超差和逐项实验进度保存在[2026-09-19 至 20 日证据汇总](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/compatibility-evidence-20260920.md)。这份历史记录保留当时的通过项、失败项和试听反馈，Windows 的后续结果单列在上表。

新增结果应链接带设备、模型、源码与配置身份的运行报告。自动检查、ASR 和人工试听分别记录；未执行的检查标为未验证。具体数值判定和资源计量遵循[基准协议](benchmark-protocol.md)，行为要求见[推理契约](inference-contract.md)。
