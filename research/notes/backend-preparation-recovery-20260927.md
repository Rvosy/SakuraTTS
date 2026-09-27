# CPU / AMD 模型准备与切换恢复验证

日期：2026-09-27。基线提交：`0df5ac2`；验证对象为后续工作区改动。设备为 Ryzen 7 7840HS / Radeon 780M，模型为 N.A.V.I V2ProPlus，固定 seed 为 1234。[保存的结果](../experiments/data/2026-09-27-backend-preparation-recovery.json)包含原始模型身份、完整合成报告、WAV 哈希和 AMD 执行证据。

## 原始模型准备

通过公共 `convert()` 从原始 GPT / SoVITS 权重及参考音频分别准备 CPU 与 DirectML 模型。CPU 生成 INT8 GPT 和 FP32 声学；AMD 生成 FP16 GPT、容量 1280 的静态 Decode 和全图 FP16 声学，随后完成设备执行检查。两次转换均成功发布独立模型目录，未调用仓库外部的研究脚本。

准备解释器使用 Python 3.12.14 与 CPU ORT 1.30.0；AMD 声学执行检查使用主解释器的 DirectML ORT 1.24.4。两种环境的 `pip check` 均通过。AMD 保存的四组声学输入通过有限输出、重复性、I/O 和 GPU 执行检查；profile 观察到 FP16 Conv，未观察到 CPU 神经计算事件。原有工程误差筛查仍为失败，人工音质未验收。

## 生成与失败恢复

新转换的两份模型各完成一次短句、一次长句，结果均为 `completed`。CPU 报告 GPT 为 INT8、声学为 FP32，均使用 CPU；AMD 两阶段均报告 DirectML FP16。普通推理进程均未导入 PyTorch。

随后通过 `Inference` 设置参考音频并生成短句，再调用 `set_weights()` 切换到只有基础 manifest、缺少目标 GPT sidecar 的目录。两种后端均返回具体缺失文件错误，释放候选并恢复旧模型。再次请求的 PCM 与切换前完全一致，参考音频选择保留。

测试替身另覆盖候选声学加载失败、旧模型恢复也失败、元数据错配和旧前端关闭失败。真机没有注入显存不足或设备丢失。

本次用于验收准备与恢复流程。短句包含首次权重加载，长句文本也不同于 Genie 对照；没有固定整机背景负载和重复次数，不据此给出提速比例。原性能对照继续见 [Genie 复测](genie-comparison-20260927.md)。

## 回归与安装包

- CPU ORT 1.30.0 准备环境：产品测试 501 项，500 通过、1 项 POSIX 专属测试跳过。
- 研究测试 151 项，149 通过、2 项平台或权限相关测试跳过。
- DirectML 环境的产品测试单独执行，缺少 PyTorch 的 BERT 测试无法导入；完整测试由上述准备环境通过，未为此给普通推理环境添加 PyTorch。
- 最后精简重复扫描后，声学准备、finite 准入和解释器隔离的 16 项相关测试通过。
- 使用项目固定版本 setuptools 构建 wheel 与源码包成功；从 wheel 解包后，三个新增准备入口在 `-I` 隔离模式启动成功。安装包布局旁放置不兼容的 NumPy / ORT 替身后，独立准备解释器仍能导出 CPU INT8 与 DirectML 静态 GPT，未导入宿主依赖。

本机日志与脚本位于 `results/backend-completion-20260927/`，新模型和构建产物位于 `outputs/backend-completion-20260927/`。这些目录不纳入源码分发。

## 未验证范围

未进行 AMD 独显、多张物理显卡、CUDA 或 Apple 真机复验，也未完成更多权重、人工听音、ASR 和长期负载验收。CPU 与 AMD 仍分别发布模型目录，尚未合并为自动选择多后端产物的单个模型描述；CPU / AMD 整合包也未在本次制作。
