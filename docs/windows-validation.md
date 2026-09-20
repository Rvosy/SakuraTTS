# Windows 接续验收

近期只推进日文，模型限“朱雀院红叶”V2Pro。Mac 保留为可执行对照，用来验证模型语义、部署数据与资源生命周期。Windows / NVIDIA 的后端、显存峰值和速度必须在实机上重新测量。

## 当前交接状态

2026-09-20 已具备转到 Windows 继续研发的条件。现有生产原型依赖 MLX / Metal，Windows 自有 CUDA 后端尚待实现；不应在 Windows 安装 `requirements/mlx-japanese.txt` 来假定引擎可运行。已完成的模型包、参考条件、文本规则和 NumPy 验收格式可继续使用。

交接时保留 `feat/reference-parity` 的最新提交。若没有推送，可从参考目录的离线 Git bundle 在新目录检出该分支；不覆盖另一台电脑已有修改。还需携带：

- GPT 与 SoVITS 的完整转换包，包含 manifest 和无损存储权重。
- `japanese-frontend-resources`、日文参考 A / B 包；词典 / Nani / Sudachi 等依赖资源另按 Windows 环境准备。
- 四例便携验收目录及原始 GPT / SoVITS 权重。开发基线另需固定官方代码、原参考音频、CNHuBERT / SV 和语言资源，普通推理无需常驻这些辅助模型。

首次操作先记录 `nvidia-smi` 的 GPU、显存和驱动，核验包身份，再建立官方 CUDA / FP32 对照和一条自有 CUDA 候选。普通 FP32 数值通过后再独立测 FP16 与其他近似，不能直接跳到最低内存配置。

自然 590 字长文仍有内容检查异常；新增 `cut2` 仅为实验入口，官方 MPS 全请求对照未完成。样例已入库，原文和所有失败记录保留，见[长文与交接记录](../research/experiments/2026-09-20-natural-long-handoff.md)。到 Windows 后应优先完成同条件官方捕获，不能仅靠换分句方式宣布问题解决。

## 可搬迁的数据包

`research/tools/export_validation_bundle.py` 从已有官方运行记录导出四条日文样例：原报告、短句、长句、标点。它不加载推理模型，不修改历史记录，也不复制原始模型。每例保存原文、规范化文本、音素、采样参数及以下数组：

| 数据 | 用途 |
|---|---|
| `phones`、`prompt`、`bert` | GPT 的完整输入；BERT 布局为 `[B,T,1024]`，日文零特征保留 |
| `tokens`、`history`、`semantic`、`logits` | 官方采样、返回历史、语义切片与每步原始 logits |
| `draw.N`、`prob.N` | 每步真实指数随机数和采样概率；前 11 步宽度 1024，之后 1025，保留 EOS 排除规则 |
| `acoustic_phones`、`ge`、`ge512`、`noise` | 声学条件；`ge512` 布局为 `[B,512,1]` |
| `acoustic.<stage>` | 12 个计算阶段，含最终 FP32 波形 |

历史记录的 15 项声学数组还包括 `ge`、`ge_projected`、`noise`。它们在包中作为三个输入保存，候选重放不能把读入这些条件算成重新完成参考编码。声学噪声来自另一次 CPU seed `20260919` 的固定条件实验，与 GPT 的真实采样随机数来源分开记录。

`manifest.json` 使用 `sakuratts.validation.v1`。所有校验入口需要打开的文件都使用相对路径。每个数组记录小端 dtype、shape、C 顺序及原始字节 SHA-256；模型包仅记录 manifest、权重归档和原始 checkpoint 的身份，权重目录由调用者另行指定。历史 JSON 快照里的绝对路径保留为来源说明，校验器不读取这些路径。

源码与 JSON 快照随包保存。历史上没有记录的上游文件哈希不会用当前文件补写成已知事实。包也不包含日文词典、语言识别资源或原始参考音频，不能用于证明完整文本前端与参考准备已经在 Windows 上通过。

## 验收入口

包内 `verify.py` 只依赖 Python 标准库和 NumPy。搬到另一目录后仍应能读取；不需要安装 MLX、Torch、Transformers 或训练环境。模型归档不随数据包重复分发，使用原转换产物，不能把无损 FP16 存储当作 FP16 推理。

本轮使用 Python 3.11、NumPy 2.4.6。以下命令中的目录可自行选择；Windows 安装与执行尚未验收。

```text
python verify.py verify --bundle <bundle目录>
python verify.py verify --bundle <bundle目录> --gpt-package <GPT目录> --sovits-package <SoVITS目录>
python verify.py compare --bundle <bundle目录> --candidate <候选目录> --output <新报告.json>
```

第一条仅验证包本身，结果会明确外部模型未检查。第二条同时核对权重身份。比较入口退出码 0 表示通过、1 表示数值或生成结果不一致、2 表示文件或格式错误；已有报告不会覆盖。

`research/tools/portable_mlx_candidate.py` 是 Mac 的执行适配器。它用同一个包分别执行固定历史 GPT、自行采样、固定输入声学与自行生成波形。后续 CUDA 实现写出同格式的候选 NPZ 和 manifest，再交给同一 NumPy 比较入口判断，不需要建立多后端框架。

候选格式为 `sakuratts.validation-candidate.v1`，绑定数据包 manifest 哈希、模型身份、参数、源码和各例数组。缺少输出、shape 或 dtype 不符、文件哈希错误不能计为通过。

比较范围分别报告：

- 固定历史 GPT logits：`atol=1e-4, rtol=1e-5`。
- 自行生成：token、history、semantic、返回索引和停止原因要求相同；采样概率为 `atol=1e-6, rtol=1e-5`。第一次 token 分歧后的步骤不再作同输入 logits / 概率判定。
- 固定声学各阶段与最终波形：`atol=1e-4, rtol=1e-5`。

自行生成的 semantic 不同，则该路径的波形数值比较标记为 `skipped_different_semantic`。生成差异仍使整例失败，声学计算是否正确由固定 semantic 的声学对照判断。

日文长句的第 274 步、第 857 个概率值，以及默认 FP32 声学编码器下日文标点的 MRTE `[0,21,125]` 仍是已知超差。包中记录它们，但不会豁免失败或放宽阈值。显式 FP64 softmax 累积可作为独立候选，不能覆盖默认路径的记录。

这些运行含逐步捕获与数组复制，只用于诊断。速度与资源测量继续使用无捕获的完整请求 Harness；ASR、人工试听与 CUDA 实机结果也独立保存。

实际导出、目录搬迁和两种精度的复跑结果见[本轮记录](../research/experiments/2026-09-20-portable-validation.md)。
