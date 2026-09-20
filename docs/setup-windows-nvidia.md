# Windows / NVIDIA 独立推理

SakuraTTS 的 Windows 路径使用自己的日文前端、GPT CUDA 执行器、采样与请求控制，再由独立 ONNX Runtime 工作进程执行 SoVITS。普通发声不导入 PyTorch，也不读取 g50、Lite、Genie 或 `SakuraTTS-References` 目录。原始权重和官方源码只在模型转换、参考准备及开发对照时使用。

当前范围是 Windows / NVIDIA、日文、单模型组合、单请求、完整 WAV。默认保留 GPT/声学 FP32 和 baseline attention；split-KV 与声学 FP16 已提供显式候选，使用条件见下文。量化、中文及混合语言、流式播放和宿主集成尚未交付。V2ProPlus 的具体权重、数值和性能结果见 [Windows 实测记录](experiments/2026-09-20-windows-nvidia-backend.md)；不能把模型家族标签当作其他权重已验证的结论。

以下命令都在取得的 SakuraTTS 仓库根目录执行。安装命令默认使用 `--offline`，只读取本机 uv 缓存；缓存不完整时停止并报告缺失项，不自动联网。

## 日常环境

前置条件是 Windows x64、可用的 NVIDIA 驱动、本机 Python 3.11 和 uv。独立声学组件还需要本机 Windows C/C++ 运行库。

```powershell
uv --offline venv .venv-windows-runtime --python 3.11
uv --offline pip install --python .venv-windows-runtime\Scripts\python.exe -r requirements-windows-runtime.txt
uv --offline pip install --python .venv-windows-runtime\Scripts\python.exe --no-deps -e .
uv --offline pip check --python .venv-windows-runtime\Scripts\python.exe
.venv-windows-runtime\Scripts\sakuratts.exe doctor --japanese
```

已有这个目录时直接使用，不要为重跑示例而删除或重建环境。固定依赖清单见 [requirements-windows-runtime.txt](../requirements-windows-runtime.txt)。主环境保留 plus 前端所需的 CPU ORT 1.30.0；本机官方对照使用下面打包的经典日文前端。声学部分使用下一节的 ORT CUDA 1.19.2 组件。主环境不安装重复的 cuDNN。

`doctor --japanese` 检查前端依赖；模型准备完成后运行 `doctor --nvidia --config models/windows-sakura/runtime.json`，可检查运行依赖、资源哈希、参考身份和独立工作进程导入。检查不会创建 TTS Session 或执行 GPU 推理，`inference_tested` 和 `quality_validated` 仍为 `false`。ORT 声明 CUDA provider 可用也不证明该设备能完成推理。`doctor --cuda` 是 PyTorch 开发环境检查，不适用于这个无 Torch 的日常环境；实际请求使用下面的 `synthesize` 入口。

## 准备独立声学组件

当前机器的可用 ORT CUDA 1.19.2 wheel 使用 CPython 3.9 ABI。为保持离线，本轮采用单独的持久工作进程，没有把该 wheel 强行装进 Python 3.11。工作进程运行 SakuraTTS 自己的业务代码。

使用本机已有的官方运行文件与开发环境中的 NVIDIA DLL 导出组件：

```powershell
.venv-windows-runtime\Scripts\python.exe -B scripts\prepare_ort_worker_runtime.py `
  --source-runtime D:\Project\sakura\tts\g50\runtime `
  --nvidia-root .venv\Lib\site-packages\nvidia `
  --output data\windows-ort-runtime
```

这是一次本地文件准备。工具只复制 Python 标准库、NumPy、ORT 和必要 DLL；不复制 Torch 包或完整 g50 环境。CUDA DLL 留在输出目录的 `cuda` 子目录，原目录之后不参与普通运行。缺少本地 DLL、许可文件或版本不符时，工具应报错；不要把 g50 加进日常运行的 `PATH` 来绕过缺失组件。

已准备的 `data/windows-ort-runtime` 可直接使用，无需重复导出。输出目录存在时工具拒绝覆盖。逐文件哈希、来源、许可和隔离导入检查见该目录中的 `runtime-manifest.json`，详细边界见 [离线 ORT 组件说明](setup-ort-worker-offline.md)。

## 一次性转换模型和准备参考

这一步需要开发 Python；普通环境不需要这些依赖。本机已准备好的 `.venv` 可以直接使用。若本地缓存齐全，也可另外新建开发环境，并离线安装 `.[japanese,nvidia,dev]` 及准备声学组件所用的 `nvidia-cudnn-cu12==9.26.0.51`。开发环境与日常环境分开，避免把导出器依赖带入普通运行。

以下以用户提供的 Sakura 目录为例，输出放入一个新的 `models/windows-sakura-new` 目录。示例不会覆盖原角色目录、权重或参考音频，也不会覆盖本机已存在的 `models/windows-sakura` 产物。

先从原始参考音频计算五组条件，并导出完整日文前端资源：

```powershell
.venv\Scripts\python.exe -B scripts\prepare_windows_resources.py `
  --official-source D:\Project\sakura\tts\g50 `
  --character D:\Project\sakura\characters\Sakura `
  --python D:\Project\sakura\tts\g50\runtime\python.exe `
  --language-model D:\Project\sakura\tts\g50\GPT_SoVITS\pretrained_models\fast_langdetect\lid.176.bin `
  --device cuda --precision fp32 `
  --output models\windows-sakura-new
```

`--python` 只选择准备阶段使用的现有解释器，不安装或修改它。准备过程加载 SoVITS、HuBERT 和说话人编码器，不加载 GPT 或中文 BERT，也不生成目标语音。它保留原音频的实际处理，并输出参考语义、音素、日文零 BERT、`ge` 和 `ge512`；完成后核对原文件哈希。

准备工具会先在所选解释器中读取实际日文前端版本。本机 g50 使用 `pyopenjtalk 0.3.4`，工具会将该模块、原主字典和许可一起复制到前端包的 `classic-python` 子目录，并记录每个文件的哈希。普通运行通过独立 Python 3.9 工作进程使用这些包内资源，无需读取 g50。开发环境中的 `pyopenjtalk-plus 0.4.1.post9` 在部分长句和标点输入上会产生不同音素，不能用它替代官方前端做相同输入对照。使用 plus 解释器准备时会明确记录 plus profile，不会悄悄改用经典版。

工具默认 `--device cpu --precision fp32`，也可显式选择 CUDA。两种设备的参考语义和音素已做过对照，但声学条件存在浮点差异；固定数值对照必须使用同一组准备产物，不能将 CPU 条件替换进 CUDA 官方对照而不说明。FP16 准备是另一种精度设置，不属于这里的 FP32 基线。

只需前端资源时可加 `--frontend-only`。词典陈旧、缺失完整 `lid.176.bin` 或准备解释器缺依赖时，工具明确失败，不重建原目录词典，不自动下载，也不会用空模块替代缺失依赖。

随后转换两份角色模型：

```powershell
.venv\Scripts\python.exe -B scripts\convert_gpt.py `
  --checkpoint D:\Project\sakura\characters\Sakura\voice\models\Sakura-e15.ckpt `
  --official-source D:\Project\sakura\tts\g50 `
  --output models\windows-sakura-new\gpt

.venv\Scripts\python.exe -B scripts\export_sovits_onnx.py `
  --checkpoint D:\Project\sakura\characters\Sakura\voice\models\Sakura_e8_s7176.pth `
  --official-source D:\Project\sakura\tts\g50 `
  --output models\windows-sakura-new\sovits
```

转换器读取真实结构并输出独立模型包。导出失败或声学验证失败时，保留诊断结果；不能通过改版本标签、跳过缺失权重或扩大容差把失败包标为可用。

## 配置模型与参考

参考准备工具会生成 `runtime.json`，其中路径相对该文件。使用当前独立声学组件时，加入 `acoustic_python`：

```json
{
  "format": "sakuratts-windows-config-v1",
  "gpt": "gpt",
  "sovits": "sovits",
  "frontend": "frontend",
  "references": {
    "中性": "references/中性",
    "不满": "references/不满",
    "害羞": "references/害羞",
    "请求": "references/请求",
    "惊讶": "references/惊讶"
  },
  "default_reference": "中性",
  "acoustic_python": "../../data/windows-ort-runtime/python.exe"
}
```

本机已准备 `frontend-classic` 供官方前端对照，原 `frontend` 包仍保留。本机已有的 `models/windows-sakura/runtime.json` 使用 `sovits-onnx-v1` 声学包和 CPU 准备的参考条件；专用 `runtime-validation.json` 使用同模型的 `references-cuda`，用于保持官方 CUDA 对照条件相同。将前端切换为经典版时，配置中的 `frontend` 应指向 `frontend-classic`。两份配置用途不同，不要为了重跑示例而互相覆盖。

自然生成可以使用 `runtime.json`；官方固定随机回放必须使用 `runtime-validation.json` 的 CUDA 参考。CPU/CUDA 准备的文本身份相同也可能得到不同 `ge/ge512`，不能混作同条件。回放 Harness 已在运行前检查参考数组一致性；早期错误参考的记录及修正结果见 [split-KV 实验](experiments/2026-09-20-windows-split-kv.md)。

正常运行需要 GPT 包、声学包、前端包、所选参考包及两个运行环境。模型包中保留的原始路径只描述来源；校验和推理读取的是包内文件，不要求原 g50 或角色目录仍存在。

## 从日文原文生成 WAV

使用本机已准备的配置：

```powershell
.venv-windows-runtime\Scripts\sakuratts.exe synthesize `
  --config models\windows-sakura\runtime.json `
  --reference 中性 `
  --text "おはよう。今日もよろしくね。" `
  --seed 1234 `
  --output outputs\sakura-neutral.wav
```

新导入模型则把 `--config` 换为 `models/windows-sakura-new/runtime.json`。更换参考只需调整 `--reference`，不用重新编码音频。

入口写入完整 WAV 和同名 JSON 记录。输出路径必须是新的 `.wav`，已有 WAV 或 JSON 都会被拒绝。退出码 `0` 表示生成按正常停止条件完成；达到生成上限时仍可能写出音频，但状态为 `stopped_at_limit`、退出码为 `2`，不能据此认定文本完整。模型、资源或推理失败使用非零退出码，并报告具体错误。

默认是 `ja`、`cut0`、Top-k 15、Top-p 1.0、温度 1.0、重复惩罚 1.35、语速 1.0。JSON 记录每片音素、语义 token、停止原因和耗时。SakuraTTS 的 NumPy 随机数与官方 Torch 随机数实现不同，相同 seed 不意味着相同抽样；严格数值对照使用固定随机输入回放。

默认 `--model-policy resident` 在当前进程内保留模型。`release-state` 和 `staged` 是可选生命周期策略，`--no-cuda-graph` 可关闭 GPT 图执行用于对照；是否值得使用以完整请求测量为准。每次 CLI 命令都会启动新进程，不能把多次 CLI 调用称作同进程热请求。

`--gpt-precision fp16` 可试用 GPT 混合精度：权重和 KV 使用半精度，主要累积与采样输入保留 FP32；只设置此参数时，声学仍为 FP32。GPT 模型包无需重新转换。默认 `fp32` 保留原数值对照；半精度会改变 logits，内容和听感尚待验收。资源与耗时见 [GPT 混合精度实验](experiments/2026-09-20-windows-gpt-fp16.md)。

### split-KV 与声学 FP16 候选

`--gpt-attention split-kv --gpt-attention-chunk-size 256` 启用分块 Decode 注意力，块长也可选 512。GPT FP32 的后续固定历史、边界和正确参考回放通过原容差，但首次 256 运行的 Prefill 异常尚未找到根因；GPT FP16 加 split-KV 未通过同精度严格检查。默认仍为 `--gpt-attention baseline`，不能把两个 GPT 精度的结果互相替代。

声学 FP16 需要独立转换并通过 screen v2 的模型包。本机通过的包是 `outputs/windows-acoustic/fp16-candidate-lowered`，其中转置卷积已改写为零插值与普通卷积，以解决重复执行波动。配置中的 `sovits` 必须指向这个已筛查包，随后显式加入 `--allow-experimental-acoustic-fp16`。该参数只允许加载候选，不会在运行时转换模型，也不会跳过绑定图、权重及 Session 设置的筛查结果。转换和筛查命令使用 [声学 FP16 实验](experiments/2026-09-20-windows-acoustic-fp16.md#使用与证据)中已验证的流程，不覆盖原 FP32 包。

本机已准备的组合配置可这样试用，GPT 仍选 FP32：

```powershell
.venv-windows-runtime\Scripts\sakuratts.exe synthesize `
  --config outputs\windows-acoustic\runtime-fp16-lowered.json `
  --reference 中性 `
  --text "おはよう。今日もよろしくね。" `
  --gpt-precision fp32 `
  --gpt-attention split-kv --gpt-attention-chunk-size 256 `
  --allow-experimental-acoustic-fp16 `
  --output outputs\sakura-neutral-mixed.wav
```

该配置引用 CUDA 参考和已筛查的 lowered 声学包。本机四类固定回放的完整请求热中位数（各三次，不含写盘）中，27.30 秒长句从原 FP32 baseline 的 3045.15 ms 降到 1592.30 ms，2.02 秒短句从 168.94 ms 降到 122.23 ms。组合候选通过预设工程筛查，仍不满足原 FP32 波形容差；未做人工听音或 ASR，也没有测流式首包。自然生成 CLI 已执行，但单次 CLI 的加载成本不能与上述热请求混比。

独立资源轮使用同一固定输入，测得全卡峰值减首个空载样本从 1804 MiB 降到 1624 MiB；包含桌面负载且可能漏采短峰值，不是进程独占显存。组合同时改变 attention 和声学路径，不能把全部收益归到单项。详细单项与组合对照见 [split-KV](experiments/2026-09-20-windows-split-kv.md)及[声学 FP16](experiments/2026-09-20-windows-acoustic-fp16.md)。

声学默认使用 `HEURISTIC` 卷积算法搜索并关闭最大 cuDNN 工作区。较大工作区的候选在实测中可以更快，但会超过这张 8 GB 显卡的可用显存预算；默认限制有速度代价，不能称为免费优化。工作区、加载策略和完整请求的具体取舍见实测记录，不根据单个算子的计时改变默认配置。

## 已核对的安装成本与边界

本机从缓存新建 `.venv-windows-runtime`，运行包依赖检查、CLI 帮助和日文前端检查均通过。Torch、torchaudio、Transformers、ONNX 导出器与 MLX 均不可导入，CUDA DLL 发现路径全部位于新的日常环境。移除重复 cuDNN 后、安装测量用 psutil 前，主环境逻辑文件大小为 1,560,256,402 字节，约 1.45 GiB。

独立 ORT 组件的复制清单为 3,069,259,906 字节，约 2.86 GiB。以上安装阶段两项合计约 4.31 GiB，尚未包含模型、参考包、共享基础 Python、编译缓存和保留的开发环境；当前目录还包含随后离线安装的测量依赖。双进程带来的 CPU 内存与传输成本也要计量，不能仅报告较小的主环境体积。

后续[逐文件清点](experiments/2026-09-20-windows-runtime-inventory.md)把共享 Python、选中资源与缓存分开计量，并确认两环境有 828.58 MiB 的相同 CUDA DLL。当前加载路径仍各自依赖这些文件，尚未实际删减，也未把潜在收益从安装量中扣除。

完整进程 CPU 内存测量使用 `psutil==7.2.2`，已声明在开发依赖中，普通合成无需安装。若在日常环境执行测量 Harness，可从已有缓存离线安装该包，并将增加的文件计入测量环境体积；本轮没有为它联网下载。

上述检查证明本机离线安装和依赖隔离，未替代另一台干净机器的验收。本轮 Windows 新样音尚未完成人工听音或日文 ASR 检查；数值对照和正常停止也不能单独证明文本完整或音质合格。实际运行过的文本、参考切换、显存、冷启动、热请求和已知问题统一记录在 [Windows 实测记录](experiments/2026-09-20-windows-nvidia-backend.md)。本页不使用 Mac 或社区性能数字推断 RTX 5060 的结果。
