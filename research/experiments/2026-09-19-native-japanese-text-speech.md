# 原始日文与离线参考包的完整请求

日期：2026-09-19。自有入口已从四条原始日文文本生成 PCM，目标音素和文本特征由本次请求计算。诊断 8 次、正常运行 32 次全部通过官方对照：目标和拼接后的音素 / BERT、采样 token、生成历史、停止条件与返回语义相同，波形在既定 FP32 容差内。

本轮使用独立的 `.venv-japanese-macos`，未导入 Torch、Transformers、中文 G2PW、ChinesePhones 或 MLX BERT。参考音频已离线准备；没有实现从新参考音频开始的无 Torch 准备，也没有扩展到其他模型家族。

## 请求入口与生命周期

`src/sakuratts/_internal/synthesis.py` 提供两个顺序步骤：

1. `prepare_text(text, language, frontend)` 接收原文，调用真实日文前端，返回本次目标音素、规范化文本和官方日文全零 BERT。返回值不保留前端对象。
2. 前端释放后，调用方加载 GPT / SoVITS，再调用 `synthesize_prepared(prepared, reference, ...)` 生成语音。GPT 接收参考与目标拼接后的音素 / BERT；SoVITS 只接收目标音素和本次 GPT 生成的语义。

`synthesize(...)` 是复用这两步的便利入口，供已经持有所有组件的调用方使用。本轮 Harness 采用前一种顺序，每次请求都释放前端，再加载 GPT / SoVITS，生成后继续释放。模型加载、计算精度及实例所有权由调用方明确控制，没有新增缓存或后端框架。

接口仅接受 `ja` / `all_ja`、`cut0` 的单个片段、`speed=1`、`top_p=1`，参考包的语言必须为 `ja`。需要多个片段时明确拒绝，不截取第一个片段。检查已加载 GPT / SoVITS 的 checkpoint 哈希与官方提交是否匹配参考包。原始参考、模型和历史记录均未修改。

实际完整请求运行使用 `ja`。`all_ja` 的前端与 API 检查已覆盖，但本轮没有单独重复其完整语音实验。中文本轮不加载、不验收。

## 输入与正确性

固定官方提交为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`，模型为“朱雀院红叶”V2Pro，参考为已验证的日文 `VO02_0204.OGG` 条件包。四条原文来自此前官方请求，未修改：原始漏句报告、短句、长句和标点句。

产品入口不读取历史 NPZ。只有 `research/tools/native_text_speech.py` 读取官方诊断，用于提供实际记录的语义指数随机数和固定声学噪声，并在请求计时结束后比较结果。目标特征、目标语义和输出波形均未从金标注入产品入口。

| 样例 | 生成语义 token | 波形最大绝对误差 | 与此前自有 prepared 请求 WAV |
|---|---:|---:|---|
| 原完整报告句 | 120 | 7.40e-5 | 字节相同 |
| 短句 | 23 | 7.86e-5 | 字节相同 |
| 长句 | 373 | 9.73e-5 | 字节相同 |
| 标点句 | 211 | 6.94e-5 | 字节相同 |

波形容差沿用 `atol=1e-4, rtol=1e-5`。诊断与正常运行的四份 WAV 逐字节相同，所有重复请求的波形也相同。这里证明的是从原文接入前端后保持已验证计算路径；固定随机数回放不等于独立多 seed 已通过。

原完整报告句的新 WAV 还与用户此前听过的 `runs/20260919T123730.996242Z-native-prepared-speech/ja-native.wav` 整文件相同，SHA-256 为 `6383cf197139ce6fc262531bee27c0b1faf9616e12876bb1bf8be2c8c8885ca4`。因此此前“四条都正常，未听出明显差异”的反馈可继续适用于这一份日文音频。它不自动覆盖另外三条日文样例；本轮没有新的 ASR 或人工试听。

## 正常计时

四条输入各首次调用 1 次、预热 2 次、测量 5 次。下面为后 5 次中位数，Apple M4；GPT 使用 CPU FP64 Prefill 和 GPU FP32 Decode，声学 encoder 为 CPU FP32，flow / decoder 为 GPU FP32。声学 softmax 使用默认 FP32，未启用 WeightNorm 折叠候选。

| 样例 | 目标前端 | GPT 语义 | SoVITS | 模型加载与校验 | 请求总耗时 | 音频主体 / 含尾静音 PCM | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原完整报告句 | 0.035 s | 0.800 s | 0.244 s | 0.401 s | 1.506 s | 4.80 / 5.10 s | 0.314 |
| 短句 | 0.00025 s | 0.280 s | 0.072 s | 0.423 s | 0.798 s | 0.92 / 1.22 s | 0.868 |
| 长句 | 0.036 s | 2.521 s | 0.729 s | 0.404 s | 3.727 s | 14.92 / 15.22 s | 0.250 |
| 标点句 | 0.037 s | 1.404 s | 0.410 s | 0.412 s | 2.284 s | 8.44 / 8.74 s | 0.271 |

RTF 分母使用音频主体，排除末尾 0.3 秒静音。各列独立取中位数，不能按列相加还原请求总耗时。

请求总耗时包含每次前端加载、文本准备、前端释放、GPT / SoVITS 包校验和加载、生成、CPU 输出复制、PCM 转换及模型释放；不包含初始 Harness / 运行时导入、准备阶段资源哈希、参考包及金标读取、结果比较与磁盘写盘。延迟到前端构造阶段才发生的 pyopenjtalk 包导入计入该次请求。正常路径不执行诊断资源采样。

每例第一次请求分别为 1.716、0.784、3.803、2.250 秒。但这些请求都发生在已启动的 worker 中，不能作为完整应用冷启动；不同样例共享进程级包缓存。初始运行时导入在正常进程中另记为 0.053 秒，诊断进程中为 0.549 秒，也不合并为稳定冷启动结论。

约 0.40–0.42 秒的每次模型加载与校验是明确成本，对短句尤其明显。保留模型与卸载模型的延迟 / 常驻取舍可以在此基础上单独比较，本轮没有将这种顺序卸载称为稳定加速。当前输出完整 PCM，不是流式首包。

## 资源观察

资源另开诊断进程记录同步边界，时间不用于上一节速度结论。每次前端释放后 MLX active / cache 都为 0，证明本轮没有将前端与合成模型同时放入 MLX 分配器。pyopenjtalk 的 Nani / Sudachi 全局缓存仍可能保留。

| 诊断口径 | 实际结果 |
|---|---:|
| GPT / SoVITS 加载后的 MLX active | 469.05 MiB |
| 请求完成、尚未释放模型的 MLX active | 565.05 MiB |
| 每次模型释放后的 MLX active / cache | 0 / 0 MiB |
| 全部诊断请求中的 MLX allocator peak | 1721.21 MiB |
| 长句完成边界的 MLX cache | 2506.61 MiB |
| 诊断进程 OS 生命周期最大 RSS | 1169.34 MiB |
| 正常进程 OS 生命周期最大 RSS | 1174.42 MiB |
| 诊断结束时进程 RSS | 529.39 MiB |

allocator peak 是 MLX 自身分配器计数，不是总 GPU / 系统峰值；cache 是边界快照，两者不能相加。RSS 包含 Harness、保存的金标 / 随机数和返回的 CPU 音频。MLX 清零不意味着进程全部内存已释放。

这些是 Apple 统一内存数据，不能当作 NVIDIA 显存，也没有证明全程资源降低了多少。Windows / CUDA 仍待实机验证。

## 证据与下一步

路径相对于 `SakuraTTS-References/`：

- 诊断：`runs/20260919T141941.619153Z-native-text-speech-diagnostic/`。
- 正常计时：`runs/20260919T141941.903341Z-native-text-speech-normal/`。
- 正常目录中的 `offline-review.json`：四份新旧 WAV、诊断 / 正常 WAV 和本轮入口 / Harness 源码快照核对。
- 同目录的 `source-review-after-docstring.json`：集成后注释与空白改动的差异、文件哈希及排除 docstring 后的 AST 核对。
- 请求边界单元验证：`runs/20260919T141622.433716Z-synthesis-unit/`，7 项通过。
- 先前用户试听记录：`runs/20260919T124243.798790Z-native-user-listening-review/result.json`。

两轮都有原命令、依赖与资源哈希、源码快照、原始 stdout / stderr 和父进程记录的真实退出码 0。`offline-review.json` 保存了首次离线核对时的源码结果。此后 `synthesis.py` 仅将 docstring 中过时的 G2PW / BERT 模型描述改为日文前端组件，`text_frontend.py` 仅移除了一处行尾空格；排除 docstring 后，两者都与快照 AST 相同。`native_text_speech.py` 仍逐字相同。不把整棵当前源码宣称为逐字一致。

复现先用 `research/tools/native_text_speech.py prepare` 新建目录，再执行 `run --run <新目录>`。完整准备参数保存在两轮 `prepared.json` 的 `command`；正常组选择 `--mode normal --warmup 2 --repeat 5`。不要重新运行原目录。

后续边界仍包括：实际更换参考包、重复卸载后的驻留变化、独立多 seed 生成，以及另外三条日文的内容和音色验收。先沿这条纯日文链路补齐证据，中文另行推进。
