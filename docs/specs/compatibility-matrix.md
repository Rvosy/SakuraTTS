# 模型与功能验证矩阵

更新日期：2026-09-19。当前证据来自 [Mac 上游功能冒烟](../experiments/2026-09-19-macos-reference-smoke.md) 、[调用链与固定历史对照](../experiments/2026-09-19-parity-and-lifecycle.md) 和用户试听反馈。SakuraTTS 尚未完成自有运行时，表中的上游生成结果不能作为自有引擎的兼容声明。

## 固定对照

| 项目 | 当前对照 |
|---|---|
| 官方实现 | GPT-SoVITS `48b1a0169a28582a8984402f82cf438d3bfa6aca`，普通、非并行、非流式路径 |
| 优化参考 | GSV-TTS-Lite `6c049397142f4c9147a85f86b6ba37546e93a188`，中文 BERT 开启、FlashAttention 关闭 |
| 模型 | “朱雀院红叶”V2Pro，GPT SHA-256 `010197bfc30b04d991f2bf060f962549932a8278b98c137d92f980e9cca8c0e9`，SoVITS SHA-256 `f8bd92196175f435ae9df2bc01e87b0119a5d94c0af81ada4573a9720536ab38` |
| 参考条件 | “开心”条目 `VO02_0204.OGG`；日文参考文本 `じゃあ、私、もっと悪い子になっちゃおうな〜` |
| 平台与精度 | Apple M4、16 GiB 统一内存；官方 MPS / FP32，自有 MLX / FP32，明确指定的 CPU FP64 GPT Prefill |
| 固定样例 | [speech_regressions.json](../../harness/cases/speech_regressions.json) 中的 `ja-reported-intro` 和 `zh-reported-greeting` |

具体环境、采样参数、原始产物路径和首次运行的限制见上述实验记录。每次新增运行仍需保存实际提交、权重和参考音频哈希，不能只引用这张表。

近期产品范围已收紧为日文。中文与混合语言的下列历史证据保留，但不代表当前日文入口接受这些语言；新增中文整链和专项优化暂缓。

## 模型家族

“生成成功”仅表示已得到非空、有限、非零的音频文件；“内容质量”单独判断漏句、发音和音色。家族名依据当前官方版本划分，未验证家族不承诺共享权重布局、特征或显存目标。

| 家族与样本 | 官方 MPS / FP32 生成 | Lite MPS / FP32 生成 | 内容质量 | SakuraTTS 自有转换与运行时 | Windows / CUDA |
|---|---|---|---|---|---|
| V2Pro：“朱雀院红叶” | 10 条中日及混合输入已生成，已保存中间结果 | 中、日固定样例多次生成，已保存中间结果 | Lite 有 3 项用户报告的失败；官方 / official_text 及后来两条自有整链回归的指定样音经用户确认；后者覆盖全文和音色，其余八例待验 | 十例 prepared 整链 token、停止、最终波形通过；独立诊断有 3 个概率值及 1 个 MRTE 中间值超差，整体仍未通过；原始文本到音频未验收 | 暂缓，未验证 |
| V2Pro：其他权重 | 未验证 | 未验证 | 未验证 | 未验证 | 暂缓，未验证 |
| V2ProPlus | 未验证 | 未验证 | 未验证 | 未验证 | 暂缓，未验证 |
| V2 | 未验证 | 未验证 | 未验证 | 未验证 | 暂缓，未验证 |
| V1 | 未验证 | 未验证 | 未验证 | 未验证 | 暂缓，未验证 |
| V3 | 未验证 | 未验证 | 未验证 | 未验证 | 暂缓，未验证 |
| V4 | 未验证 | 未验证 | 未验证 | 未验证 | 暂缓，未验证 |

本表仅说明已识别家族的验证状态。新增官方版本或变体时，需要先核对固定官方提交中的实际结构，再增加独立条目。

## 已知语音问题

以下现象保留为 `user_reported_failure`。反馈来自用户对 Lite 样音的试听，已建立官方生成与固定历史数值对照，并已完成指定官方与 official_text 候选的复听，用户回复“一切正常”；原 Lite 失败记录保留，不能将候选结果推广到其他路径。

| 回归样例 | 用户报告 | 当前验证状态 |
|---|---|---|
| `ja-reported-intro` | 缺少开头的“こんにちは。” | 参考文本条件差异是消融定位重点；官方与 official_text 候选开头经 ASR 识别及用户确认 |
| `ja-reported-intro` | “今日はいい天気ですね。”中的助词“は”听起来读成 ha，预期为 wa | 两边音素均为 w a；用户确认官方与 official_text 候选正常，ASR 不用于证明发音 |
| `zh-reported-greeting` | 开头的“你好”听起来像“哼哼” | 确认前导停顿与 BERT 标点差异；用户确认官方与 official_text 候选开头正常 |

后续自有整链的两条原始回归已有独立复听，用户对指定的四个官方 / 自有 WAV 回复“四条都正常，未听出明显差异”，覆盖上述三项问题、全文及音色。具体文件与哈希见 [整链记录](../experiments/2026-09-19-native-prepared-speech.md)，不覆盖未听过的扩展样例。

两条原始输入保持不变。短句等新增输入用于缩小触发范围，不能替代原始样例的回归验收。排查范围包括规范化、分句、音素与 BERT 对齐、参考和目标拼接、生成历史、停止与语义切片，以及音频后处理；当前不预设根因。

## 语言与功能

| 语言或功能 | 当前证据 | 待验证内容 |
|---|---|---|
| 日文多句 | 官方和 Lite 均生成文件；Lite 有漏句和助词读音反馈 | 固定历史 121 步 Lite logits 完全一致；指定官方 / official_text 的开头和助词经确认；后续官方 / 自有整链两条对应日文样音的全文和音色也获确认，扩展样例待验 |
| 中文多句，BERT 启用 | 官方和 Lite 均生成文件；Lite 有开头发音反馈 | 固定历史 147 步 Lite logits 完全一致；BERT 输入差异已定位；指定官方 / official_text 的“你好”经确认；后续官方 / 自有整链两条对应中文样音的全文和音色也获确认，扩展样例待验 |
| 中、日文单独短句与长句 | 官方均已生成并保存 trace；纯 FP32 MLX 日文长句第 329 步超差，高精度 Prefill 通过原容差 | prepared 整链最终波形已通过；采样概率 / 声学中间量的已知失败保留，扩展语音质量待验 |
| 标点、引号、问句和省略号 | 官方生成并保存 trace；两条 MLX 固定历史通过预设容差 | 内容、规范化、分句和停顿的质量验收 |
| 中英、日英混合 | 首次缺英文资源失败已保留；补齐后官方两条已生成并保存 trace | 高精度 Prefill 候选固定历史通过；语言切换、逐段内容与音色待验 |
| 纯英文及其他官方语言 | 未建立样例，未验证 | 先核对模型家族和官方语言前端 |
| 当前日文参考条件 | 两端已用于生成；官方准备后释放辅助模型和复用 embedding 保持固定样例 WAV | 单参考持久包已独立重载；其他参考与音色验证 |
| 中文 BERT 第 22 层 | CPU/MPS 裁剪后特征完全一致；独立 MLX CPU 的 5 段、120 项对照通过，正常计时和卸载已测 | MLX GPU 仍有 1 个中间元素超差；独立路径接入音频、其他语料与分发待验 |
| 其他参考语言、无参考文本、多参考条件 | 未验证 | 逐项核对官方支持范围和条件语义 |
| 自有 NumPy 采样与非流式停止 | 十条 Top-p 1 请求的 1805 步 token、停止和语义切片与官方相同；logits 通过，2 例 3 个概率值超差，整轮仍失败 | GPT 误差传播、其他 seed、独立 RNG、Top-p 小于 1 及 GPU 采样；见 [十例生成](../experiments/2026-09-19-expanded-native-generation.md) |
| 准备好条件的自有语音链 | 十例自有 GPT → 语义 → 声学 → PCM 的 token、停止和最终波形通过；原始两例 WAV 与用户确认的四样音逐字节相同 | 既有概率及 MRTE 超差仍保留；原始文本和参考准备、独立 RNG、扩展质量及完整延迟；见 [十例整链](../experiments/2026-09-19-expanded-prepared-speech.md) |
| 自有完整声学链 | 650 张量解码包严格重载通过；CPU encoder + GPU flow/decoder 的十例波形通过，119/120 阶段通过 | 日文标点 MRTE 单元素超差；全 GPU 两例仍有中文波形超差；见 [扩展声学](../experiments/2026-09-19-expanded-acoustic.md) |
| 声学 CPU softmax 高精度累积 | 显式 `encoder_softmax=fp64-accumulation` 的十例 120 阶段通过原容差；源码输出逐位复现候选；默认 FP32 保留原行为 | 编码器时间与工作区增加；新输出的整链、试听待验，其他模型与 GPU 不覆盖；见 [精度与成本](../experiments/2026-09-19-softmax-candidates.md) |
| 声学速度与缓存 | 调度优化前两条正常声学请求比官方慢约 29%–41%；后续缓存、工作区和整链取舍单列 | 各条件的数据不可混用；MLX 统计仅适用于 Apple 统一内存，见 [原声学实验](../experiments/2026-09-19-mlx-sovits-complete.md) 与 [工作区实验](../experiments/2026-09-19-decoder-workspace.md) |
| 声码器工作区 | 新 pair 求值方式的十例波形逐位保持；两条 decoder-only 峰值少约 45%，慢约 10%–14%；整链已复核 | 长句降幅较小；完整文本路径的峰值与部署成本；见 [工作区实验](../experiments/2026-09-19-decoder-workspace.md) |
| 中文 tokenizer | 自有接口与官方词元及 G2PW 所需基础数组一致，600 字 BERT 输入保持 602 tokens | 完整中文规范化、G2PW 推理和音素/BERT 对齐；见 [tokenizer 实验](../experiments/2026-09-19-tokenizer-runtime.md) |
| G2PW 输入准备 | 自有纯 NumPy 实现的 18 映射、366 组输入与官方一致；保留官方请求内分词与实例静态表复用 | 同源 ONNX 概率已另行对照；正在连接多音字前后处理，>510 异质长文本的官方去重边界保留；见 [G2PW 输入实验](../experiments/2026-09-19-g2pw-inputs.md) |
| G2PW 文本准备 | 25 组与固定官方函数对照一致，含 23 组完整输出和 2 组相同域外异常；保留 OpenCC / PyPinyin / 上下文 | 完整拼音与音素链尚未验收；见 [文本准备](../experiments/2026-09-19-g2pw-text.md) |
| G2PW CPU 模型推理 | 自有 ORT 接口 8 组输入、16 组概率及标签/置信度与官方逐位一致，无 Torch/Transformers | 文本接口仍在组合；>510 异质词元的官方去重边界保留；三轮关闭后 RSS 约 380 MiB 趋稳，长期情况未验；见 [ONNX](../experiments/2026-09-19-g2pw-onnx.md) 与 [生命周期](../experiments/2026-09-19-g2pw-lifecycle.md) |
| G2PW 完整拼音接口 | 规范化中文片段经 Text → Inputs → Session → 拼音填回，27 组输出 / 异常一致，353 数组逐位相同，无 Torch/Transformers；原边界保留 | 中文段规则已另行接通，完整请求待验；见 [拼音闭环](../experiments/2026-09-19-g2pw-pinyin.md) |
| 中文 V2 语言段 | 10 组真实 G2PW 后规范化、音素、ID、word2ph 全同；27 组规范化和 45,050 词典检查一致；五组 MLX CPU BERT 通过原阈值 | 上层语言路由与原始文本到音频仍待接入；见 [中文前端](../experiments/2026-09-19-chinese-phones.md) |
| 日文语言段 | 26 组完整 NJD、labels、韵律和音素 ID 与官方一致，实际覆盖 Nani、Sudachi 与用户词典，无 Torch | 上层路由、全链和新试听待验；见 [日文前端](../experiments/2026-09-19-japanese-g2p.md) |
| G2PW 映射 ORT 包 | 实际接口 27 组 / 353 数组逐位一致；同条件 OS 最高 RSS 约 1357→851 MiB | Mac CPU 数据；Windows 需重建包，完整 TTS 资源未测；见 [映射实验](../experiments/2026-09-19-g2pw-mapped-ort.md) |
| 单参考持久包 | 五数组新进程逐字节相同，12 项身份 / 损坏拒绝检查通过；完整包约 203 KiB | 不含无 Torch 的新参考准备，历史缺失身份已明列；见 [参考包](../experiments/2026-09-19-reference-condition-package.md) |
| 静态 WeightNorm 折叠 | 显式选项下 131 权重、十例 330 阶段逐位相同；Flow 峰值少约 105–107 MB，完整声学收益较小 | 加载后缓存增加、完整速度有波动，默认关闭；Mac GPU 数据，见 [预计算取舍](../experiments/2026-09-19-sovits-static-weights.md) |
| 声学共享归档 | 650 权重与十例 120 阶段逐位相同；整包校验 / 打开从三次变一次；五新进程加载中位 231→155 ms | 权重常驻不变，Mac 加载数据不代表 CUDA；见 [共享加载](../experiments/2026-09-19-sovits-shared-loading.md) |
| 权重无损存储 | 三归档少 801.37 MiB；加载恢复的 1,303 张量逐位相同；复用已加载权重收回 Prefill 重复读取代价 | 加载成本和最终分发仍需衡量；不代表运行权重减少，见 [存储实验](../experiments/2026-09-19-lossless-weight-storage.md) |
| GPT 按阶段释放 | 两条 prepared 请求的 GPT/KV 可在声学生成前释放，输出逐位保持；结合 pair 调度与缓存设置，本轮 allocator peak 为 622.05 MiB | 每次重载增加延迟；不含文本/参考准备、不推广到长句与其他模型；见 [生命周期](../experiments/2026-09-19-gpt-lifecycle.md) 与 [工作区](../experiments/2026-09-19-decoder-workspace.md) |
| 当前单参考声学条件预计算 | ge/ge512 与官方实际条件相同；10 条 WAV 逐字节保持；请求后 allocated 少 148.99 MiB | RSS 峰值未降、通用持久化与多参考；仅显式实验选项 |
| 模型卸载 | 官方单进程卸载后 MPS allocated / driver 边界已记录 | 多轮泄漏、空闲恢复与角色切换 |
| 流式、取消与切换 | 未验证 | 输出顺序、资源释放及后续请求正确性 |
| FP16、量化、CUDA Graph、FlashAttention | 本轮未验证 | 独立于 MPS / FP32 对照评估数值、质量、性能 |

## 证据如何更新

固定语料的 `status` 表示样例来源：`user_reported_failure` 是需要解决的用户反馈，`exploratory` 是尚未验收的扩展输入。每条样例有稳定 `id`、`language`、原始 `text`、覆盖标签 `coverage`、`user_reports` 和独立的 `checks`；运行结果引用样例 ID，保存在参考目录的新运行目录中。

自动检查、ASR 和人工试听分别记录结论与产物。音频文件有效、哈希一致、前端一致或固定历史数值通过，都不能直接把人工试听状态改成通过。ASR 未执行时记录 `not_run`；人工没有听过的输出不得记为已复听。

每次修复后应更新本矩阵对应条目并链接新的实验记录，保留旧失败证据。正式列为支持需要完成 [推理契约](inference-contract.md) 与 [基准协议](benchmark-protocol.md) 的相关验证。MPS 统一内存值与 NVIDIA 独立显存分开报告，执行边界快照不称为峰值；当前没有可用于全部模型版本的 0.8 GB 承诺。

指定样音的人工检查见 [ASR 与复听记录](../experiments/2026-09-19-asr-review.md)。另有 [自有整链四样音的独立复听](../experiments/2026-09-19-native-prepared-speech.md)。这些通过结果不代表原 Lite 已修复，也不代表原始文本到音频或扩展模型全部通过。

参考资源释放与 BERT 裁剪的组合已通过 10 条扩展输入的逐字节 WAV 对照，见 [扩展回归](../experiments/2026-09-19-expanded-regression.md)。这批扩展音频尚无新增 ASR 或人工检查。

长句误差的逐层定位与通用高精度候选见 [MLX 数值实验](../experiments/2026-09-19-mlx-numerics.md)。固定条件下的官方 / Lite 声学中间数组与波形对照见 [SoVITS 实验](../experiments/2026-09-19-sovits-fixed-conditions.md)。
