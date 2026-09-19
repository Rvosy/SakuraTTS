# 日文语言段前端脱离 Torch

日期：2026-09-19。自有 `JapaneseG2P` 已用固定 pyopenjtalk-plus 和原词典完成 26 组日文语言段对照。规范化文本、全部 NJD 字段、完整标签、韵律音素、V2 的 `UNK` 映射与音素 ID 均与固定官方函数完全相同。官方和自有实现各运行诊断、正常计时两个独立进程，四个进程实际退出码均为 0，没有导入 Torch、Transformers 或 MLX。

本轮把官方日文文本规则接到独立 OpenJTalk 实例，继续使用相同的前端库和词典。没有生成语音，也没有新增 ASR 或人工试听；结果不能代替语音质量验收。

## 接口与边界

`src/sakuratts/japanese.py` 提供 `JapaneseG2P(main_dictionary, user_dictionary)`：

- `normalize(text)` 保留官方连续标点处理。
- `g2p(normalized_text)` 保留百分号替换、英文字母转小写、日文子句和标点拆分、默认韵律符号及标点映射，返回原始音素列表。
- `labels(sentence)` 通过同一个显式 OpenJTalk 实例执行 `run_frontend` 和 `make_label`。
- `close()` 只释放本实例。关闭后调用标签分析会报错，避免意外退回没有指定用户词典的全局实例。

主词典与用户词典路径由调用者指定。推理入口不编译词典、不覆盖原文件；缺文件或缺少“何”的两个 ONNX Session 会直接报出原因。官方全局用户词典加载方式与自有显式实例方式在本轮产出了相同结果。

默认发音能力没有调整：Sudachi 读音修正和 `predict_nani` 开启；`use_vanilla`、marine、tsqyomi、`use_read_as_pron`、`revert_long_vowels`、`revert_yotsugana` 关闭，`normalize_mode="None"`。韵律算法仍把无声化的大写元音转小写，移除每个子句的首尾边界，并保留内部 `#` 等符号。

V2 符号表不收录的音素继续映射为 `UNK`，没有删除。该映射和 ID 转换在 Harness 中按原 `symbols2.py` 执行；运行时类只返回原始音素，便于之后接到公共文本入口。日文 `word2ph` 为 `None`。

本接口不负责上层语言路由，也不执行目标文本的前导标点、短句重试或参考文本的末尾补点。混合语料本轮只对照实际分出的日文段，没有把整个英日混合句送入日文模块来代替原语言路由。

## 同条件对照

固定官方提交为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`。Harness 从该提交的 `japanese.py` 提取原始纯函数作为 oracle，先明确加载原用户词典，再执行原 `run_frontend → make_label` 路径，避开模块导入时可能重编词典和吞掉加载错误的副作用。官方及自有包的 pyopenjtalk Python 源码、主词典、Sudachi 资源和两个小 ONNX 文件身份均相同。

26 组输入包括四条既有日文回归、混合语料的两个日文段、参考原文及补点版本、“は”、六个“何”语境、两个 Sudachi 语境、数字、连续标点、未知符号、空格、空串和用户词典词条。

| 实际覆盖 | 观察 |
|---|---|
| 原失败句 | 58 个音素、2 个 `UNK`；完整文本没有缩短 |
| 已补点的参考文本 | 47 个音素、4 个 `UNK`；原始未补点版本为 46 个音素 |
| `今日はいい天気ですね。` | “は”的 `read=ハ`、`pron=ワ`；全部 NJD 节点与之前保存的官方结果一致 |
| `これは何ですか。` | 实际执行 Nani 编码器和分类器各 1 次，原始输入及输出完全一致；二类概率为 `[0.19944684207439423, 0.8005530834197998]` |
| 其他五个“何”语境 | 本轮没有进入 ONNX 预测；保留实际规则或分词行为，没有按样例名称推断模型覆盖 |
| Sudachi | 共执行 5 次；包含长句中的 3 次、风的两种读法及方的两种读法 |
| 用户词典 | `hello` 和 `abandonment` 的 MeCab `dictionary_index` 均为 1；两实现节点、代价和读音完全一致 |
| 数字、标点、未知符号、空串 | 规范化、原始音素、`UNK` 和 ID 全同 |

`abandonment` 沿用原用户词典中的 `アバッシュド`。它用于确认资源命中，不作为读音正确的金标，也没有在自有接口里改词典来修正它。

26 组诊断记录逐字段相等。两个正常进程各执行 26 × 8 次调用，每组首次、2 次预热、5 次计时；全部重复输出与诊断金标相同。另将结果与更早的 `20260919-frontend-audit/official-frontend.json` 对照，6 项目标 / 短句 / 助词 / 参考组合的规范化文本、符号和 ID 完全一致。源码快照与当前文件逐字一致。

## 正常路径耗时与资源

诊断进程记录 NJD、labels、Nani 的数组和 Sudachi 调用。正常进程不安装这些捕获钩子，计时覆盖规范化、G2P、`UNK` 与 ID 转换，不含包导入、实例初始化、资源哈希、检查和 JSON 写盘。同一进程顺序处理全部样例，因此某一例的“首次”不是新进程冷启动。

下表是每例 5 次正常调用的中位数：

| 日文段 | 官方 | 自有 |
|---|---:|---:|
| 原完整失败句 | 0.217 ms | 0.207 ms |
| 既有长句 | 1.009 ms | 1.044 ms |
| 既有标点句 | 0.447 ms | 0.448 ms |
| 已补点的参考 | 0.218 ms | 0.166 ms |
| 实际 Nani 模型分支 | 0.114 ms | 0.109 ms |
| 数字组合 | 0.358 ms | 0.363 ms |

这些是一次顺序实验中的短耗时样本，没有证明稳定加速。官方与自有正常进程导入分别耗时 108.62 / 58.51 ms，实例与入口初始化为 3.42 / 2.27 ms；更早的自有诊断进程首次导入耗时 1.99 秒。导入受缓存和系统状态影响，不能只挑正常进程的较小数值当作完整应用冷启动。

| 正常进程 RSS 边界 | 官方 | 自有 |
|---|---:|---:|
| 导入前 | 27.88 MiB | 27.72 MiB |
| 包导入后 | 73.63 MiB | 68.39 MiB |
| 主 / 用户词典实例初始化后 | 76.31 MiB | 69.89 MiB |
| 全部样例后 | 148.48 MiB | 142.27 MiB |
| 自有实例关闭并 GC 后 | 未执行对应释放 | 131.83 MiB |

这是进程 RSS 快照，OS 生命周期最大值也另存；不是阶段峰值或 GPU 显存。两个环境使用 NumPy 1.26.4 / 2.4.6，不能把小幅 RSS 差异归因为运行时优化。

Nani 的两个 Session 在 pyopenjtalk 导入时创建，Sudachi 的 Dictionary 和线程 tokenizer 由包全局保留。关闭自有 OpenJTalk 实例后仍有约 132 MiB RSS，本轮没有宣称卸载全部日文资源或解决长期生命周期问题。

## 安装与资源

安装前 dry-run 和安装后清单位于 `runs/20260919T132946.638426Z-japanese-install/`。仅新增以下固定 wheel：

```text
pyopenjtalk-plus==0.4.1.post9
SudachiPy==0.6.11
SudachiDict-core==20260723
pydantic==2.13.5
pydantic-core==2.46.5
annotated-types==0.8.0
typing-inspection==0.4.4
```

既有 NumPy 2.4.6、ORT 1.30.0、tokenizers 0.21.4、MLX 0.32.2 及其他包版本均未更改，`pip check` 通过，没有安装 Torch 或 Transformers。新增文件共 344,479,006 字节，约 328.52 MiB；它是本次环境的文件增量，包含生成文件，不是最终分发包的最小体积。

主词典、Sudachi 和 Nani 模型随固定 wheel 提供。自有环境直接使用自己的包内资源，官方用户词典只读引用 `GPT-SoVITS/GPT_SoVITS/text/ja_userdic/user.dict`。对应原 CSV 与 md5 在准备阶段一致。资源总量没有通过删词典、禁用读音修正或省略小模型缩小。

源码保留 GPT-SoVITS、VITS 和 ESPnet 来源。`docs/third-party/` 新增 VITS MIT、pyopenjtalk MIT、OpenJTalk 词典 `COPYING` 和固定 SudachiDict 的 `LEGAL`；已有 Apache-2.0 文本用于 ESPnet / Sudachi。Sudachi 的 LEGAL 含 UniDic、NEologd 及相关来源通知，不能只保留 Apache 文本。

官方用户词典可追到 [PR #1660](https://github.com/RVC-Boss/GPT-SoVITS/pull/1660) 的 LLM 转写 CMUDICT-KATAKANA 和 [#1670](https://github.com/RVC-Boss/GPT-SoVITS/pull/1670) 的导入修复。确切 CMU 数据版本及完整资源通知仍需在分发前补齐；本轮未制作或发布运行包。

## 证据与复现

主目录：`SakuraTTS-References/runs/20260919T133731.746098Z-japanese-g2p/`。

- `prepared.json`、`cases.json`、`source/`：固定源码、全部资源路径 / 大小 / 哈希和原始输入。
- `official-diagnostic.json`、`native-diagnostic.json`：逐句 NJD、labels、音素和 ID，Nani 输入输出、Sudachi 调用、用户词典命中。
- `official-normal.json`、`native-normal.json`：每次正常耗时和 RSS 边界。
- `comparison.json`、`processes.json`、`exit-observation.json`、原始日志：逐项对照、真实命令和退出码。
- `offline-review.py`、`offline-review.json`：源文件与资源身份、之前官方前端结果的独立核对、许可文本哈希和来源。

在项目根目录执行，始终新建运行目录：

```sh
/Users/beyondpower/Documents/Projects/SakuraTTS-References/.venv-mlx-macos/bin/python \
  harness/japanese_equivalence.py prepare

/Users/beyondpower/Documents/Projects/SakuraTTS-References/.venv-mlx-macos/bin/python \
  harness/japanese_equivalence.py run --run <新生成的目录>
```

下一步可以把这个语言段接口接到公共 `clean_text` / 符号表入口，再沿已保存的参考与目标规则验证完整日文条件。尚未覆盖的语言路由、独立原始文本到音频和音频质量仍需分别验收。
