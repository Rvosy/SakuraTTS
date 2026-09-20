# 独立 tokenizer 接口与官方输入对照

日期：2026-09-19。`src/sakuratts/frontend/tokenizer.py` 已提供 G2PW 所需的两个 tokenizer 接口，以及中文特征 BERT 的单句输入编码。候选环境没有安装 PyTorch、Transformers；实际验证进程也没有导入二者。本轮没有加载 G2PW 或 BERT 模型。

## 实现和资源来源

`ChineseBertTokenizer` 接收本地 `tokenizer.json` 路径。规范化、词表、WordPiece 和特殊词元均来自该文件，不重新训练词表，也不修改输入文本。实例保留实际加载路径及 SHA-256，便于请求证据引用。

当前对照资源为参考目录下的 `models/shared/chinese-roberta-wwm-ext-large/tokenizer.json`，SHA-256：

```text
173796956820ea27bd14f76bf28162607ff4254807e2948253eb5b46f5bb643b
```

接口范围：

| 接口 | 行为 |
|---|---|
| `tokenize(text)` | 不额外添加 CLS / SEP，供 G2PW 的字符映射调用 |
| `convert_tokens_to_ids(tokens)` | 接收单个词元或列表；词表外词元返回同源 `[UNK]` 的 ID，保留 ID 为 0 的 `[PAD]` |
| `encode_features(text)` | 返回形状 `[1, tokens]`、类型 `int64` 的 `input_ids`、`token_type_ids`、`attention_mask`；按资源配置加入 CLS / SEP |

接口不做 padding 或截断，与本轮官方 `AutoTokenizer` 单句默认调用一致。G2PW 的上下文窗口和 512 词元截断仍由其输入准备函数负责。特征 BERT 超出位置长度的输入应由上游分句和模型长度检查处理，不能在 tokenizer 中静默丢弃尾部文本。

## 固定样例验证

`research/tools/tokenizer_equivalence.py` 分为两个进程。`prepare` 使用固定官方提交 `48b1a0169a28582a8984402f82cf438d3bfa6aca` 的 G2PW 纯函数和 `BertTokenizerFast` 保存预期结果；`validate` 只使用自有 tokenizer 接口。两者均关闭框架加载并使用本地资源。

为避免官方包初始化加载模型，Harness 从实际 `utils.py`、`dataset.py` 中提取指定纯函数，保留原文件、哈希与多音字表。候选阶段复用这些已保存函数，只替换 tokenizer 接口。因此本轮证明的是接口替换，尚未证明 G2PW 输入准备已移植到自有运行时。

样例包括原有 10 条固定回归输入、5 段已保存的官方规范化中文，以及 3 条附加边界输入。附加输入覆盖中英数字混排、重音字符与 `[UNK]`、600 个连续“重”字，原失败句没有改动。

| 检查 | 结果 |
|---|---|
| 原始词元、ID、字符与词元双向映射 | 18 / 18 完全一致 |
| 特征 BERT 三组输入数组 | 54 / 54 类型、形状、元素完全一致 |
| 有多音字查询的 G2PW 六组输入数组 | 102 / 102 类型、形状、元素完全一致，覆盖 17 条输入 |
| 特殊词元及词表外词元 ID 回退 | 完全一致 |
| 候选进程导入 Torch / Transformers | 均为 false |

600 字样例的特征 BERT 编码长度为 602，未截断；G2PW 对开头、中间、末尾三个查询分别截取，结果为 `[3, 512]`。这只验证输入处理，不能据此宣称 BERT 模型支持 602 词元。

证据目录均位于 `SakuraTTS-References/runs/`：

- `20260919T122122.560614Z-tokenizer-equivalence/`：官方输入、原始数组、映射、源码和资源快照、版本、命令与哈希。
- `20260919T122152.133861Z-tokenizer-candidate/`：候选数组、逐项比较、自有源码快照及运行结果。
- `20260919T121959.537365Z-tokenizer-install/`：安装前后 distribution 清单、dry-run、实际安装命令、wheel 来源与哈希、安装日志和 `pip check` 结果。

## 依赖和轻量测量

候选环境原先未安装 `tokenizers`。本轮按正常 wheel 依赖安装 `tokenizers==0.21.4`，共新增 14 个 distribution，文件逻辑大小合计 30,248,999 字节，约 28.85 MiB。其中 `tokenizers` 本身约 7.79 MiB；其声明的依赖还包含 Hugging Face Hub、hf-xet 等包。没有更新已有包，`pip check` 通过，没有下载模型。

`requirements/mlx-candidate.txt` 已固定 tokenizer 版本。这是候选环境的增量，不是最终安装包大小；完整依赖版本保存在本轮安装证据中。

Harness 在保存正确性结果之外，另做 2 次热身和 5 次计时。每轮包含全部 18 条文本的词元接口、G2PW 字符映射和输入打包、BERT 输入编码，不含模型推理、哈希或文件写入。

| 本轮诊断数据 | 官方 tokenizer 进程 | 候选进程 |
|---|---:|---:|
| tokenizer 加载 | 9.73 ms | 9.15 ms |
| 整批接口调用中位耗时 | 2.329 s | 2.157 s |
| 进程生命周期最大 RSS | 152.64 MiB | 101.34 MiB |

这些时间包含重复的 G2PW mask 准备及 Python 结果分配；它们不是正常 TTS 请求延迟。两进程的 NumPy 版本分别为 1.26.4 和 2.4.6，测量期间其他研发验证可能并行，不能把差值单独归因于 tokenizer 替换。RSS 包含导入和整个 Harness，不是空闲常驻、生成峰值或 GPU 显存。

## 复现

在 SakuraTTS 项目根目录执行：

```sh
/Users/beyondpower/Documents/Projects/SakuraTTS-References/.venv-official-macos/bin/python \
  research/tools/tokenizer_equivalence.py prepare \
  --official-root /Users/beyondpower/Documents/Projects/SakuraTTS-References/GPT-SoVITS \
  --source-model /Users/beyondpower/Documents/Projects/SakuraTTS-References/models/shared/chinese-roberta-wwm-ext-large \
  --input-provenance /Users/beyondpower/Documents/Projects/SakuraTTS-References/runs/20260919T112854.850786Z-mlx-bert-cpu/input-provenance.json

/Users/beyondpower/Documents/Projects/SakuraTTS-References/.venv-mlx-macos/bin/python \
  research/tools/tokenizer_equivalence.py validate \
  --run /Users/beyondpower/Documents/Projects/SakuraTTS-References/runs/20260919T122122.560614Z-tokenizer-equivalence
```

`prepare` 和 `validate` 都创建新运行目录。第二条命令可复查本轮已保存对照；重新准备后，也可改用第一条输出的新路径。

下一步适合迁移 G2PW 的纯 NumPy 字符映射、查询截断和数组打包，并保留来源注释及许可证。继续以这批数组逐项对照后，再接原 G2PW 模型的 CPU 推理；不能用简单拼音替代多音字模型。本轮没有新音频、ASR 或人工试听证据，也没有改变现有语音回归结论。
