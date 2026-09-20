# 官方中日文前端的依赖与移植边界

日期：2026-09-19。范围：现有“朱雀院红叶”V2Pro 的中日文、两条中英 / 日英混合回归输入。本文是只读调用链审计和轻量 tokenizer 检查，没有移植整个前端，也没有新增语音质量结论。

## 来源与本轮证据

官方源码固定为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`。已核对 Git HEAD 和工作区状态；只有既有 `polyphonic.md5` 变化，以及生成的日文 `user.dict`、`userdict.md5`，相关 Python 源码未修改。不能把 Git 提交号当成运行资源版本的全部：第三方包、词典及模型也需要固定。

新增证据在参考目录 `runs/20260919T115747.718856Z-text-frontend-dependencies/`：

| 文件 | 内容 |
|---|---|
| `static-audit.json`、`audit_static.py` | 源码哈希、实际依赖版本及许可证位置、资源文件大小；未读取大模型内容 |
| `mlx-import-check.json`、`check_mlx_imports.py` | 无 torch / Transformers 的现有 MLX 环境中，实际导入成功与失败记录 |
| `tokenizer-adapter-check.json`、`check_tokenizer_adapter.py` | 独立 Rust tokenizer 与 AutoTokenizer 的字符映射、G2PW 输入打包对照 |

文本及音素的既有对照来自 `runs/20260919-frontend-audit/official-frontend.json` 和 `runs/20260919T110311.350332Z-official-mps/*-trace.json`。本轮没有安装依赖、下载资源或加载 BERT / G2PW / 语言识别模型。涉及上游模块的导入检查已关闭 bytecode 写入，未更新官方词典缓存。

## 实际调用链

现有 Harness 使用 `cut0`、非流式、非并行分批。对应官方路径为：

```text
TTS.run
  ├─ 参考文本：去首尾换行 → 补末尾句号 → segment_and_extract_feature_for_text
  └─ 目标文本：preprocess
       → replace_consecutive_punctuation
       → pre_seg_text（前导标点、cut0、合并短片段、补句号、长文本切分）
       → segment_and_extract_feature_for_text
共同段处理：get_phones_and_bert
  → 空格压缩 → LangSegmenter → 各语言 clean_text_inf
  → clean_text → 规范化 / G2P → 未收录音素映射 UNK
  → cleaned_text_to_sequence（V2 音素表）
  → 中文 BERT；其他语言生成同宽零特征
  → 按段拼接 phones / BERT / normalized text
  → 少于 6 个音素时加前导点并重做一次
TTS.to_batch：参考 phones + 目标 phones；参考 BERT + 目标 BERT
```

主要源码入口是 `GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py`、`TTS.py` 的 `run` / `to_batch`、`text/cleaner.py` 和 `text/__init__.py`。

### 前导标点、参考和目标不能合并成同一种处理

目标 `pre_seg_text` 判断原文本是否以 `splits` 中的符号开头；若不是，且首个标点之前的内容少于 4 个字符，则中文 / 日文等语言补 `。`，英文补 `.`。随后才应用切句策略、过滤空片段、合并不足 5 字的片段、补末尾标点，并处理大于 510 字的文本。

固定中文的首段是两个字“你好”，因此有效输入为 `。你好，欢迎使用樱花语音。现在正在测试苹果电脑上的语音合成。`，规范化后为 `.你好,欢迎使用樱花语音.现在正在测试苹果电脑上的语音合成.`，共 56 个音素。前导点必须保留，不能视作装饰标点去掉。

固定日文以 5 字“こんにちは”开头，不触发上述规则。它保留 58 个音素，其中 2 个 `UNK`。另有独立规则：整段拼接后音素少于 6 个时，`get_phones_and_bert` 给文本加 `.` 并重做一次；这与“开头少于 4 字”的规则不是同一条件。

参考文本经过末尾补点，直接进入共同段处理，不调用目标 `preprocess`。现有参考 `じゃあ、私、もっと悪い子になっちゃおうな〜` 会变为 `じゃあ、私、もっと悪い子になっちゃおうな〜。`，输出 47 个音素，含 4 个 `UNK`。移植时应保留两个简单的参考 / 目标入口，共用语言分段、G2P 和特征逻辑；不要给参考文本额外套用目标前导标点或切句流程。

官方非流式与流式入口的标点预处理也不完全相同。本轮只对已记录的非流式路径给出结论。

### 中文：G2PW 和特征 BERT 是两个模型

V2 的 `cleaner.clean_text` 选择 `text/chinese2.py`。实际链路是 PaddleSpeech 来源的 `TextNormalizer`、官方标点替换、按标点分段、`G2PWPinyin`、`jieba_fast.posseg`、多音字修正词典、变调 / 儿化处理，最后通过 `opencpop-strict.txt` 映射音素。`word2ph` 逐规范化字符记录展开数量，同时检查 `sum(word2ph) == len(phones)` 和 `len(word2ph) == len(normalized_text)`。

`chinese2.py` 将 `is_g2pw` 固定为 `True`，在导入时构造 ONNX Runtime Session。G2PW 权重文件为 635,212,732 字节（605.79 MiB）。它使用 OpenCC、单音 / 多音字表、注音映射和 PyPinyin 回退；不能通过关闭 G2PW 或统一使用 `lazy_pinyin` 来宣称保留官方能力。

固定提交已经包含 G2PW 句子去重、输入词元缓存、`g2pw_polyphonic_context_chars=16` 默认上下文裁剪和 2 线程 CPU Session。这些规则已有实现，不应作为 SakuraTTS 新增收益重复计算。上下文截取从首个多音字之前 16 字延伸到最后一个多音字之后 16 字，不是每个查询独立截取 32 字。

G2PW 的 `dataset.py` / `utils.py` 已使用 NumPy 组装输入，没有 torch 依赖。Transformers 用在 `onnx_api.py` 的 `AutoTokenizer`；最小替换只需要相同的 `tokenize(word)` 和 `convert_tokens_to_ids(tokens)`，并保留官方字符到 token 的映射与截断逻辑。

用于 GPT 条件的中文特征 BERT 是另一条链路：规范化文本 → tokenizer 加 CLS / SEP → `hidden_states[-3]` → 去 CLS / SEP → 按原 `word2ph` 重复 → `[1024, phones]`。此前验证的 MLX CPU 22 层实现只替代这一条特征计算，不包含 G2PW。

### 日文：保留发音字段、韵律与 UNK

`japanese.py` 使用 `pyopenjtalk.run_frontend` → `make_label` → `pyopenjtalk_g2p_prosody`。保留 `with_prosody=True`；每个日文子句去掉边界 `^` / `$` 后，保留 `[`、`]` 和词组边界等符号。随后 `cleaner` 把不在 V2 表中的符号映射成 `UNK`，不删除它们。

V2 音素表包含 `[`、`]`，但不包含 `#`，因此已有日文中的部分韵律边界成为 `UNK`。此前 Lite 的删 `UNK` 行为改变了目标和参考长度，不能照搬。

既有 OpenJTalk 节点记录中，“は”的 `read` 是 `ハ`，`pron` 是 `ワ`。应沿用 label / pron 路径，不把 `read` 字段直接当发音。此事实不能替代人工试听，也不能据此断定用户听到 ha 的问题已经修复。

实际安装的是 **pyopenjtalk-plus 0.4.1.post9**，不是依赖名表面上的原版 pyopenjtalk。其默认参数包括：`use_vanilla=False`、`use_sudachi_kanji_yomi=True`、`predict_nani=True`、`use_read_as_pron=False`、`run_marine=False`、`use_tsqyomi=False`。Sudachi 的读音修正属于当前基线。`nani_predict.py` 还在导入时创建两个小 ONNX Session；缺少 ORT 时会退化为固定预测，禁用“何”的上下文读音能力。当前样例未覆盖该字，不能因这几句不受影响就省略依赖。

日文需要固定主词典、Sudachi 词典和官方用户词典。上游导入时可能重编用户词典并写 md5，异常又会被捕获而继续运行；自有包应在准备阶段产出相同词典并检查身份，推理阶段直接读取。未加载用户词典不能冒充同条件结果。

### 混合文本需要保留真实语言切分

`LangSegmenter` 使用 `split-lang`，后者使用 `fast-langdetect` 的 `model="full"`、fastText、BudouX 和 Pydantic。`all_zh` / `all_ja` 也先执行切分，英文片段仍可保持英文，不能把模式理解成逐字强制转成某一种语言。

已保存的 `auto` 结果如下：

| 样例 | 规范化后的段顺序和音素数 |
|---|---|
| 中英混合 | `.你好,欢迎使用`（zh，14）→ `Sakura T T S. Please check the audio. `（en，27）→ `现在继续播放中文.`（zh，17） |
| 日英混合 | `こんにちは。`（ja，11）→ `Welcome to Sakura T T S. `（en，21）→ `今日は音声を確認します。`（ja，31） |

英文段还需要 `english.py`、`en_normalization/expend.py`、g2p-en、NLTK POS / CMUdict 资源、wordsegment、inflect、官方热词 / 名字词典，以及 g2p-en 自带预测权重。它们的计算不依赖 torch，但并非只有一张静态字典。g2p-en 在找不到 NLTK 资源时会尝试下载，因此需要预先完整提供资源。

## 可移植部分与实际依赖

| 部分 | 最小保留内容 | 当前可去掉的外围内容 |
|---|---|---|
| 入口控制 | 参考 / 目标规则、切句、短段合并、语言路由、音素 ID 和顺序 | TTS 模型加载、音频库、进度条、i18n 打印 |
| 中文规范化 / 音素 | `chinese2` 的实际规则、`zh_normalization`、`tone_sandhi`、`jieba_fast`、PyPinyin、G2PW 及词典 | `TextPreprocessor` 中仅导入未使用的旧 `chinese` 模块 |
| G2PW tokenizer | 同源 `tokenizer.json`、独立 `tokenizers`、官方 NumPy 输入打包 | `AutoTokenizer` / Transformers 依赖 |
| 特征 BERT | 同 tokenizer、已通过的 MLX CPU 22 层、原 `word2ph` 展开 | 完整 MLM 和 PyTorch |
| 日文 | 固定 pyopenjtalk-plus、主 / 用户 / Sudachi 词典、默认读音处理及小型 ORT 模型 | 未启用的 marine / tsqyomi 扩展；不能删当前默认功能 |
| 混合语言 | 固定 split-lang / fast-langdetect / fastText / BudouX 及英文前端 | 不复用整个 WebUI / TTS 环境 |

还有两处可以在移植时删除的未使用导入：`chinese2.py` 的 cn2an 只出现在未调用的 `normalizer` lambda；`LangSegmenter` 中普通 jieba 只用于设置日志级别，实际中文分词使用 jieba_fast。删除这些外围依赖需要对移植后的前端运行原回归，而不是改写上游参考源码。

现有 MLX 环境中，`text` 和 `text.cleaner` 可以直接导入。按包路径导入 `TTS_infer_pack.text_segmentation_method` 却失败，因为包 `__init__.py` 先导入完整 TTS，再要求 torchaudio；隔离加载这个纯文本文件的 `cut0` 检查通过。`zh_normalization` 也不是完全零依赖：`constants.py` 导入 `pypinyin.constants.SUPPORT_UCS4`，现有 MLX 环境缺 PyPinyin，实际导入失败。其余 G2P / tokenizer 依赖当前均未安装到 MLX 环境，不能声称前端已经独立运行。

## tokenizer 替换的轻量实测

本轮从固定官方源码提取实际 `wordize_and_map`、`tokenize_and_map` 和 `prepare_onnx_input` 等函数，避免触发包初始化和模型加载。用独立 `tokenizers.Tokenizer.from_file()` 适配所需两个接口，对照原 AutoTokenizer。

18 组输入包括 10 条固定语料、5 段官方规范化中文，以及字母数字 / 未知 token / 600 字长文本边界。所有 token 和字符映射完全相同。17 组包含多音字查询的输入，6 项 G2PW 数组也完全相同：`input_ids`、`token_type_ids`、`attention_masks`、`phoneme_masks`、`char_ids`、`position_ids`，包含超 512 token 截断。

该参考进程设置 `USE_TORCH=0`、`USE_TF=0`、`USE_FLAX=0`，实际未导入 torch；Transformers 只用于 tokenizer 对照。候选适配器只使用 tokenizers。该结果支持接口替换，但没有运行 G2PW 推理，也不覆盖全部文本或词典行为。

## 文件体积与许可证

下列大小是本地文件大小，既不是运行内存，也不是最终可裁剪体积：G2PW ONNX 605.79 MiB、语言识别 `lid.176.bin` 125.19 MiB、OpenJTalk 主词典约 102.39 MiB、Sudachi 主词典 207.39 MiB、官方用户词典 CSV 16.39 MiB / 编译产物 20.33 MiB。G2PW 和语言识别资源不能重复打包原 ZIP、转换副本及缓存后再声称是最小运行包。

| 来源 / 依赖 | 本轮查到的许可依据 | 分发时需要保留或补齐 |
|---|---|---|
| GPT-SoVITS 本仓源码 | 根 `LICENSE` 为 MIT | 版权和许可文本；修改来源和固定提交 |
| PaddleSpeech 来源的中文归一化、变调、G2PW `dataset.py` / `utils.py` | 文件头明确 Apache-2.0 | 原版权、许可证及适用 NOTICE，标明修改 |
| 日文韵律代码 / G2PW 来源 | 源码注明 ESPnet、VITS、GitYCC/g2pW、pypinyin-g2pW 等来源 | 本轮未补齐所有原上游许可文件；不能只用 GPT-SoVITS 根 MIT 代替 |
| tokenizers 0.21.4、g2p-en 2.1.0、NLTK 3.10.3、wordsegment 1.3.1、BudouX 0.9.2 | 已安装包的许可字段 / 文件为 Apache 类许可；BudouX 明确 Apache-2.0 | 各包许可证、模型 / 词典来源及适用 NOTICE |
| PyPinyin 0.55.0、jieba_fast 0.53、split-lang 2.1.1、fast-langdetect 1.0.1、fasttext-predict 0.9.2.4、Pydantic 2.13.5、inflect 7.5.0 | 元数据为 MIT | 各自版权和许可文本 |
| opencc-python-reimplemented 0.1.7 | Apache 许可，`opencc/NOTICE.txt` 标注 OpenCC 字典来源 | LICENSE、NOTICE 及字典身份 |
| pyopenjtalk-plus 0.4.1.post9 | 包 `LICENSE.md` 为 MIT；词典 `COPYING` 含 NAIST / UniDic / Open JTalk 的 BSD 风格条款 | 同时保留代码和词典条款，不能只保留 wheel 的 MIT |
| SudachiPy 0.6.11、SudachiDict-core 20260723 | 包元数据为 Apache-2.0 | 字典包内许可及来源；固定实际词典版本 |
| CMUdict 0.7a | 本地 NLTK `corpora/cmudict/README` 包含 CMU 的两条保留通知条件 | 随资源保留 README 中的版权、条件和免责声明 |
| ONNX Runtime 1.30.0 / NumPy | 包元数据分别为 MIT / BSD | 各包及其第三方组件许可 |

G2PW 模型目录、独立 `lid.176.bin`、官方用户词典、`opencpop-strict.txt`、修正词典及本地 BERT 模型目录，没有在这次审计范围内建立完整的资源再分发许可链。模型与数据许可不能由代码包许可推断；这部分是打包前的具体待办，不阻止当前本地兼容实验。

现有环境还有 `g2p-en` 声明依赖的 Distance 0.1.3，元数据标为 GPL。当前检查的 g2p-en Python 源码没有导入 Distance，但常规完整安装会带入它。最终依赖清单应按实际运行引用核对，不能把整个现有环境当作已完成许可审查的运行包。

## 最小落实顺序

1. 在 SakuraTTS 增加一个简单文本入口，只移入固定版本的文本规则及所需资源引用，保留原版权；分别暴露参考和目标处理，共用段处理。用 NumPy 表达音素和特征，不导入 `TTS_infer_pack` 包。
2. 先落地已对照的 tokenizers 接口，接入已通过的 MLX CPU BERT。G2PW 使用当前已验证的固定 CPU 执行方式和原模型，不搭建后端选择框架；需要统一计算库时再单独转换并验证 G2PW。
3. 日文固定 pyopenjtalk-plus 和默认选项，预先准备词典。保留 Sudachi / “何”预测；缺资源时报告具体缺失，避免用不同读音能力作为基线。中英 / 日英混合沿用实际语言识别和英文路径。
4. 用固定 10 条目标文本和同一参考文本，对照分句、语言段、规范化文本、原音素、ID、`word2ph`、两种 tokenizer 输入及最终 BERT 特征；再把相同条件送入已经建立的 GPT / SoVITS 对照。新增“何”、多音字、数字、未知符号和长文本边界作为扩展样例，不修改原失败句。

当前最明确的实施点是拆开外围导入和 tokenizer 依赖。G2PW、Sudachi、语言识别及英文资源仍需要按真实能力保留；前端整合完成前，不能宣布原始文本到音频已经摆脱官方运行环境。
