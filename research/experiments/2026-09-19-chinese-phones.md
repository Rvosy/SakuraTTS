# 中文 V2 音素入口与 BERT 对齐

本轮把固定官方提交 `48b1a0169a28582a8984402f82cf438d3bfa6aca` 的中文段规则迁入 SakuraTTS。新入口 `ChinesePhones` 接收已有 G2PW 实例和离线资源目录，不在运行时读取上游 Python。规范化、词边界、修正词典、变调、儿化、音素及 `word2ph` 都保留官方含义。

10 组中文段输出与官方规则一致；五组真实文本进一步通过了自有 G2PW 与 MLX CPU BERT 的组合验证。进程没有导入 Torch 或 Transformers。这轮没有运行语义生成和波形生成，也没有做 ASR 或人工试听，不能据此宣布“你好”的听感问题已解决。

## 实现范围

官方 `chinese2` 实际调用 `g2pw._g2pw`，不会经过 `UltimateConverter` 的逐字回退。自有入口沿用这个调用关系，不添加新的拼音回退。

处理顺序为规范化、按标点分段、批量 G2PW、`jieba_fast.posseg` 分词、变调前合词、按词切分拼音、词典修正、声韵母转换、变调、儿化、opencpop 映射。标点占一个音素，普通汉字占两个；儿化不删除字符槽位。普通中文段保留 `len(phones) == sum(word2ph)` 和 `len(normalized) == len(word2ph)` 两个约束。

`￥` 和 `^` 的处理顺序以及“将该段所有逗号音素替换成特殊停顿”的行为也按官方 `clean_special` 保留。这个行为看起来容易误解，因此加入了对照样例，没有按直觉改写。

`chinese_bert_features` 接收 `MLXBertFeatures` 和原 tokenizer，将已验证的 `hidden_states[-3]` 编码器输出去掉 CLS/SEP，再按 `word2ph` 扩展为 `[1024, phones]`。中文没有零特征路径。空的规范化片段不能进入这个特征函数；上游官方此处同样无法拼接空特征列表，自有入口给出明确错误。

```python
phones, word2ph, normalized = frontend.clean(chinese_segment)
phone_ids = frontend.phone_ids(phones)
with mx.stream(mx.cpu):
    features = chinese_bert_features(normalized, word2ph, bert, tokenizer)
```

语言识别、参考与目标的标点准备、GPT 输入拼接仍由调用方负责。这不是多语言总入口，也未验证 V1 中文规则。

## 资源、来源与许可

`tools/prepare_chinese_resources.py` 先核对上游文件与固定 Git blob，再导出 JSON。两个修正文件共 45,080 行，最后得到 45,047 个词，发生 33 次覆盖。其中“咖喱”由 `ga1 li5` 覆盖成 `ga1 li2`，说明去重时不能任意选取首项。转换器使用 `ast.literal_eval`；部署过程无需 `eval`、pickle 或写回原始词典缓存。

独立官方 oracle 仍从原始 `.rep` 文件执行官方 `read_dict`，随后逐项核对导出的修正词典、opencpop 表和 V2 符号序列。双方共用同一份 JSON 不足以证明导出正确，这项独立检查已包含在 Harness 中。

规范化的七个模块与 `tone_sandhi.py` 原样保留 PaddlePaddle 的 Apache-2.0 版权头。`chinese.py` 注明 GPT-SoVITS 的 MIT 来源，逐文件 SHA-256 和迁移说明保存在 `docs/third-party/chinese-source-manifest.json`。已有 Apache-2.0 和 GPT-SoVITS 许可正文继续适用。G2PW 表、修正词典和 opencpop 数据的独立来源与再分发许可尚未闭合，目前仅生成本地研究资源。

运行环境仅新增 `jieba-fast==0.53`。其安装包声明 MIT，但 sdist 和已装包均未包含 LICENSE；本轮从包元数据指向的项目取得许可正文，保存了 API 返回、Git blob `9d7e66b431461c785329a1b52199d4207daefacc` 与 SHA-256。副本位于 `docs/third-party/jieba-fast-LICENSE.txt`。

新增包安装前后 `pip check` 均通过，原有包没有改变。native 安装文件合计 25,625,316 字节，约 24.44 MiB，包含 Python、字典、HMM/POS 数据和本机编译扩展。官方与 native 的代码及数据文件相同；本地扩展二进制和安装元数据存在预期差异。这不是最终裁剪后的分发体积。

安装证据：

```text
SakuraTTS-References/runs/20260919T133543.126220Z-chinese-phones-env/
```

## 正确性证据

主证据目录：

```text
SakuraTTS-References/runs/20260919T134452.171531Z-chinese-phones/
```

目录保留源代码快照、资源来源、原始输出、命令、日志及比较结果。之前的 `134236.818620Z-chinese-phones` 是初版探索，资源导出的独立检查以主证据目录为准。

真实模型计算完成后，补充了空片段进入 BERT 前的明确错误，单独验证记录在 `feature-empty-guard/`，非空计算未变。Harness 也补充了缺少已声明 gold 时立即失败，以及在 `compare` 中核对完整链结果；最终脚本和通过结果保存在 `final-research/tools/`。

离线规则对照包括：

- 五组历史真实中文段：短句、长句、标点，以及混合文本中实际划出的两个中文段。
- 用户原始中文失败句，未增删其输入文字；另有 `￥`、`^`、空串和纯拉丁过滤样例，共 10 组。
- 27 组规范化样例，覆盖繁简、全角、日期、时间、温度、号码、分数、百分数、版本号、数学符号和连续标点。
- 45,047 项修正词典直接命中以及三个单字回退，合计 45,050 项结果摘要一致。

随后 `full-native` 用真实 G2PW 重算上述 10 组；对照保存的官方拼音及规则结果，规范化、音素字符串、ID、`word2ph` 和拼音批次全部一致。五组已有官方 BERT gold 的片段均通过原有 `rtol=1e-4, atol=1e-5` 阈值，最大绝对差为 `1.2397766e-5`，没有放宽容差。

| 中文片段 | 字符数 | 音素数 | BERT 最大绝对差 |
| --- | ---: | ---: | ---: |
| 用户失败句在官方 trace 中的片段 | 30 | 56 | 9.5367e-6 |
| 长句 | 104 | 200 | 1.2398e-5 |
| 标点样例 | 32 | 57 | 7.6294e-6 |
| 混合文本中文段一 | 8 | 14 | 5.7220e-6 |
| 混合文本中文段二 | 9 | 17 | 1.1444e-5 |

表格第一行保留官方调用链传入的起始标点。用户未经改写的原文作为独立的 `user-zh-exact` 样例，得到 29 字符、55 音素；这项只验证了音素，没有单独生成 BERT gold 或音频。

## 资源与耗时口径

本轮是带拼音比较、拷贝和结果保存的诊断进程，不是正常端到端基准。G2PW 使用原 ONNX 构造路径，尚未使用随后新增的 ORT mmap 包。先完成并释放 G2PW，再加载 BERT，因此两个模型未同时常驻。

| 执行边界 | 进程 RSS，MiB |
| --- | ---: |
| 初始 | 34.42 |
| G2PW 加载后 | 1524.78 |
| G2PW 调用后 | 1608.67 |
| G2PW 关闭后 | 663.88 |
| BERT 加载后 | 1977.48 |
| BERT 调用后 | 2003.53 |
| BERT 释放并清理 MLX 缓存后 | 839.48 |

这些是 Mac 进程 RSS 执行边界，不能称为阶段峰值或 NVIDIA 显存。OS 进程 lifetime max RSS 为 2003.53 MiB，也不是某一算子的显存峰值。关闭后的 RSS 仍包含 Python、分词器全局缓存、ORT/MLX 等分配器保留，本轮未把它全部归因于某个模块。

G2PW 加载约 0.585 秒，BERT 加载约 0.688 秒。五组正常片段的诊断 G2PW+规则耗时约 8–239 ms，BERT 约 39–98 ms；首次调用包含分词器初始化。没有经过冷启动及多轮热运行的独立基准，因此不将这些数字作为优化收益。

## 复现

在项目根目录执行；模型和大型产物仍保存在参考目录。

```sh
../SakuraTTS-References/.venv-mlx-macos/bin/python research/tools/chinese_equivalence.py prepare \
  --references ../SakuraTTS-References \
  --pinyin-run ../SakuraTTS-References/runs/20260919T132622.077205Z-g2pw-pinyin \
  --bert-run ../SakuraTTS-References/runs/20260919T112854.850786Z-mlx-bert-cpu

../SakuraTTS-References/.venv-official-macos/bin/python research/tools/chinese_equivalence.py worker \
  --run "$CHINESE_RUN" --backend official
../SakuraTTS-References/.venv-mlx-macos/bin/python research/tools/chinese_equivalence.py worker \
  --run "$CHINESE_RUN" --backend native
../SakuraTTS-References/.venv-mlx-macos/bin/python research/tools/chinese_equivalence.py compare \
  --run "$CHINESE_RUN"
../SakuraTTS-References/.venv-mlx-macos/bin/python research/tools/chinese_equivalence.py worker \
  --run "$CHINESE_RUN" --backend native --full-native
```

`CHINESE_RUN` 设置为 `prepare` 输出的新目录。主证据实际使用绝对路径，并直接执行该目录中保存的 Harness 快照；运行命令位于各结果 JSON。

下一步把已验证中文段入口接入完整自有合成请求，并把资源生命周期安排落实到请求流程。Windows/CUDA 的同等能力、语音及显存对照仍待实机验证。
