# 原始日文目标文本入口

本轮新增 `TextFrontend`，把原始日文目标文本接到已验证的 `JapaneseG2P`。它保留固定官方提交 `48b1a0169a28582a8984402f82cf438d3bfa6aca` 的真实语言路由、前导标点、切句、短段合并、末尾标点和短音素重试规则。

当前仅接受 `ja`、`all_ja`。按最新研发范围，中文入口暂缓，已有中文实现和证据保留。新入口不导入或构造 ChinesePhones、G2PW、中文 BERT，也不要求中文资源目录。日文的 `[1024, phones]` FP32 零 BERT 特征来自官方规则。

## 接口和处理顺序

```python
segmenter = LanguageSegmenter(japanese_resource_dir)
japanese = JapaneseG2P(main_dictionary, user_dictionary)
frontend = TextFrontend(japanese, symbols, segmenter)
fragments = frontend.prepare_target(original_text, "ja", split_method="cut0")
```

每个片段返回 `phones`、`bert_features`、`norm_text`，以及保留每个语言段输入、规范化文本、音素 ID 和 `word2ph` 的 `segments`。日文 `word2ph` 为 `None`。组件加载和释放由调用方负责。

处理顺序与官方非流式 `preprocess` 一致：

1. 合并连续的指定标点，去掉首尾换行。
2. 首个标点之前少于 4 字、且文本不以标点开头时，补前导 `。`。
3. 执行 `cut0` 至 `cut5`，过滤空项，合并不足 5 字的片段，补末尾标点。
4. 对大于 510 字的片段调用官方 `split_big_text`。
5. 真实语言路由、日文 G2P、保留 `UNK` 并映射 V2 ID，按原顺序拼接音素和零 BERT。
6. 拼接后少于 6 个音素时，在该片段前加 `.`，仅重做一次。

前导 `。` 与少于 6 音素时的 `.` 是两个独立条件。参考文本不经过目标的前导标点和切句规则；本模块不处理参考音频，也不拼接参考条件。

`ja` 和 `all_ja` 都先经过 `split-lang`、BudouX 和完整 fastText 检测模型。明确指定日文也可能分出英文，不能用正则替代检测。当前遇到英文段会报 `NotImplementedError`，不会将它丢弃或强行送入日文 G2P。语言检测器沿用上游的进程级 detector 配置；目前按每进程一个固定资源配置使用，尚未验证多配置并发切换。

## 实际验证

主要证据：

```text
SakuraTTS-References/runs/20260919T141515.227480Z-japanese-target-text/
```

官方和自有路径在独立进程运行。官方 oracle 使用固定源文件中的原函数，避开整个 TTS 包初始化。其特征操作只需 FP32 零数组和拼接，Harness 用 NumPy 执行，并额外与历史官方 Torch/MPS 保存的数组逐位核对。

- 26 组既有日文探针分别执行默认语言路由和强制日文默认路由，输出一致。空串直接路由会在官方 `split-lang` 中抛出 `IndexError`，两侧同样记录了这个边界。
- 26 组分别作为 `ja/all_ja` 目标输入，共 52 项结果一致。其中三个纯英文探针在两个模式下均明确不支持；空串和纯标点按目标规则处理，不把无输出计为语音通过。
- 四条原始日文回归请求与历史官方 trace 完全一致，包括目标预处理结果、规范化文本、音素 ID、段顺序、`word2ph=None` 和 BERT 数组。

| 原始日文请求 | 音素数 | BERT 形状 | 历史官方对照 |
| --- | ---: | --- | --- |
| 用户报告的三句问候 | 58 | `[1024, 58]` | 逐位相同 |
| `こんにちは。` | 11 | `[1024, 11]` | 逐位相同 |
| 长句 | 267 | `[1024, 267]` | 逐位相同 |
| 引号和标点样例 | 117 | `[1024, 117]` | 逐位相同 |

另有 216 组离线切句/标点检查，覆盖 26 组原探针和空白、数字小数、连续标点、超长无标点等边界，并遍历六个切句方法。1、5、6 音素的合成控制流探针确认：少于 6 音素只重试一次，达到 6 时不重试。离线证据位于先前的 `141358.883032Z-japanese-target-text/offline/`。

首轮真实诊断因 Harness 尚未捕获“空串直接路由”的官方异常而中止，原日志保留在 `141358.883032Z-japanese-target-text/official-diagnostic.log`。随后只修改 Harness 的异常记录，没有改变运行时规则。原请求入口对空文本的处理保持不变。

`split_big_text` 仍有此前确认的官方限制：无标点超长段可能产生空片段和超过 510 字的片段。这次保留并对照该行为，未把这些边界宣称为可成功合成的支持范围。

新进程的 `sys.modules` 中没有 Torch、Transformers、MLX、中文模块、G2PW、BERT、tokenizers、OpenCC、PyPinyin 或 jieba_fast。本轮验证的是日文目标条件，还没有据此生成新音频或做新试听；完整日文请求由后续合成 Harness 验证。

## 独立资源和安装成本

日文资源已复制到独立目录，所有原文件保持不变：

```text
SakuraTTS-References/models/converted/20260919T141358.883032Z-japanese-frontend-resources/
```

运行时可只接收该目录及安装包提供的日文主词典，不需要官方 checkout 或中文资源目录。

| 文件 | 字节数 | 用途 |
| --- | ---: | --- |
| `symbols-v2.json` | 6,894 | 官方 V2 音素 ID 顺序 |
| `user.dict` | 21,321,666 | 官方日文用户词典 |
| `lid.176.bin` | 131,266,198 | 完整语言检测模型 |

目录含来源和 SHA-256 manifest，总计 152,595,967 字节，约 145.53 MiB。它不包含安装包中的 OpenJTalk 主词典、Sudachi 词典和 Nani 模型，因此不是完整运行包体积。研究目录保留了原资源，新增副本用于验证独立部署布局；发布时应只分发一份。

固定路由依赖只新增六包，安装文件共 1,717,413 字节，不含 `.pyc`：`split-lang==2.1.1`、`fast-langdetect==1.0.1`、`fasttext-predict==0.9.2.4`、`budoux==0.9.2`、`robust-downloader==0.0.2`、`colorlog==6.12.0`。全部使用 wheel，原有包未改变，安装前后 `pip check` 均通过。最终 dry-run、安装报告、逐文件清单和包内 LICENSE/NOTICE 路径位于：

```text
SakuraTTS-References/runs/20260919T135758.688564Z-text-router-env/
```

最早 dry-run 曾选用 colorlog 6.9.0；在实际安装前查到官方版本为 6.12.0，随即以 `final-pins.json` 和 `final-dry-run.json` 重新核对。实际安装的六包都与官方版本相同。

`requirements/mlx-japanese.txt` 单独列出当前 Mac 日文合成候选运行时的固定依赖，不含中文和训练栈。它包含用于后续 GPT/SoVITS 的 MLX，但纯文本入口不会导入 MLX。主任务已用同一份文件新建 `.venv-japanese-macos`，安装和 `pip check` 通过，确认缺少中文、Torch 和 Transformers 包，证据位于 `141553.840721Z-japanese-clean-env/`。该环境的完整合成检查由后续 Harness 记录。

代码沿用 GPT-SoVITS 的 MIT 来源声明。新增依赖的许可证随 wheel 安装并已记录哈希；BudouX 和 robust-downloader 为 Apache-2.0，其余四包元数据声明 MIT。fast-langdetect 包另带预训练语言模型的 CC BY-SA 3.0 notice；完整 `lid.176.bin`、日文词典及整包分发的许可材料仍需在发布整理时一并核对。

## 资源与计时边界

诊断进程的 native RSS 为：初始 41.02 MiB，组件构造后 83.17 MiB，全部调用后 290.95 MiB，释放本地 OpenJTalk 实例和语言模型引用后 281.47 MiB。组件构造时语言模型尚未完成首次推理加载，因此不能把 83.17 MiB 当作完整常驻。

这些是 Mac 进程 RSS 执行边界，OS lifetime max RSS 为 290.95 MiB，不能解释为阶段峰值或 NVIDIA 显存。释放后仍有 pyopenjtalk 的共享 Nani/Sudachi 缓存及分配器保留。

四个回归请求在 26 组探针之后再次执行，调用计时约 0.105–1.662 ms，已经受缓存影响。这是诊断记录，不是冷启动或独立正常性能基准，也不用于宣称优化收益。后续正常端到端合成计时应在环境安装结束后单独执行。

## 复现

```sh
../SakuraTTS-References/.venv-mlx-macos/bin/python research/tools/text_frontend_equivalence.py prepare \
  --references ../SakuraTTS-References \
  --japanese-run ../SakuraTTS-References/runs/20260919T133731.746098Z-japanese-g2p \
  --resources ../SakuraTTS-References/models/converted/20260919T141358.883032Z-japanese-frontend-resources

../SakuraTTS-References/.venv-mlx-macos/bin/python research/tools/text_frontend_equivalence.py offline --run "$JAPANESE_TEXT_RUN"
../SakuraTTS-References/.venv-official-macos/bin/python research/tools/text_frontend_equivalence.py worker \
  --run "$JAPANESE_TEXT_RUN" --backend official
../SakuraTTS-References/.venv-mlx-macos/bin/python research/tools/text_frontend_equivalence.py worker \
  --run "$JAPANESE_TEXT_RUN" --backend native
../SakuraTTS-References/.venv-mlx-macos/bin/python research/tools/text_frontend_equivalence.py compare --run "$JAPANESE_TEXT_RUN"
```

`JAPANESE_TEXT_RUN` 取 `prepare` 输出的新目录。证据实际运行保存于目录内的 Harness 快照，命令及退出码写入 `diagnostic-processes.json`。Windows/CUDA、混合英文、流式请求、多配置并发语言模型和新参考音频准备仍不在本轮验收范围。
