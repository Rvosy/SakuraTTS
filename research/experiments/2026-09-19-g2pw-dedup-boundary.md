# G2PW 异质长文本的去重边界

日期：2026-09-19。纯输入诊断确认：固定官方提交 `48b1a0169a28582a8984402f82cf438d3bfa6aca` 在超过 510 个内容词元时，按原文分组的去重可能丢弃查询各自的截断序列。保留下来的 `position_ids` 于是可能指向首行中的另一个字。本轮保留默认官方语义，没有修改去重规则，也没有运行模型判断概率或发音影响。

## 调用链和证据

`research/tools/g2pw_dedup_diagnostic.py` 从固定提交提取并执行真实 `_prepare_data`、`prepare_onnx_input` 和 `_predict_with_sentence_dedup`。OpenCC `s2tw`、PyPinyin、词表、排除字集合和默认 16 字上下文均按官方设置。只有 `_predict` 被替换成数组记录器，返回的 `None` 只用于让包装方法完成，不代表模型预测。

样例由官方多音字集合之外的异质汉字组成，在开头、中间、末尾各放一个“重”。这使实际 `_prepare_data` 只生成三个查询，且默认上下文保留整段。原始用户回归句没有改动。

| 内容词元数 | 去重前各查询的 token 行 | 去重后的问题 |
|---:|---|---|
| 509、510 | 三行相同 | 本组未出现输入差异 |
| 511 | 末查询使用不同截断 | 末查询的位置本应指向“重”，首行同位置是“湖” |
| 512、600 | 中间、末查询使用不同截断 | 两个位置在首行分别指向“春”和“湖” |
| 600 个重复“重” | 所有截断行恰好相同 | 对照组未出现差异，说明重复字样例不能覆盖此边界 |

511 字起，官方仍按原始 `texts` 分组，将 `input_ids`、`token_type_ids`、`attention_masks` 从 `[3,512]` 缩为 `[1,512]`，同时保留三个查询的位置、字符 ID 和 phoneme mask。512 / 600 字样例的中间、末查询各有 510 个内容 token 与实际送出的首行不同。

上游 `TextPreprocessor.pre_seg_text` 会在超过 510 字时调用 `split_big_text`，但该函数只按标点拼接。实际调用 `split_big_text(600字无标点文本 + "。")` 得到长度 `[0,600,1]`，没有硬切分长段。因此这一步本身不能排除该边界；本轮没有运行完整 TTS 请求来判断后续是否还会拦截或报错。

六组样例中，自有输入准备器的全部 36 组数组仍与官方一致。本次发现属于官方去重的已知边界，没有证据表明它是原先短句漏句或“你好”错误的原因。

证据目录：`SakuraTTS-References/runs/20260919T124245.772586Z-g2pw-dedup-inputs/`。其中保留每组完整输入、OpenCC 结果、查询、部分拼音结果、去重前六组数组、实际 `_predict` 调用数组、源码快照和资源哈希。未导入 Torch 或 ONNX Runtime；Transformers 仅用于官方 tokenizer。

复现：

```sh
/Users/beyondpower/Documents/Projects/SakuraTTS-References/.venv-official-macos/bin/python \
  research/tools/g2pw_dedup_diagnostic.py
```

## CPU ONNX 的依赖与下一步接口

只读依赖审计保存在 `runs/20260919T124450.661551Z-g2pw-dependency-audit/`。当前候选环境缺少 ONNX Runtime、flatbuffers、protobuf；已有 NumPy 2.4.6 和 packaging 26.3 满足官方 ORT wheel 声明的版本范围。

与官方环境对齐需要增加 `onnxruntime==1.30.0`、`flatbuffers==25.12.19`、`protobuf==7.36.2`。三者在官方环境的文件逻辑大小合计 79,873,136 字节，约 76.17 MiB；这是安装文件参考值，不是压缩包大小或运行内存。ORT wheel 标签为 `cp311-cp311-macosx_14_0_arm64`。最小固定输入推理不需要 `onnx`、Torch 或 Transformers；NumPy 2 的实际 ABI 与执行兼容性仍需运行验证。

最小 `G2PWSession` 接口应接受明确的本地模型路径和有序标签：`run(inputs)` 返回原始概率，`predict(inputs, texts)` 保留默认官方去重并返回标签及置信度。CPU 设置保持 `ORT_ENABLE_ALL`、`ORT_SEQUENTIAL`、`intra_op_num_threads=2`，并沿用已验证的导入前 `ORT_DISABLE_TELEMETRY=1`。大于 510 词元的上述行为作为兼容基线的已知边界保留，不能静默更改。

固定数组推理通过后，G2PW 的完整文本接口还需要接回 OpenCC、特殊多音字排除、单音字与 PyPinyin 回退、查询上下文裁剪、注音到拼音转换。当前 `chinese2._g2p` 直接批量调用 `g2pw._g2pw`；首轮移植应对齐这条实际调用链。后续声母韵母、分词、变调、儿化和 `word2ph` 仍属于完整中文前端的验证范围。

依赖审计复现命令：

```sh
/Users/beyondpower/Documents/Projects/SakuraTTS-References/.venv-official-macos/bin/python \
  research/tools/audit_g2pw_dependencies.py
```

本页记录审计当时的安装状态。后续实际安装与模型推理需各自保存新运行记录。
