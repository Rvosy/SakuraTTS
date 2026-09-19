# V2Pro 语音问题定位与资源生命周期实验

日期：2026-09-19。平台：Apple M4，16 GiB 统一内存，MPS / FP32。当前只验证“朱雀院红叶”这一组权重，不代表其他 V2Pro 模型或其他版本通过。

## 用户报告与固定输入

用户试听 Lite 后报告：日文缺少开头「こんにちは。」；「今日はいい天気ですね。」中的助词「は」听成 ha；中文开头「你好」听成「哼哼」。这些是待解决的失败现象，不能用音频非空或哈希稳定覆盖。

原始两条文本已逐字保存到 `harness/cases/speech_regressions.json`，另加 8 条短句、长句、标点和混合文本。新样例尚未获得内容质量验收。自动检查、ASR、人工试听分别记录；本轮初始定位时未执行 ASR 或人工试听；后续结果见 [ASR 与用户复听](2026-09-19-asr-review.md)。

模型、开心参考音频、seed 1234、top-k 15、top-p 1、temperature 1、重复惩罚 1.35 和语速 1 与首轮相同。权重、参考音频的实际 SHA-256 随新结果重新保存。官方提交仍为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`，Lite 为 `6c049397142f4c9147a85f86b6ba37546e93a188`。没有修改上游源码或原模型。

## 已确认的差异

| 环节 | 实际证据 | 能说明什么 |
|---|---|---|
| 日文开头与助词 | 两边都保留完整输入；OpenJTalk 把两个「は」转成 `w a` | 不能把失败直接归为前端漏句或输出了 `h a` 音素 |
| 日文音素 | 官方目标 58 个、Lite 56 个；参考分别 47、42 个 | Lite 删除官方保留的 UNK，其中包括韵律 `#`；参考末尾标点也不同 |
| 中文音素 | 官方为短开头补前导句号；对齐该步骤后 phones、word2ph 一致 | 不能忽略上游前导停顿语义 |
| 中文 BERT | 两边实际送入 BERT 的标点不同，CPU 特征最大绝对差 6.6018658；相同官方规范化文本和 word2ph 下特征逐元素一致 | 保持 BERT 启用还不够，实际输入和对齐也必须一致 |
| 参考语义 | 官方将 9600 个零追加到 16 kHz 波形，即 0.6 秒；Lite 是 4800 个，即 0.3 秒；重采样路径也不同 | 两边的参考 token 不能默认相同 |
| 参考声学 | 官方频谱 `center=False` 配合显式 pad；Lite 使用不同中心处理 | 固定语义 token 后仍需对齐声学参考与噪声 |
| 首个生成 token | 用未修改 Lite 函数与固定样本 `[100,101,102,103,104,EOS]`，实际返回 `[101,102,103,104]` | Lite 末尾 `-idx` 切片丢掉 prefill 的第一个 token；约 40 ms 本身不足以解释整句遗漏 |
| 采样与停止 | Lite 早期额外禁用 token 280/486；EOS 限制、检查条件与检查间隔不同 | 同 seed 并不等价，前几步随机张量的维度也不同 |
| 音频裁头 | 实际日文裁 4224 样本（0.132 秒），中文裁 3712 样本（0.116 秒） | 本次不能用裁头单独解释整句消失；低幅度控制样本确实存在裁掉 2 秒的行为 |

前端原始记录保存在 `../SakuraTTS-References/runs/20260919-frontend-audit/`，含 CPU 脚本、音素、OpenJTalk 节点、BERT 输入与特征。控制流实验是 `harness/audit_lite_controls.py`，证据目录为 `runs/20260919T103023.000057Z-lite-control-audit-cpu/`。

## 固定历史 GPT 对照

新增诊断模式保存规范化、音素、BERT、参考 token、GPT 输入、每步原始 logits、采样 token、停止条件、SoVITS 输入和裁头前波形。诊断模式会同步并复制到 CPU，不能将其时间用作正常速度基准。

| 语言 | Lite 自由采样步数 / 输出 token | 官方自由采样步数 / 输出 token | 相同官方历史下比较步数 | GPT logits 最大绝对差 |
|---|---:|---:|---:|---:|
| 日文 | 86 / 81 | 121 / 120 | 121 | 0 |
| 中文 | 141 / 138 | 147 / 146 | 147 | 0 |

`harness/gpt_fixed_history.py` 只加载 GPT，用官方 x、参考 token、BERT 和每一步历史重放 Lite 的 Prefill 与 Decode。上述全部 1025 维 logits 逐元素一致，top-1 也全部一致。这两条样例没有发现 GPT 权重映射、位置编码或缓存解码计算差异。该结论不覆盖自由采样规则、SoVITS、其他精度或其他模型。

原始目录（相对于参考目录）：

- `runs/20260919T102817.486120Z-lite-mps/`：Lite 插桩；波形与原冒烟哈希相同。
- `runs/20260919T102853.549769Z-official-mps/`：官方插桩；波形与原冒烟哈希相同。
- `runs/20260919T103411.671932Z-gpt-fixed-history-mps/`：固定历史逐步数值、差值、SHA-256 与脚本副本。

## 第一项资源改动

`harness/prepared_reference.py` 使用官方原始方法准备参考 token、频谱、参考文本特征与说话人 embedding，保存这些条件，然后释放 CNHuBERT 和 ERes2Net。后续请求复用完全相同的说话人 embedding。该适配器只接受这一个参考张量，换参考时明确报错，尚不是通用缓存加载器或正式运行时。

首轮两条文本各运行 3 次，所有 WAV 与官方对照哈希一致；这说明本轮资源改动没有改变已有输出，不能据此说明官方输出的内容已经通过试听。最后一次中文的 MPS allocated 边界约为 2038.19 MiB，官方普通路径约为 2650.76 MiB。正常计时、重复运行与卸载检查继续补测，暂不据这些探索样本宣布速度收益。

证据：`runs/20260919T103126.665796Z-official-mps/`。该结果包含参考准备时间、准备后释放边界和逐次 WAV。后续普通对照 `runs/20260919T103238.622081Z-official-mps/` 在卸载并 empty_cache 后仍有约 9.17 MiB allocated、183.33 MiB driver；这是进程剩余状态，不承诺归零。

MPS 是统一内存。allocated、driver、RSS 各有不同口径，不能相加，边界快照不是峰值，也不代表 NVIDIA 显存。

## 复现

在项目根目录，已有参考环境下执行。每次创建新结果目录，不覆盖旧结果：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-macos/bin/python" -u harness/reference_smoke.py --references "$REF" --backend lite --device mps --repeat 1 --diagnostic
"$REF/.venv-official-macos/bin/python" -u harness/reference_smoke.py --references "$REF" --backend official --device mps --repeat 1 --diagnostic
"$REF/.venv-macos/bin/python" harness/audit_lite_controls.py --references "$REF"
"$REF/.venv-macos/bin/python" harness/gpt_fixed_history.py --references "$REF" --official-run "$REF/runs/20260919T102853.549769Z-official-mps" --device mps
"$REF/.venv-official-macos/bin/python" -u harness/reference_smoke.py --references "$REF" --backend official --device mps --repeat 4 --prepare-reference
```

新增语料可通过 `--case-ids zh-short ja-short` 等明确选择。完整诊断保存与正常端到端测量分开执行。

## 后续工作

继续做条件消融，分别对齐文本、参考 token 与语义切片，保存候选音频后再检查内容。并行验证中文 BERT 目标输出不依赖的末两层与 MLM head 裁剪；只有数值对照通过后才进入端到端资源测试。自有 GPU 运行时、量化、其他模型与 NVIDIA 实机结论仍未完成。

后续补充：目标/参考文本条件已拆开消融；只替换参考文本条件即可让本例日文开头重新被 ASR 识别，而只替换目标条件仍缺开头。用户对官方与同时对齐文本条件的候选回复“一切正常”。原始失败、其他候选和具体前端差异的进一步定位继续保留。
