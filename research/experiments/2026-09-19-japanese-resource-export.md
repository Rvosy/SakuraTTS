# 独立导出日文前端资源

日期：2026-09-19。`tools/prepare_japanese_resources.py` 直接接收固定官方源码、现成用户词典、完整语言识别模型和新输出目录，不再通过文本等价 Harness 读取历史日文运行记录。

工具仅使用 Python 标准库和 Git。它检查官方 HEAD 为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`，并将 `GPT_SoVITS/text/symbols2.py` 与该提交中的文件逐字比较。只有通过检查的独立符号表参与执行；不会导入官方 text 包、中文前端或模型。

输出保持 `sakuratts-japanese-frontend-resources-v1` 格式，仍只有三个资源文件及 manifest：

| 文件 | 字节数 | 来源 |
|---|---:|---|
| `symbols-v2.json` | 6,894 | 固定官方 `symbols2.py`，保持原序列和 JSON 编码 |
| `user.dict` | 21,321,666 | 现成 OpenJTalk 用户词典，直接复制 |
| `lid.176.bin` | 131,266,198 | 本机已验证的完整 fastText 模型，直接复制 |

统一 V2 表中的 732 个符号全部保留，顺序不变。日文入口仍使用原始符号 ID；没有因本轮只处理日文而删掉其他语言的符号或缩小语言识别模型。

完整 `lid.176.bin` 的 SHA-256 固定为 `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e`。缩小版、截断文件或其他内容会被拒绝，不自动下载或替换。用户词典不会重建。目标路径已存在时，无论是否为空都拒绝；原资源和旧包不修改。

manifest 沿用 `format`、`official_commit`、`symbol_source_sha256`、`sources` 和 `files` 字段。普通日文合成 CLI 仍通过相同格式名、官方提交、三个文件名、大小和 SHA-256 检查资源包，无需改变加载方式。OpenJTalk 主词典、Nani 和 Sudachi 等 wheel 内资源仍由日文运行环境提供，本工具没有把它们误算成已经移除的依赖。

## 实际验证

导出进程实际退出 0，14 项离线检查全部通过。新旧包的三个资源逐字节一致；manifest 所需字段、资源大小和散列符合当前合成 CLI 的校验规则。新 manifest 的来源路径记录解析后的真实地址，因此 `lid.176.bin` 的路径由官方目录中的符号链接变为 `models/shared/fast_langdetect/lid.176.bin`。manifest 本身没有声明与旧包逐字相同。

实际导入记录未出现 NumPy、Torch、Transformers、MLX、ONNX Runtime、pyopenjtalk、fastText 或中日文前端。导出使用标准库运行器记录模块列表，不初始化资源或模型。原始输入、旧包和官方 checkout 状态在验证前后相同。

重复输出路径的进程退出 1，保留原始 `FileExistsError`，新包大小及散列未变。提供名称正确但内容无效的 `lid.176.bin` 时进程退出 1，没有创建目标目录。静态编译、`--help` 和 `git diff --check` 也通过。

三个资源共 152,594,758 bytes，manifest 为 1,183 bytes，新包共 152,595,941 bytes，约 145.53 MiB。本轮没有性能测量或资源裁剪；导出入口去除了历史运行目录依赖，没有降低完整语言模型的分发体积。

证据相对于 `SakuraTTS-References/`：

- 运行记录：`runs/20260919T144302.180511Z-japanese-resource-export/`，含源码快照、原命令、实际进程退出码、stdout / stderr、模块导入清单和逐项检查结果。
- 新包：`models/converted/20260919T144302.180511Z-japanese-frontend-resources/`。
- 旧包对照：`models/converted/20260919T141358.883032Z-japanese-frontend-resources/`。
- 新 manifest SHA-256：`3fa316ec2341b1f95ed92ac1bc87d84b39b21d99eebe47e88a6fa7e5033c19a1`。

## 复现

从项目根目录执行；更换 `--output` 为不存在的目录：

```sh
SAKURA_REFS=/Users/beyondpower/Documents/Projects/SakuraTTS-References
python3 tools/prepare_japanese_resources.py \
  --official-source "$SAKURA_REFS/GPT-SoVITS" \
  --user-dictionary "$SAKURA_REFS/GPT-SoVITS/GPT_SoVITS/text/ja_userdic/user.dict" \
  --language-model "$SAKURA_REFS/GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin" \
  --output "$SAKURA_REFS/models/converted/20260919T144302.180511Z-japanese-frontend-resources"
```

导出保留完整语言识别能力和现有文件布局，没有资源缩小收益。本轮不执行文本前端、目标语音生成、ASR 或试听。
