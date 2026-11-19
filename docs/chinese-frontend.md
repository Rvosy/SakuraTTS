# 普通话合成与离线资源

## 用 Windows 中日英包合成普通话

中日英完整包（`zh-ja-en`）已经带有 G2PW 多音字模型、中文特征 BERT 和发音词表。准备一对支持普通话的 GPT / SoVITS 权重，以及一段参考录音和对应的转写。按[快速开始](../README.md#快速开始)配置模型并启动服务。

下面的 PowerShell 示例生成 `mandarin.wav`。请将 `ref_audio_path` 和 `prompt_text` 换成你自己的录音路径和实际台词。参考语言 `prompt_lang` 与目标语言 `text_lang` 分别设置。

```powershell
$request = @{
    text           = '今天下午三点，我们在图书馆见。'
    text_lang      = 'zh'
    ref_audio_path = 'D:/Voices/reference-zh.wav'
    prompt_text    = '这是我的参考录音。'
    prompt_lang    = 'zh'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:9880/tts' -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($request)) -OutFile 'mandarin.wav'
```

`zh` 适用于普通话以及中文夹英文的文本。`all_zh` 指定以中文为分段的默认语言，`auto` 根据内容识别各段语言。短汉字词可能同时出现在中文和日文中，已知语言时直接指定 `zh` 或 `ja`。日英包遇到需要按普通话读的段落时会报告缺少中文处理能力。

## 从本地 GPT-SoVITS 目录准备资源

以下步骤用于源码安装和构建整合包。在 SakuraTTS 源码根目录，用安装了 `convert` 依赖的 Python 运行导出工具。上游目录需要已有 `GPT_SoVITS/text/G2PWModel/` 中的模型和字表、`GPT_SoVITS/text/g2pw/` 中的发音修正表，以及 `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/` 中的权重、配置和分词器。

```powershell
$official = 'D:/Models/GPT-SoVITS'
python tools/prepare_chinese_frontend.py --official-source $official --output "$official/chinese"
```

输出目录须为新目录。导出器把 G2PW 和 BERT 的 FP32 大权重分别写入 `g2pw.bin` 和 `bert.bin`，与同名 `.onnx` 图一起列入资源清单。迁移时复制整个 `chinese/` 目录。已有内嵌权重的 ONNX 模型可继续使用。重新导出并从原始 GPT / SoVITS 权重转换，即可使用新的内存布局。[普通话前端资源来源](third-party/mandarin-resources.md)

中文前端在 CPU 上运行，GPT 和 SoVITS 使用所选的推理档位。中文 BERT 处理文本特征，只有句中出现需要模型判音的字才加载 G2PW。两个会话在后续请求中复用，随前端关闭而释放。[Windows 内存与耗时实测](../research/notes/windows-mandarin-frontend-memory-20261120.md)

源码运行环境在所选后端依赖之外安装项目的 `english`、`chinese` extras，例如在仓库根目录运行 `python -m pip install ".[english,chinese]"`。按[英文前端说明](english-frontend.md)备好同一上游目录下的 `english/`。使用这套资源从原始权重转换模型时，转换器会将两个语言目录加入模型前端。已有的日英转换模型可从原始权重重新转换到新目录。通过 HTTP 配置原始权重时，服务会按资源清单选择转换缓存。

命令行转换时，如需在模型中预置普通话参考，在 `sakuratts convert` 命令的 `--reference`、`--reference-text` 旁加上 `--reference-language zh`。使用该模型生成中文音频时指定 `sakuratts tts MODEL --language zh --text "今天下午三点见。" --output mandarin.wav`。整合包构建与分发范围见 [Windows 整合包](portable-bundle.md)。
