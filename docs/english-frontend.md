# 英文与日英混合前端

Windows / NVIDIA、V2ProPlus 路径支持 `en`，以及含英文段的 `ja`、`all_ja` 和 `auto`。目标文本与参考转写分别使用 `text_lang`、`prompt_lang`，可以独立选择。英文和日文遵循原版的零 BERT 特征规则。

`auto` 保留原版语言检测结果。目前只有日文和英文处理器；检测到中文、韩文等段落时返回不支持的错误。已知是日文夹英文时，建议使用 `ja`，避免短汉字段被自动检测为中文。

## 准备资源

旧日文资源包可以继续使用；遇到英文段会明确报错。启用英文需要额外依赖和带英文数据的新前端包，不会自动下载 NLTK 数据或修改旧包。

在 SakuraTTS 运行环境中安装 `sakuratts[english]`；源码环境使用：

```powershell
python -m pip install -e ".[english]"
```

用原版 GPT-SoVITS 的可信准备环境导出资源。这个环境须已有可用的英文 G2P、词典缓存、NLTK POS 与 CMUdict 数据，源文件版本由工具核对。运行下列命令的主 Python 也须安装英文可选依赖，以便在生成后核对音素：

```powershell
python tools/prepare_english_frontend.py `
  --frontend models/my-voice/frontend `
  --output models/my-voice/frontend-ja-en `
  --official-source D:/GPT-SoVITS `
  --python D:/GPT-SoVITS/runtime/python.exe
```

工具复制已校验的日文资源，增加英文词典、词性模型和发音预测权重。它对照原版检查规范化文本与音素序列，通过后才生成目标目录；不会覆盖已有输出。导出阶段读取原版环境中的可信 pickle 数据，推理端只加载 JSON 和 NPZ。

把模型配置中的 `frontend` 改为新目录；若模型声明了 `languages`，可设为 `["ja", "en"]`。路径格式见[模型目录](model-format.md)。新参考音频准备仍可复用该前端包。原始权重首次转换生成的前端目前只有日文，转换后按上述步骤增加英文资源。

整合包构建时，在自用 recipe 的 `languages` 中加入 `en`，以包含英文运行依赖，并使用上述前端包。默认日文 recipe 和已构建的旧包不会自动获得英文能力。

## 请求

沿用 [API V2 请求](api-v2-guide.md#发起一次合成)，将 `text_lang` 设为 `ja`、`en` 或 `auto`。仍须显式传 `parallel_infer=false`、`batch_size=1`；英文支持不改变并行推理和采样参数限制。参考音频的转写和语言必须与实际内容对应。

资源导出时的固定样例涵盖缩写、所有格、时间、数字和未登录词。音素一致不代表特定权重的英文音色、口音和内容完整性已验收；听感仍取决于模型及参考音频。代码与数据归属见[英文前端来源](third-party/english.md)。
