# 英文与日英混合前端

Windows / NVIDIA、V2ProPlus 路径支持 `en`，以及含英文段的 `ja`、`all_ja` 和 `auto`。目标文本与参考转写分别使用 `text_lang`、`prompt_lang`，可以独立选择。英文和日文遵循原版的零 BERT 特征规则。

`auto` 保留原版语言检测结果。目前只有日文和英文处理器；检测到中文、韩文等段落时返回不支持的错误。已知是日文夹英文时，建议使用 `ja`，避免短汉字段被自动检测为中文。

## 准备资源

默认整合包包含日语和英语运行依赖。准备组件内置导出的英语词典、词性模型和发音预测权重；首次转换原始模型时会一起生成日英文本处理资源，无需转换后手动追加，也不会下载 NLTK 数据。

构建准备组件前，用已有的 GPT-SoVITS 环境离线导出一次英语资源：

```powershell
$env:NLTK_DATA = "LOCAL_NLTK_DATA"
LOCAL_PREPARATION_PYTHON tools/prepare_english_frontend.py --export `
  --official-source LOCAL_SUPPORTED_GPT_SOVITS_SOURCE `
  --output LOCAL_ENGLISH_RESOURCES
```

输出目录的父目录须已存在。准备环境须已有英文 G2P、词典缓存、NLTK POS 与 CMUdict 数据。导出读取可信准备环境中的 pickle 数据，推理只加载 JSON 和 NPZ。

运行 `scripts/build_preparation.py` 时传入 `--english-resources LOCAL_ENGLISH_RESOURCES`，构建器将其放到准备组件的 `official/english/`。完整构建命令见[整合包构建](portable-bundle.md)。源码环境直接转换时，在所用 GPT-SoVITS 源码根目录的 `english/` 放置同一份导出资源，并安装 `sakuratts[english]`。

模型配置的 `frontend` 指向生成的目录；若声明 `languages`，使用 `["ja", "en"]`。旧缓存不自动迁移，调试时使用新的缓存目录重新转换。

## 请求

沿用 [API V2 请求](api-v2-guide.md#发起一次合成)，将 `text_lang` 设为 `ja`、`en` 或 `auto`。仍须显式传 `parallel_infer=false`、`batch_size=1`；英文支持不改变并行推理和采样参数限制。参考音频的转写和语言必须与实际内容对应。

资源导出时的固定样例涵盖缩写、所有格、时间、数字和未登录词。音素一致不代表特定权重的英文音色、口音和内容完整性已验收；听感仍取决于模型及参考音频。代码与数据归属见[英文前端来源](third-party/english.md)。
