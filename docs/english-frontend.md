# 英文与日英混合前端

英文使用 `en`。日文夹英文可以使用 `ja` 或 `all_ja`，目标文本与参考转写分别通过 `text_lang`、`prompt_lang` 选择语言。英文和日文遵循原版的零 BERT 特征规则。

`auto` 保留原版语言检测结果。如果输入是日文夹英文，显式选择 `ja` 可以把短汉字词交给日语处理器。日英包遇到识别为普通话的段落会报告缺少中文处理能力。中文夹英文可用中日英包的 `zh` 模式。[普通话合成与离线资源](chinese-frontend.md)

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

模型配置的 `frontend` 指向生成的目录。带日英前端的模型声明 `["ja", "en"]`，带普通话资源的模型还会声明 `"zh"`。已转换的模型保留原有资源与语言声明。

## 请求

沿用 [API V2 请求](api-v2-guide.md#发起一次合成)，将 `text_lang` 设为 `ja`、`all_ja`、`en` 或 `auto`。`batch_size` 默认是 1，`parallel_infer` 可省略。参考转写应与录音中的台词相同，`prompt_lang` 选择录音实际使用的语言。

资源导出时的固定样例涵盖缩写、所有格、时间、数字和未登录词。音素一致不代表特定权重的英文音色、口音和内容完整性已验收；听感仍取决于模型及参考音频。代码与数据归属见[英文前端来源](third-party/english.md)。
