# Mac 日文研究入口

Apple Silicon 的 MLX 实现用于复现日文 V2Pro 实验，尚未接入公共 Engine。Windows 产品用法见[部署指南](setup-windows-nvidia.md)。

研究需要 Git checkout、[Mac 实验环境](development.md#apple-silicon-日文实验环境)和四个独立资源包：GPT、SoVITS、日文前端、参考条件。使用自己的资源目录执行：

```sh
.venv/bin/python tools/synthesize_japanese.py \
  --frontend-package models/frontend \
  --reference-package models/reference \
  --gpt-package models/gpt \
  --sovits-package models/sovits \
  --text 'こんにちは。今日はいい天気ですね。' \
  --seed 0 \
  --output outputs/japanese/speech.wav
```

输出为 WAV 和同名 JSON，目标路径须不存在。JSON 记录输入、资源身份、实际停止原因、片段与计时。参数定义见 [synthesize_japanese.main](../tools/synthesize_japanese.py)，调用与资源释放约束见[推理契约](specs/inference-contract.md)。

前端资源可由 `tools/prepare_japanese_resources.py` 从固定官方源码、词典和完整语言识别模型导出；参考准备使用 `research/tools/prepare_japanese_reference.py`，需要独立开发环境。各工具的 `--help` 列出输入要求。

具体模型、精度、默认参数、可搬迁部署和当时的体积测量保存在[2026-09-19 至 20 日运行记录](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/japanese-runtime-20260920.md)。固定回放、独立随机生成与人工试听分别有记录，数值失败仍保留在[历史兼容性证据](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/compatibility-evidence-20260920.md)。
