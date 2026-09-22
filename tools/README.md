# 离线维护工具

普通使用从 `sakuratts convert`、`sakuratts tts` 和 `sakuratts serve` 进入。这里保留需要在源码目录执行的资源准备、格式转换和图分析工具；它们不由推理服务自动执行。

| 用途 | 工具 |
| --- | --- |
| 准备独立声学解释器 | `prepare_ort_worker_runtime.py` |
| 准备语言资源 | `prepare_japanese_resources.py`、`prepare_chinese_resources.py`、`prepare_english_resources.py`、`prepare_g2pw_ort.py`；原生英文包使用 [`prepare_english_frontend.py`](../docs/english-frontend.md) |
| 转换、重打包历史模型与参考 | `convert_sovits.py`、`repack_weights.py`、`prepare_reference_package.py` |
| 声学图转换与检查 | `convert_sovits_onnx_fp16.py`、`conv_transpose_polyphase.py`、`split_sovits_vocoder.py`、`vocoder_receptive_field.py` |
| 历史合成与官方对照 | `synthesize_japanese.py`、`windows_official_baseline.py` |

依赖实验回放证据的 `package_sovits_chunks.py` 和旧 Mac 参考准备工具 `prepare_japanese_reference.py` 已移到 `research/tools/`，需要 Git checkout。公共模型转换及 HTTP 参考准备使用包内 `_internal/conversion/`，不依赖该目录。
