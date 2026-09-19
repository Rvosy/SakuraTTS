# 日文运行入口

当前可在 Apple Silicon Mac 上，从原始日文文本生成完整 PCM16 WAV。已验证的模型是“朱雀院红叶”V2Pro；Windows / CUDA、其他模型、流式和混合英文尚未验收。中文新增整链与优化暂缓，已有源码和回归证据保留。

日常合成使用四个独立目录：GPT 推理包、SoVITS 推理包、日文前端资源包、单参考条件包。不读取历史实验的目标特征、token 或随机数，也不需要官方 checkout、Torch、Transformers 或中文 BERT。首次转换模型和准备参考仍由开发环境完成。

## 生成日文

以下命令从项目根目录执行，使用本机已验证的日文环境和模型包。输出路径必须不存在；重复运行时换一个名称，脚本不会覆盖 WAV 或同名 JSON。

```sh
SAKURA_REFS=../SakuraTTS-References

"$SAKURA_REFS/.venv-japanese-macos/bin/python" scripts/synthesize_japanese.py \
  --frontend-package "$SAKURA_REFS/models/converted/20260919T141358.883032Z-japanese-frontend-resources" \
  --reference-package "$SAKURA_REFS/models/converted/20260919T142950.546665Z-v2pro-japanese-reference" \
  --gpt-package "$SAKURA_REFS/models/converted/20260919T123226.764557Z-20260919T104221.749176Z-010197bfc30b-gpt-fp32-lossless-storage" \
  --sovits-package "$SAKURA_REFS/models/converted/20260919T123227.630582Z-20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32-lossless-storage" \
  --text 'こんにちは。今日はいい天気ですね。よろしくお願いします。' \
  --seed 0 \
  --output "$SAKURA_REFS/runs/manual-japanese-001/speech.wav"
```

模型和语言资源不随源码仓库分发。日文运行依赖固定在 `requirements-mlx-japanese.txt`，当前只验证了 macOS / Python 3.11。新环境可安装这份文件，无需安装开发环境的完整依赖。独立环境的安装清单与边界见 [干净环境记录](experiments/2026-09-19-japanese-clean-environment.md)。

脚本输出 `speech.wav` 和 `speech.json`。JSON 保存原文、规范化文本、音素、模型与参考身份、采样参数、实际停止原因和分段耗时。音频有效或正常 EOS 只说明生成完成，内容与音色仍需独立检查。当前 seed 使用 NumPy RNG，不等同于官方 Torch 的同值 seed。

默认配置为 CPU FP64 GPT Prefill、GPU FP32 Decode、CPU FP32 声学 encoder 和 GPU FP32 flow / decoder。`top_p=1`、`speed=1`，WeightNorm 折叠候选默认关闭。`--early-stop-num 2700` 是已验证请求的显式参数，并非从所有模型包推导出的默认上限。KV 容量默认 1024，超过容量会报错，不会截掉原文。

退出码 `0` 表示正常结束，`2` 表示达到生成次数限制，`1` 表示运行异常。后两类的 JSON 保留停止原因或异常，不算质量验收通过。语言范围为 `ja/all_ja`，一次只处理一个 `cut0` 片段；识别到尚未实现的英文段或拆成多个片段时明确报错。

## 导出日文前端资源

已有固定官方源码、编译好的日文用户词典和完整语言识别模型时，可以直接导出资源，无需先运行历史 Harness：

```sh
SAKURA_REFS=../SakuraTTS-References
python3 scripts/prepare_japanese_resources.py \
  --official-source "$SAKURA_REFS/GPT-SoVITS" \
  --user-dictionary "$SAKURA_REFS/GPT-SoVITS/GPT_SoVITS/text/ja_userdic/user.dict" \
  --language-model "$SAKURA_REFS/models/shared/fast_langdetect/lid.176.bin" \
  --output "$SAKURA_REFS/models/converted/manual-japanese-resources-001"
```

该工具仅使用标准库，不重编词典、不下载资源，也不覆盖现有目录。它保存官方 V2 符号顺序、原词典和完整 `lid.176.bin`；新目录可作为合成的 `--frontend-package`。三资源与既有包的逐字节对照见[导出验证](experiments/2026-09-19-japanese-resource-export.md)。

## 准备参考

更换参考音频时，先在开发环境运行 `scripts/prepare_japanese_reference.py`。它使用固定官方代码，从原始音频和日文转写生成参考条件并退出；普通合成只读取结果包。

准备工具需要官方 SoVITS、CNHuBERT、SV、日文词典和语言识别资源。GPT checkpoint 仅用于绑定模型身份，不加载 GPT 网络；中文 BERT 不加载。两个官方音频预处理分支各自保留，未将语义与声学参考混用。工具要求明确给出输入路径，详见 `--help` 和 [参考准备实验](experiments/2026-09-19-japanese-reference-preparation.md)中的完整复现命令。

条件包同时绑定官方提交、GPT / SoVITS 权重哈希、原始音频哈希和转写。原模型和音频保持不变；新包写到参考目录的新路径。当前新准备包保存完整来源证据，元数据也计入体积，不能把约 195 KiB 的数组数据称为整个新包大小。

## 当前依赖与模型体积

2026-09-19 对实际日文环境的已安装 distribution 文件去重计数，并单独清点上面四个目录：

| 内容 | 字节数 | 约 MiB |
|---|---:|---:|
| 日文环境的已安装包，含 pip / setuptools | 707,773,683 | 674.99 |
| GPT 推理目录 | 163,824,866 | 156.24 |
| SoVITS 推理目录 | 87,937,784 | 83.86 |
| 日文前端资源目录 | 152,595,967 | 145.53 |
| 本轮日文参考条件目录，含完整来源元数据 | 2,411,450 | 2.30 |
| 合计 | 1,114,543,750 | 1062.91 |

这约 1.04 GiB 是当前文件的逻辑字节总和，未计基础 Python 解释器、引擎源码、未登记缓存、系统共享库和下载缓存；模型目录也含转换证据文件。它不是最终安装包大小或文件系统实际占用，没有用它宣称相对完整开发环境的缩减比例。ASR 和参考准备环境不计入日常合成范围。原始逐文件清单、统计脚本及退出码 0 保存在 `SakuraTTS-References/runs/20260919T145332.842142Z-japanese-runtime-footprint/`。

## 证据边界

四条原始日文在固定官方随机输入下，已完成 [原文到 PCM 对照](experiments/2026-09-19-native-japanese-text-speech.md)。原报告句的 WAV 与用户此前确认正常的文件逐字节相同；这个听感结论只适用于同一文件。

独立随机生成另行记录，不继承固定回放的音质结论。当前完整请求计时包含前端、模型加载和释放；导入、初始包校验与输出写盘另列。完整 WAV 生成时间不等于流式首包时间，Apple 统一内存计数不等于 NVIDIA 显存。后续继续验证阶段释放、参考更换和取消，再在 Windows 实机验证后端和 GPU 执行优化。

Python 调用方已经可以在多条请求之间复用 `MLXGPT` / `MLXSoVITS`，每条调用 `prepare_text` 和 `synthesize_prepared`。生成完成或请求失败后，可显式调用 `gpt.release_request_state()` 丢弃本条 KV、位置和诊断状态，保留权重；下一次 Decode 必须先重新 Prefill。方法不替调用方清空分配器缓存或卸载模型。[三策略实测](experiments/2026-09-19-native-model-lifecycle.md)保存了连续请求的速度、空闲占用和失败恢复结果。

需要减少 GPT 状态与声学工作区的重叠时，调用 `synthesize_prepared(..., release_gpt_state=True)`；默认 `False` 保留调用方对状态寿命的控制。进入语义生成后，该选项在成功或异常时通过 `finally` 清空 KV，声学阶段开始前已完成释放，耗时计入 `semantic_seconds`。普通单次 CLI 已显式启用此选项，并在最后释放全部模型；输出 JSON 的 `runtime_policy` 记录选择。四条日文的[提前释放对照](experiments/2026-09-19-gpt-state-before-acoustic.md)中波形保持，MLX 请求峰值少约 96 MiB，耗时增加 2–16 毫秒；缓存和 RSS 未因此等量下降。普通 CLI 的[同 seed 回归](experiments/2026-09-19-japanese-cli-early-release.md)也已确认整份 WAV 不变。
