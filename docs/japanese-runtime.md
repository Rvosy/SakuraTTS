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

日文运行链和相关开发工具的 JSON 读写统一使用 UTF-8。[默认编码回归](experiments/2026-09-20-utf8-package-portability.md)已在 Mac 模拟 CP932 / CP936 下完成真实入口验证；Windows 依赖与 CUDA 执行仍待实机验证。

默认配置为 CPU FP64 GPT Prefill、GPU FP32 Decode、CPU FP32 声学 encoder 和 GPU FP32 flow / decoder。`top_p=1`、`speed=1`，WeightNorm 折叠候选默认关闭。`--early-stop-num 2700` 是已验证请求的显式参数，并非从所有模型包推导出的默认上限。KV 容量默认 1024，超过容量会报错，不会截掉原文。

退出码 `0` 表示正常结束，`2` 表示至少一片达到生成次数限制，`1` 表示运行异常。后两类的 JSON 保留停止原因或异常，不算质量验收通过。语言范围为 `ja/all_ja`；识别到尚未实现的英文段时明确报错。

CLI 会一次准备原文的所有 `cut0` 片段，再顺序合成完整 WAV。换行可能被官方短片合并规则合并；超过 510 字符时按官方标点规则继续拆分，不擅自硬切或丢弃原文。每片使用新的 GPT 历史和同一参考，整条请求共用一个 RNG；某片遇到 EOS 或次数限制后仍处理后续片段。JSON 的 `fragments` 保存每片文本、停止原因、PCM 偏移和长度；全部合成成功后才写 WAV，后片异常不会留下只含前片的音频。字符切分不保证 KV 容量足够，长输入仍可能需要显式调整 `--capacity`。

CLI 默认 `--model-policy staged`：每片完成语义生成后卸载 GPT，再加载 SoVITS，片尾卸载后处理下一片。`--model-policy simultaneous` 在所有片段间复用两模型，GPT 请求状态仍逐片释放。两者生成规则相同；此前[单片同条件实测](experiments/2026-09-19-native-staged-loading.md)中四例 WAV 保持相同，长句请求内 MLX 分配器峰值约少 304 MiB、耗时增加约 51 ms。多片段需要重复加载，代价另见[多片段验证](experiments/2026-09-20-japanese-multifragment.md)。这些是 Mac 数据，不能推作 NVIDIA 显存结论。

`--bind-reference` 是另一个可选项，默认关闭。它在加载 SoVITS 时先算出当前参考的五项声学投影，再跳过对应的 14 个常驻权重。JSON 的 `runtime_policy.bind_reference` 记录选择，`reference_projection_seconds` 是 `sovits_load_seconds` 的子项，不重复累加。Python 对应 `MLXSoVITS.load(..., reference=reference)`；绑定后换参考须新建实例，错用参考会报错。原模型包和完整权重对照路径保留，安装体积不会因此缩小。

绑定加载会产生临时工作区。CLI 在加载完成并同步后清理空闲分配器缓存，这段耗时也包含在 `sovits_load_seconds` 内；Python 模型加载器本身不清理调用方的全局缓存。需要保留模型的宿主应分别测活跃内存和缓存，不能仅凭前者下降判断总占用。

Python 调用方可用 `generate_prepared_semantic` 与 `synthesize_acoustic` 分开安排加载；阶段结果绑定目标音素、参考条件和同一个 RNG，不持有 GPT 模型。调用方负责完成在途工作并卸载模型，阶段计算时间不含中间加载和卸载。既有 `synthesize_prepared` 与权重常驻方式继续可用。

多片段调用方用 `prepare_text_request(...).fragments` 取得完整准备结果，再逐片调用上述接口；所有片共用同一个 RNG，并在进入下一片前完成本片声学计算。前端耗时只保存在请求级 `seconds`，每片 `seconds` 为零，汇总时只加一次。原 `prepare_text` / `synthesize_prepared` 仍限定单片，不会默认取第一片。

Python 接口可通过 `cancel_requested=event.is_set` 请求取消，并捕获 `sakuratts.generation.SynthesisCancelled` 查看取消阶段。它在语义步和完整声学计算的边界检查；声学已经开始时，要等该次计算返回，结果会被丢弃。取消不会返回部分 PCM，也不会自动卸载调用方持有的模型。分阶段调用时须向两个接口分别传入谓词。CLI 当前没有新增取消协议，宿主播放队列与 Windows 的打断延迟仍待集成验证。

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

## 在新目录组装运行环境

`harness/export_japanese_runtime.py` 用于 Mac 部署实验。传入 `--venv`、上述四个 `--*-package` 和不存在的 `--output` 目录，它复制运行源码、完整资源包、依赖及默认捆绑的基础 Python，然后在最终路径创建新虚拟环境。它不下载资源、不执行模型，也不修改输入目录。

导出保留日文运行依赖的 distribution 元数据和词典 / Nani 等包内资源，排除 Python 字节码缓存。默认还省去 pip / setuptools 及捆绑 Python 的 ensurepip；需要完整环境对照时用 `--include-install-tools`。推理目录的依赖升级应在开发环境完成后重新导出。`manifest.json` 逐文件记录来源、哈希、符号链接与逻辑字节数，并分别记录源环境和导出后的 distribution。默认 `--python-mode bundled` 包含解释器；显式选择 `external` 时，解释器仍是外部依赖，大小不计入目录。

运行入口位于新目录的 `venv/bin/python` 和 `scripts/synthesize_japanese.py`，四个包位于 `resources/{frontend,reference,gpt,sovits}`。虚拟环境绑定创建时的绝对路径；以后再移动目录，需要重建虚拟环境。导出成功本身不等于隔离运行已通过，也不证明 Windows 部署或任意平台可搬迁。

[实际隔离验证](experiments/2026-09-20-runtime-relocation.md)已完成：禁用旧源码、环境、模型、历史实验、用户缓存 / 临时目录及网络后，四条日文与原环境 WAV 逐字节相同。包含解释器与清单的安装目录为 1.070 GiB，系统库、驱动服务和生成输出另计；首轮隔离启动明显更慢，未作性能达标声明。

后续[去掉安装工具](experiments/2026-09-20-runtime-install-tools.md)后，默认推理目录为 1.047 GiB，含清单净少 23.865 MiB，四条日文严格隔离生成的 WAV 仍相同。完整模型和日文资源保持，运行内存与速度未因此宣称改善。

## 证据边界

四条原始日文在固定官方随机输入下，已完成 [原文到 PCM 对照](experiments/2026-09-19-native-japanese-text-speech.md)。原报告句的 WAV 与用户此前确认正常的文件逐字节相同；这个听感结论只适用于同一文件。

独立随机生成另行记录，不继承固定回放的音质结论。当前完整请求计时包含前端、模型加载和释放；导入、初始包校验与输出写盘另列。完整 WAV 生成时间不等于流式首包时间，Apple 统一内存计数不等于 NVIDIA 显存。阶段释放、两参考切换、Python 计算边界取消及[便携验收包](windows-validation.md)已有各自记录；Windows 后端和 GPU 执行优化仍待实机验证。

Python 调用方已经可以在多条请求之间复用 `MLXGPT` / `MLXSoVITS`，每条调用 `prepare_text` 和 `synthesize_prepared`。生成完成或请求失败后，可显式调用 `gpt.release_request_state()` 丢弃本条 KV、位置和诊断状态，保留权重；下一次 Decode 必须先重新 Prefill。方法不替调用方清空分配器缓存或卸载模型。[三策略实测](experiments/2026-09-19-native-model-lifecycle.md)保存了连续请求的速度、空闲占用和失败恢复结果。

需要减少 GPT 状态与声学工作区的重叠时，调用 `synthesize_prepared(..., release_gpt_state=True)`；默认 `False` 保留调用方对状态寿命的控制。进入语义生成后，该选项在成功或异常时通过 `finally` 清空 KV，声学阶段开始前已完成释放，耗时计入 `semantic_seconds`。普通单次 CLI 已显式启用此选项，并在最后释放全部模型；输出 JSON 的 `runtime_policy` 记录选择。四条日文的[提前释放对照](experiments/2026-09-19-gpt-state-before-acoustic.md)中波形保持，MLX 请求峰值少约 96 MiB，耗时增加 2–16 毫秒；缓存和 RSS 未因此等量下降。普通 CLI 的[同 seed 回归](experiments/2026-09-19-japanese-cli-early-release.md)也已确认整份 WAV 不变。
