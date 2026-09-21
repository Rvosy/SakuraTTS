# 从原始日文参考音频准备条件包

日期：2026-09-19。`research/tools/prepare_japanese_reference.py` 已用固定官方代码重新准备现有日文参考，五组输出数组与旧条件包逐字节相同。准备 worker 和独立离线复核进程均实际退出 0。

本轮补齐从原始参考音频到可读取条件包的工具入口。工具使用开发环境中的 Torch，正常合成读取条件包时仍无需 Torch。本轮只验证“朱雀院红叶”V2Pro、同一份日文参考、MPS / FP32；没有更换音色、合成目标语音、训练、ASR 或新增试听。

## 工具边界

命令显式接收官方源码、GPT / SoVITS checkpoint、音频、参考文本、CNHuBERT、SV checkpoint、OpenJTalk 主词典、用户词典和语言识别模型路径。GPT checkpoint 只用于绑定模型身份，不加载 GPT；日文参考使用官方全零 BERT，不加载中文 BERT / tokenizer。实际加载的模型为官方 SoVITS、CNHuBERT 和 ERes2NetV2。

准备发生在短生命周期子进程中，父进程只使用标准库和 NumPy。`--check-only` 只验证和散列输入、资源及源码，不导入推理后端。工具没有目标合成入口。初版仅支持 macOS，原因是实际已加载动态库的记录目前使用 dyld；`cpu` 参数未在本轮运行验证。

工具复用现有 `research/tools/prepared_reference.py` 与 `research/tools/prepared_acoustic.py`，生成现有 `sakuratts-prepared-reference-v1` 格式。没有修改原导出器或 reader。新包包含 `reference_phones`、`prompt_semantic`、`reference_bert`、`ge`、`ge512`；包加载时检查模型和参考身份、文件散列、数组结构及内容散列。

官方提交固定为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`。工具逐字检查显式列出的官方实现，并记录实际导入的官方文件和第三方实现。输入、资源及本地准备代码在准备前后分别核对。现有官方词典生成文件保留原样；CSV 与 MD5 不一致时直接拒绝运行，不触发源码目录中的自动重建。官方 `TTS_Config` 的临时目录和配置写到新运行目录，Numba / Matplotlib 缓存也写入该目录。

## 保留的官方参考语义

参考文本沿用官方参考路径：去掉首尾换行、补末尾标点、执行 `segment_and_extract_feature_for_text`。它没有走目标文本的短句前缀标点规则。本次原文为 `じゃあ、私、もっと悪い子になっちゃおうな〜`，准备后补为 `じゃあ、私、もっと悪い子になっちゃおうな〜。`，得到 47 个音素。Nani encoder / model 均已加载，provider 均为 `CPUExecutionProvider`。

同一份 OGG 在官方逻辑中有两条解码与重采样路径。此次实际观察到的解码后端都是 soundfile；不能根据已安装 FFmpeg 就说本次使用了 FFmpeg。

| 用途 | 实际路径 | 输出形状 |
|---|---|---|
| GPT 参考语义 | librosa / soundfile 读取 44100 Hz，再以 `soxr_hq` 到 16000 Hz | `[62827]` |
| SoVITS 参考谱 | torchaudio / soundfile 读取 44100 Hz，再以官方 torchaudio Resample 到 32000 Hz | `[1, 125654]` |
| SV 条件 | 从上一条 32000 Hz 数据再次重采样到 16000 Hz | `[1, 62827]` |

两个 torchaudio Resample 都使用 `sinc_interp_hann`、`lowpass_filter_width=6`、`rolloff=0.99`、`beta=None`，运行在 MPS。Libsndfile 为 1.2.2，Libsoxr 为 `0.1.3-14-ga66f3ee`。两条 16000 Hz 输出的散列不同，因此不能把它们合并为一次“等价”重采样。各步完整散列保存在 `worker-result.json` 和包 manifest。

参考语义保留官方的 `int(model_sampling_rate * 0.3)` 零填充。本模型采样率为 32000，实际追加 9600 个零；它作用于 16000 Hz 流，相当于 0.6 秒。没有擅自改为 4800 个零。CNHuBERT 仍直接调用官方 `.model(...)`，然后通过 SoVITS 的 `ssl_proj` 和量化器提取语义。

参考谱仍保留官方声道处理、幅值归一化条件和 `center=False`。SV 沿用官方 80 维 Kaldi fbank、16000 Hz、`dither=0` 和 ERes2Net `forward3`。声学条件仍保留 `ref_enc`、speaker embedding 投影、PReLU、单参考 `stack(...).mean(0)` 和 `ge_to512` 的顺序。

## 数组对照

原始音频为 `models/suzakuinmomiji/voice/refs/tone_refs/VO02_0204.OGG`，SHA-256 为 `c265a87781115e67d787ed81a7d2a3755a0aff54dd15d6c304b5fe61b01b91b4`。对照包为 `models/converted/20260919T135540.094161Z-v2pro-reference`。

| 数组 | dtype / shape | 结果 |
|---|---|---|
| `reference_phones` | int64 `[47]` | 字节相同 |
| `prompt_semantic` | int64 `[113]` | 字节相同 |
| `reference_bert` | float32 `[1024, 47]`，全零 | 字节相同，最大误差 0 |
| `ge` | float32 `[1, 1024, 1]` | 字节相同，最大误差 0 |
| `ge512` | float32 `[1, 512, 1]` | 字节相同，最大误差 0 |

整数音素 / token 要求精确一致。浮点另外记录 `atol=1e-4, rtol=1e-5`、最大绝对误差、RMS 和超差数量，但容差通过不替代本次字节相同的验收。实际三个浮点数组的最大误差、RMS、超差数量均为 0。

新包的 `conditions.npz` 为 201,222 bytes，manifest 为 2,210,228 bytes，总计 2,411,450 bytes。元数据包含本次实际导入实现的身份，体积比旧包增加；本轮没有缩小条件包或安装包的收益结论。

独立复核使用 `.venv-japanese-macos`，该环境未安装 Torch / Transformers。reader 成功读取新旧包、重新比较五组数组并确认数组只读，进程实际退出 0；此进程没有导入 Torch、Transformers、MLX 或 ONNX Runtime。复核还覆盖 3 个输入、24 个资源、18 个显式官方源码、4 个本地源码、5035 个导入实现、8 个动态库、20 个 distribution RECORD 和 96 份源码快照。原模型、原音频、原资源及官方 checkout 状态均与准备前相同。

## 准备成本与释放

这里只运行了一次准备，没有冷热重复性能验收。计时包含同步、诊断散列 / CPU 复制和阶段内的产物保存，不代表正常合成延迟。worker 启动至实际退出为 24.611 秒，不包含父进程预检及最终导出耗时。

| 阶段 | 耗时 |
|---|---:|
| 导入和前端初始化 | 12.551 s |
| 模型加载 | 0.945 s |
| 参考准备及阶段产物保存 | 4.040 s |
| 声学条件准备及保存 | 0.198 s |

以下为同步边界快照，单位 MiB。MPS allocated 和 driver 是不同计数，不能相加；它们与 RSS 的统计口径也不同。

| 边界 | RSS | MPS allocated | MPS driver |
|---|---:|---:|---:|
| 加载模型前 | 627.156 | 0 | 0.453 |
| 模型加载后 | 1309.703 | 873.525 | 1192.469 |
| 参考准备并释放 CNHuBERT / SV 后 | 1771.188 | 312.880 | 1231.922 |
| 声学条件准备并释放相关模块后 | 1796.734 | 176.614 | 1231.953 |
| 模型释放后 | 1738.594 | 0.977 | 119.953 |

最后一次观察到的进程生命周期 RSS high-water 为 1936.906 MiB。该计数不是 GPU 峰值，也不是每阶段峰值。释放模型后的 MPS allocated 仍为 1,024,512 bytes，不能声称清零；残留的具体持有者未在本轮定位。随后 worker 实际退出 0。RSS 中还包含开发环境的导入和缓存。

这些是 Apple 统一内存数据。没有 NVIDIA 显存测量，也没有证明相对旧完整准备流程降低了多少峰值或耗时。原始 stdout / stderr 保留，包括官方完整 SoVITS 初始化时的 `enc_q` missing keys；没有把该输出改写为无告警日志。

## 证据与复现

下列路径相对于 `SakuraTTS-References/`：

- 准备运行：`runs/20260919T142950.546665Z-japanese-reference-prepare/`。
- 新包：`models/converted/20260919T142950.546665Z-v2pro-japanese-reference/`。
- 新包 manifest SHA-256：`53230f7cdb48ba8dd3f1d54e319a56b82e1c57d7b2fd40beac56a8f69021988d`。
- 原始参数 / 环境：`preflight.json`、`worker-result.json`；真实退出、日志与完整命令：`process-result.json`、`process.stdout-stderr.log`、`result.json`。
- 精确对照：`comparison.json`；独立离线复核：`offline-audit-final.json`、`offline-audit-final-process.json` 及对应日志。

首次 `offline-audit.json` 已保留。它在记录文件清单时把尚在写入的自身日志计入，日志随后发生了预期变化。最终复核排除复核本身的输出、日志和进程记录，重新检查其他证据文件，没有重跑模型。

复现命令如下。工具会新建带时间戳的目录，不重用本轮历史目录：

```sh
REFS=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REFS/.venv-official-macos/bin/python" \
  /Users/beyondpower/Documents/Projects/SakuraTTS/tools/prepare_japanese_reference.py \
  --references "$REFS" \
  --official-source "$REFS/GPT-SoVITS" \
  --gpt-checkpoint "$REFS/models/suzakuinmomiji/voice/models/朱雀院红叶-e15.ckpt" \
  --sovits-checkpoint "$REFS/models/suzakuinmomiji/voice/models/朱雀院红叶_e8_s38928.pth" \
  --audio "$REFS/models/suzakuinmomiji/voice/refs/tone_refs/VO02_0204.OGG" \
  --text 'じゃあ、私、もっと悪い子になっちゃおうな〜' \
  --cnhubert "$REFS/models/shared/chinese-hubert-base" \
  --sv-checkpoint "$REFS/models/shared/sv/pretrained_eres2netv2w24s4ep4.ckpt" \
  --main-dictionary "$REFS/.venv-official-macos/lib/python3.11/site-packages/pyopenjtalk/dictionary" \
  --user-dictionary "$REFS/GPT-SoVITS/GPT_SoVITS/text/ja_userdic/user.dict" \
  --language-model "$REFS/GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin" \
  --device mps \
  --comparison-package "$REFS/models/converted/20260919T135540.094161Z-v2pro-reference"
```

后续可将新包接入普通日文生成，继续检查参考身份绑定和独立采样路径。更换实际参考音频、多参考、其他模型家族、CPU 准备、Windows / CUDA，以及完全无 Torch 的原始参考准备仍未验证。
