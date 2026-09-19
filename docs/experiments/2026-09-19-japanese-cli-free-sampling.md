# 普通日文 CLI：独立随机生成验证

日期：2026-09-19。模型：朱雀院红叶 V2Pro。平台：Apple M4 / macOS，独立日文运行环境。

`scripts/synthesize_japanese.py` 已用原始日文、独立模型包和新准备的参考条件完成三次生成。seed 0、seed 1 各运行一个独立进程，再用第三个进程重复 seed 0；三次均退出 0，并由 EOS 停止。运行过程没有读取历史目标特征、目标 token、随机噪声或对照 trace。

这轮验证普通入口能否独立发声。生成时没有执行 ASR 或人工试听，后续另做的本地 ASR 见文末；目前不能宣布发音和音色通过。当前实际验收仍限于这套红叶 V2Pro 权重；CLI 能接收其他目录不表示那些模型已经兼容。

## 输入与执行范围

原始目标文本没有改写：

```text
こんにちは。今日はいい天気ですね。よろしくお願いします。
```

使用 `ja`、单个 `cut0` 片段和一组日文参考条件。参考包来自本轮从原始音频重新准备的 `20260919T142950.546665Z-v2pro-japanese-reference`。GPT、SoVITS 权重包和前端资源包保持不变，官方参考提交为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`。

CLI 使用 NumPy `default_rng(seed)`，直接调用 `prepare_text` 和 `synthesize_prepared`。语义采样和声学噪声都在请求内生成；没有重新实现采样器。同一整数种子不等同于 PyTorch 的随机序列，本轮也没有用自由采样的差异判断数值兼容性。

参数为 `top_k=15`、`top_p=1`、`temperature=1`、`repetition_penalty=1.35`、`early_stop_num=2700`、`speed=1`、`noise_scale=0.5`、尾静音 `0.3 s`，KV 容量为 1024。2700 是本轮显式请求参数，不由当前 GPT 包推导；该包的配置没有 `max_sec`。

计算路径沿用已验证设置：GPT Prefill 在 CPU 以 FP64 计算，Decode 为 GPU FP32；声学 Encoder 和 softmax 为 CPU FP32，Flow/Decoder 为 GPU FP32，不折叠 WeightNorm。前端准备完成后关闭 OpenJTalk 和语言检测器，再加载 GPT/SoVITS；生成后释放模型。Nani/Sudachi 的包级缓存可能留到进程退出。

三个进程结束时均检查到以下模块未导入：`torch`、`transformers`、`sakuratts.chinese`、`sakuratts.g2pw`、`sakuratts.mlx_bert`。

## 结果

| 进程 | sampled / semantic token | 停止原因 | 波形主体 / PCM 时长 | 完整请求 | RTF，按波形主体 | 父进程墙钟 |
|---|---:|---|---:|---:|---:|---:|
| seed 0，第 1 次 | 115 / 114 | argmax EOS、sample EOS | 4.56 / 4.86 s | 1.808 s | 0.3964 | 2.058 s |
| seed 1，第 1 次 | 110 / 109 | argmax EOS、sample EOS | 4.36 / 4.66 s | 1.603 s | 0.3676 | 1.824 s |
| seed 0，重复 | 115 / 114 | argmax EOS、sample EOS | 4.56 / 4.86 s | 1.668 s | 0.3658 | 1.892 s |

“波形主体”指声学模型返回的完整波形，包括其中自然存在的停顿；只排除之后追加的 0.3 秒尾静音。CLI 同时保存 `waveform_seconds` 与 `pcm_seconds_with_trailing_silence`，主字段 `rtf` 按前者计算，含尾静音的结果另存为 `rtf_with_trailing_silence`。

完整请求计时包括前端加载、目标特征准备、前端释放、推理模型加载及权重校验、生成与模型释放。包预检查、初始模块导入和写文件单独计时。父进程墙钟覆盖子进程启动到退出；三个进程彼此独立，但系统文件缓存和编译缓存可能已热，不能将这些数值当作清空缓存后的冷启动基准。没有诊断采样，也没有做首包延迟、内存峰值或 CUDA 显存测量。

两次 seed 0 的整个 WAV 逐字节相同，SHA-256 为 `b023ea2a447538028b9b034c9cc7f5d273ae9710fb4701c88d4803934188ecbe`。seed 1 的 SHA-256 为 `b2c1d4d9d46fd1599a0ae86c032da9f52e06557b2a87c8302402cf54499f2c0a`，与 seed 0 不同。这只验证本轮条件下的重现性和种子确实影响输出。

三次规范化文本均完整保留原始三句，目标序列均为 58 个音素，包含三个句末标点。开头是 `k o [ N n i ch i w a .`，“今日は”的部分是 `ky o ] o w a`。序列中的两个 `UNK` 继续保留，没有删去。完整音素 ID、符号和语言分段分别在每次 JSON 与汇总结果中保存。

这些检查确认开头文本进入了前端，并且 `は` 在对应音素位置准备为 `w a`。它们不能代替对新 WAV 的试听；EOS 和 WAV 有效也不能证明每个输入词都实际读出。各次记录的 `quality.asr` 和 `quality.human_listening` 都是 `not_run`。

## 证据与复现

原始证据目录：

```text
/Users/beyondpower/Documents/Projects/SakuraTTS-References/runs/20260919T143508.441566Z-japanese-cli-free-sampling
```

`prepared.json` 保存输入、三个种子和 15 个源文件的哈希；`source/` 是实际执行的源码快照。`run_processes.py` 启动独立进程，每次的 `process.json` 保存实际命令、退出码、工作目录、环境覆盖项和墙钟时间，完整 stdout/stderr 另存。CLI 本身为每条 WAV 写同名 JSON，记录包及参考身份、实际参数、文本、停止原因、计时和质量检查状态。

`verify_results.py` 只做离线检查。三次各 17 项检查均通过，包括原文及音素覆盖、正常 EOS、依赖边界、WAV 头、尾静音、时长/RTF 口径、源码身份与质量状态；汇总为 `result.json`。`static/` 保存修正后源码的 8 项静态检查，包含 `--help`、不加载推理依赖、拒绝覆盖以及提前拒绝 `top_k=0`。此前 `20260919T143124.030908Z-japanese-cli-static` 中的旧源码快照和结果保留原样。

以下命令使用本轮快照，输出路径需要没有同名 WAV 或 JSON。将 `--seed 0` 改为 `--seed 1` 时也应换一个输出路径。

```bash
tts_refs=/Users/beyondpower/Documents/Projects/SakuraTTS-References
tts_source="$tts_refs/runs/20260919T143508.441566Z-japanese-cli-free-sampling/source"
"$tts_refs/.venv-japanese-macos/bin/python" "$tts_source/scripts/synthesize_japanese.py" \
  --frontend-package "$tts_refs/models/converted/20260919T141358.883032Z-japanese-frontend-resources" \
  --reference-package "$tts_refs/models/converted/20260919T142950.546665Z-v2pro-japanese-reference" \
  --gpt-package "$tts_refs/models/converted/20260919T123226.764557Z-20260919T104221.749176Z-010197bfc30b-gpt-fp32-lossless-storage" \
  --sovits-package "$tts_refs/models/converted/20260919T123227.630582Z-20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32-lossless-storage" \
  --text 'こんにちは。今日はいい天気ですね。よろしくお願いします。' \
  --language ja --seed 0 --top-k 15 --temperature 1 --repetition-penalty 1.35 \
  --early-stop-num 2700 --capacity 1024 \
  --output "$tts_refs/runs/manual-japanese-seed0/speech.wav"
```

后续需要试听这两条新随机样音，分别确认开头、助词和三句完整性。此前对另一条固定噪声 WAV 的反馈不能直接移用于本轮新音频。

## 新样音的独立本地 ASR

随后对 `seed0-first/speech.wav` 和 `seed1-first/speech.wav` 执行了本地离线转写。只读取已有 WAV，没有重新生成 TTS 音频，也没有修改原 CLI JSON 中的 `quality` 状态。转写前后，两条 WAV 与两份原 CLI JSON 的 SHA-256 均保持不变。

识别仍用固定的 `mlx-community/whisper-small-mlx`，提交 `45f3915923c7a79a5a5b5a7d909d39aeb0e5630e`，权重 SHA-256 为 `55b6674c9b339702d486e2b1573839a66f8ec8f821ed2886993ef717a86b09f5`。已有缓存完整，命令启用 `--offline`，没有下载新文件或上传音频。实际环境为独立的 `.venv-asr-macos`，使用 `mlx-whisper 0.4.3` 与 `mlx 0.32.2`，不属于 TTS 运行依赖。

识别语言显式设为 `ja`，`temperature=0`、`initial_prompt=None`、`condition_on_previous_text=False`。没有给识别器目标原句。下面保留 ASR 原文，没有按目标文本修改用字或补标点：

| 样音 | ASR 原文 |
|---|---|
| seed 0 | こんにちは。今日は良い天気ですね。よろしくお願いします。 |
| seed 1 | こんにちは今日はいい天気ですねよろしくお願いします |

两条转写都包含开头“こんにちは”及后两句，为新样音的内容检查补充了自动证据。Whisper 会按上下文恢复常见拼写，转写中的“今日は”不能证明实际读音是 wa；音色、自然度与逐字准确性仍需人工试听。本轮没有据转写给音频作质量排名。

每条得到一个 ASR segment，原始 JSON 保存了 token、时间戳、log probability 和无语音概率。seed 0、seed 1 的 segment 结束时间分别为 5.00、6.12 秒，超过实际 PCM 时长 4.86、4.66 秒；时间戳原样保留，不能用于判断准确的词边界。

ASR 子进程实际退出码为 0，父进程墙钟为 2.190 秒。逐文件识别时间为 0.432、0.232 秒，其中首个文件含模型加载。这些只记录 ASR 执行过程，不加入 TTS 性能或资源数据。

证据分为两个独立目录：

```text
SakuraTTS-References/runs/20260919T144738.003222Z-japanese-cli-asr-dispatch/
SakuraTTS-References/runs/20260919T145447.141713Z-asr-review/
```

前者保存父进程的实际命令、退出码、完整 stdout/stderr、源文件与模型哈希，以及转写前后的输入核对；后者保存 Harness 快照、依赖列表、参数和两份原始转写。seed 0 的原始 ASR JSON 哈希为 `7fb0e9dd4e48fd210f685ae314033e7a3f0e379fdd0ab1c5ecc14650d6af6e7c`，seed 1 为 `39656a40d37387e1b0389d7cfb407dc76be78cd02fc0853c3cd0746f927c246c`。

复现只读取这两份样音，每次创建新的 ASR 结果目录：

```bash
tts_refs=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$tts_refs/.venv-asr-macos/bin/python" -u \
  "$tts_refs/runs/20260919T144738.003222Z-japanese-cli-asr-dispatch/transcribe_regressions.py" \
  --references "$tts_refs" --offline \
  --model mlx-community/whisper-small-mlx \
  --revision 45f3915923c7a79a5a5b5a7d909d39aeb0e5630e \
  --audio ja "$tts_refs/runs/20260919T143508.441566Z-japanese-cli-free-sampling/seed0-first/speech.wav" \
  --audio ja "$tts_refs/runs/20260919T143508.441566Z-japanese-cli-free-sampling/seed1-first/speech.wav"
```
