# Windows 已保存音频的 ASR 内容检查

首轮 16 条音频的两组候选对照没有出现新增的转录字符差异，但对照本身有几处值得复听。补测官方原输出和速度候选的自然随机输出后，共检查了 24 条音频：官方原输出也有开头问候语未被识别的情况，速度候选的自然随机短句、多句则识别到了问候语，标点例仍有疑点。ASR 结果不能单独确认实际漏读或发音错误。音色、自然度和听感尚未验收，`quality_accepted` 保持 `false`。

## 数据与识别设置

首轮每个配置选取 `short`、`long`、`multi`、`punctuation` 的第一条热请求 WAV，共 16 条；ASR 只读取已保存的 WAV，没有裁切音频。原始输入、转录和全部 segments 保留在 [首轮审计数据](data/2026-09-20-windows-asr.json) 中；新增 8 条及自然随机性能记录见 [补测审计数据](data/2026-09-20-windows-asr-supplement.json)。首轮证据保留原样。

| 配置 | GPT / 声学 | 随机输入 | 本轮音频目录 |
| --- | --- | --- | --- |
| `baseline-fp32` | FP32 baseline / FP32 整图 | 固定官方回放 | `outputs/windows-vocoder-public/baseline-fp32` |
| `fast-timing` | FP32 split-KV / FP16 分块 256 | 固定官方回放 | `outputs/windows-vocoder-public/fast-timing` |
| `original-timing` | FP16 baseline / FP16 整图 | 自然随机，seed 1234 | `outputs/windows-vocoder-worker/original-timing` |
| `natural-timing` | FP16 baseline / FP16 分块 256 | 自然随机，seed 1234 | `outputs/windows-vocoder-public/natural-timing` |

识别使用已有的本地 `mobiuslabsgmbh/faster-whisper-large-v3-turbo`，固定 revision `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`。后端为 faster-whisper 1.2.1、CTranslate2 4.6.0，CPU INT8、4 线程、1 worker。语言固定为日文，temperature 0、beam size 5；不传入目标文本、prompt 或 hotwords，不启用 VAD，也不上传音频。

ASR 使用独立开发环境 `.venv-windows-asr`，没有并入 TTS 日常运行环境。准备记录中，已有模型文件为 1,621,665,983 字节，新增权重下载为 0；当前选定依赖的 wheel 合计 87.79 MB，环境在准备完成时为 293.19 MB。若计入一次被替换的 setuptools wheel，准备阶段依赖产物合计 88.61 MB；这个数字包含缓存命中的产物，且不含 HTTP 开销，不能等同于实际网络流量。这些都是 ASR 开发工具的独立开销，不计入 TTS 安装体积或推理性能。来源为 `outputs/windows-asr-preparation-20260920/preparation-summary.json`，版本锁在 `data/windows-asr/requirements.lock.txt`，许可证副本在 `data/windows-asr/licenses/`。

两次识别都调用 `harness/windows_asr_review.py`，完整音频路径和解码参数保存在各自的 `result.json`。下面展开补测 8 条的命令；复跑时使用新输出目录，脚本不会覆盖旧结果。`-X utf8` 固定控制台编码，不改变识别参数。

```powershell
$asrCases = @('short', 'long', 'multi', 'punctuation')
$asrDirectories = @(
    'outputs/windows-baseline/fp32-naive',
    'outputs/windows-vocoder-public/fast-natural-timing'
)
$asrAudioArgs = foreach ($asrDirectory in $asrDirectories) {
    foreach ($asrCase in $asrCases) {
        '--audio'
        "$asrDirectory/hot-neutral-$asrCase-00.wav"
    }
}
.\.venv-windows-asr\Scripts\python.exe -X utf8 -B harness/windows_asr_review.py `
    --resources data/windows-asr/manifest.json --cpu-threads 4 `
    --output outputs/windows-asr/official-and-fresh-fast-new @asrAudioArgs
```

首轮采用相同参数，音频目录依次为上表中的 `baseline-fp32`、`fast-timing`、`natural-timing`、`original-timing`，输出到 `outputs/windows-asr/initial-four-configurations`。补测原输出目录为 `outputs/windows-asr/official-and-fresh-fast`。

独立 CPU 审计脚本为 `outputs/windows-asr/summarize_initial.py`。实际执行：

```powershell
.\.venv-windows-runtime\Scripts\python.exe -B outputs/windows-asr/summarize_initial.py --publish
.\.venv-windows-runtime\Scripts\python.exe -B outputs/windows-asr/summarize_supplement.py --publish
```

审计重新读取了 16 份 WAV 和 16 份 ASR JSON，核对完整 WAV 哈希、PCM 载荷哈希、帧数、采样率，以及原 TTS 报告的请求文本、前端归一化文本和请求身份。模型 5 个文件的 SHA-256、资源清单中的 Git blob / LFS 对象身份、依赖版本和固定解码选项也已核对。上游 revision 的来源是已准备的资源清单，本次审计没有重新访问网络。

ASR 的 `result.json` 已是 `completed_asr_only`。控制台曾出现日文乱码；原始 JSON 按 UTF-8 严格解码成功，没有替换字符，以下引用均来自文件。

## 候选与对应对照

`baseline-fp32 → fast-timing` 四对中，三对原始转录完全一致，标点例只差空白。`original-timing → natural-timing` 四对中，也有三对原始转录完全一致；多句例的标点和分段不同，整段文字的内容相同。两组均为四对去标点、空白后的转录一致。

两组对照的 TTS 文本、前端结果、采样 token、返回语义 token、停止原因和 PCM 长度也已核对一致。这里的结论仅限于对应配置迁移后没有新增 ASR 字符差异。固定回放与自然随机使用不同的随机输入机制，生成的语义序列和音频时长不同，不能把两组的识别差异归因于某一项精度改动。

## 已发现的待复听位置

| 位置 | 原目标文本 | 原始 ASR 结果与观察 |
| --- | --- | --- |
| 固定回放短句开头 | `おはよう。今日もよろしくね。` | baseline 和 fast 均为 `今日もよろしくね`，没有转录到 `おはよう`。自然随机两组均转录到完整问候语。 |
| 固定回放多句开头 | `おかえりなさい。今日はどんな一日だった？` | baseline 和 fast 均从 `一興はどんな一日だった?` 开始，没有转录到 `おかえりなさい`；后两句仍有转录。自然随机两组均转录到开头问候语。 |
| 固定回放标点例句末 | `ちゃんと言ってね。` | baseline 和 fast 均识别为 `ちゃんと限定ね。`。 |
| 自然随机标点例首尾 | `えっ、本当？` / `ちゃんと言ってね。` | original 和 public 均识别为 `返答?` / `ちゃんと銀手ね。`。 |
| 自然随机长句中部 | `図書館へ行って` | original 和 public 均转录为 `図書館行って`；是否实际缺少助词 `へ` 需要听音确认。 |

长句的最终句子在四个配置中均有转录，结尾保留 `明日の予定をゆっくり考えたいな。`，本轮没有发现候选新增的末句缺失。ASR 时间戳未经词级或音素级对齐验证，不能据此精确判断音频截断位置。

`よかった → 良かった`、`あとで → 後で` 属于可见的正字差异。`返答`、`限定ね`、`銀手ね` 则可能涉及 ASR 听辨，也可能指向合成发音问题；这里不改写转录来匹配目标文本，也不替模型作听感判断。

只读检查也没有发现保存脚本主动裁去音头：`scripts/windows_official_baseline.py` 将 `tts.run(inputs)` 返回的音频全部拼接后写入 WAV，诊断捕获保留返回语义切片；官方和自有实现都使用 `[-idx:]` 约定。这有助于排除“WAV 保存时直接裁掉开头”这一解释，仍不能代替对音频及语义生成边界的核查。

## 字符指标的范围

指标先做 Unicode NFKC，再移除 Unicode 标点和空白，最后计算单位代价的字符编辑距离。不做汉字转假名、同音词替换或目标文本引导修正。表中为“编辑距离 / 目标字符数”；它是定位转录差异的工具，不是实际发音错误率。

| 文本 | baseline / fast | original / public 自然随机 |
| --- | ---: | ---: |
| short | 4 / 12 | 0 / 12 |
| long | 1 / 114 | 2 / 114 |
| multi | 11 / 45 | 2 / 45 |
| punctuation | 6 / 26 | 7 / 26 |

本轮没有执行听音验收；样本也只有一个 seed、四种日文文本。官方 FP32 严格 PCM 比较中，fast 候选原有的失败仍然保留，不能用 ASR 转录一致替代该数值判定。

## 官方原输出与速度候选自然随机补测

新增识别的 8 条分别来自 `outputs/windows-baseline/fp32-naive` 和 `outputs/windows-vocoder-public/fast-natural-timing`。解码选项、模型文件和依赖身份与首轮一致，8 份原始 JSON 及 segments 已重新核对。官方原计时报告没有记录这四条请求的前端归一化文本，审计将该字段保留为空；没有拿自有前端结果代填。

官方原短句转录为 `用もよろしくね`，自有 FP32 回放为 `今日もよろしくね`，两者不是相同转录。两条都没有识别到目标开头 `おはよう`。官方原多句同样从 `一興はどんな一日だった?` 开始，没有识别到 `おかえりなさい`；标点例同样出现 `限定ね`。官方与自有 FP32 回放四对中，三对去标点和空白后的转录相同，短句有字符差异。PCM 数值容差通过也不保证 ASR 文本逐字相同。

速度候选的自然随机短句转录为 `おはよう。今日もよろしくね。`，多句也识别到了 `おかえりなさい`。它的长句仍将 `図書館へ行って` 转录为 `図書館行って`；标点例为 `返答?それなら約束だよ。また明日ってちゃんと限定ね。`。与首轮自然 FP16 配置相比，三个文本的归一化转录相同，标点例从 `銀手ね` 变为 `限定ね`。两种写法都需复听，不能凭字符距离相同就认定发音问题相同或已经解决。

这轮补测让开头疑点的范围更清楚：它也出现在原官方的这组固定种子输出中，没有在当前速度候选的这组自然随机输出中重现。仍需听音确认疑点，并结合官方语义切片边界检查；不能把 ASR 未识别到问候语直接归为分块引入的漏字，也不能由四条自然随机样本宣称问题已被修复。

## 速度候选的自然随机性能补测

补测保留 FP32 GPT split-KV 和 FP16 声学分块 256，改用自然随机 seed 1234，不注入官方回放随机输入。计时轮 `fast-natural-timing` 有 13 次请求，每类热请求 3 次，关闭后台资源采样；独立 `fast-natural-memory` 轮有 5 次请求，只用于资源数据。

| 文本 | 热请求中位耗时 | 完整 PCM 时长 |
| --- | ---: | ---: |
| short | 208.36 ms | 3.46 s |
| long | 1,557.08 ms | 25.90 s |
| multi | 857.81 ms | 14.42 s |
| punctuation | 473.98 ms | 7.66 s |

以上耗时从原始文本提交到完整 PCM 返回，包含请求内的计算和传输，不是流式首包延迟。自然随机与此前固定回放产生不同长度的语义序列和音频，不能直接相减来计算优化收益。

独立资源轮的原始首个全卡样本为 **2,143 MiB**，采样峰值为 **3,251 MiB**，差值 **1,108 MiB**。这是 WDDM 下的全卡采样口径，包含桌面和其他进程，100 ms 采样可能漏掉瞬时峰值，不能称为进程独占显存，也不能直接对标其他项目的“0.8 GB”。

18 份 WAV 均重新核对了 PCM、请求契约、分块计划和逻辑传输量。计时轮同文本重复结果一致，计时与采样轮的 5 对结果也逐位一致；长句与此前公开生命周期 `lifecycle-fast` 的同输入基线 NPY 逐位一致。这些是执行稳定性和身份核验，不是音质验收。补测数据单独保存，没有改写先前提交的公开入口性能证据。
