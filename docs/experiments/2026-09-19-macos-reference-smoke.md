# Mac 上游参考实现首轮验证

日期：2026-09-19。状态：功能冒烟通过，性能与音质尚未正式验收。

使用用户选定的“朱雀院红叶”V2Pro 模型，在 Apple M4 上完成了 GPT-SoVITS 官方实现和 GSV-TTS-Lite 的 MPS / FP32 推理。两条路径均生成了中、日文音频，每种语言运行两次，共保存 8 个 WAV。

本次完成的是模型、参考音频、文本前端、GPU 推理到音频文件的连通验证。SakuraTTS 自有原生引擎和转换器尚未实现；Windows 验证按用户要求暂缓。

## 环境与输入

| 项目 | 实际配置 |
|---|---|
| 机器 | Mac mini，Apple M4，16 GiB 统一内存 |
| 系统 | macOS 26.5.2，arm64 |
| Python | 3.11.15；官方和 Lite 各用一个独立虚拟环境 |
| 推理依赖 | PyTorch / torchaudio 2.7.1，transformers 4.51.3，NumPy 1.26.4 |
| 设备与精度 | MPS / FP32；`PYTORCH_ENABLE_MPS_FALLBACK` 未设置 |
| 文本处理 | 中文 BERT 启用；pyopenjtalk-plus 0.4.1.post9；官方 G2PW 使用 ONNX Runtime 1.30.0 的 CPU 路径 |
| 执行方式 | 单请求、非流式；PyTorch CPU 线程数为 4 |
| 采样 | seed 1234，top-k 15，top-p 1.0，temperature 1.0，重复惩罚 1.35，语速 1.0 |
| Lite | `use_bert=True`，`use_flash_attn=False`，GPT 缓存 `[(1, 1024)]`，SoVITS 图缓存为空 |
| 官方 | `batch_size=1`，`parallel_infer=False`，`cut0`，不返回分片、不启用流式 |

MPS 可用性、配置和设备内存计数均已记录，Lite 日志也明确输出 `Device: mps, dtype: torch.float32`。这证明所选 MPS 路径能完成推理；本次没有逐算子 profiling，不能据此声称整条链路没有 CPU 计算。官方 G2PW、语言识别和部分文本处理本来就在 CPU 上执行。

三个参考仓库位于本机同级目录 `../SakuraTTS-References/`，未作为子仓库放入 SakuraTTS。该目录内的模型、样音、日志和环境仅保留在本机，不随本仓库上传；下文列出的本地证据路径均相对于该目录。

| 仓库 | 固定提交 | 本次用途 |
|---|---|---|
| GPT-SoVITS | `48b1a0169a28582a8984402f82cf438d3bfa6aca` | 官方 MPS 对照 |
| GSV-TTS-Lite | `6c049397142f4c9147a85f86b6ba37546e93a188` | Lite MPS 对照 |
| Genie-TTS | `d347fd0f8683e9a362b69f59fa0a4799ddb5e828` | 仅保存源码，未运行 CPU 推理 |

角色包来自本机 `Downloads/suzakuinmomiji.zip`。只复制其中的 `voice/` 文件，未修改原压缩包或原 Sakura 运行数据。压缩包使用 GBK 文件名，Python 3.11 校验时显式使用 `metadata_encoding="gbk"`。13 个复制文件均与压缩包内容及来源清单的 SHA-256 一致。

| 模型 | SHA-256 |
|---|---|
| `朱雀院红叶-e15.ckpt` | `010197bfc30b04d991f2bf060f962549932a8278b98c137d92f980e9cca8c0e9` |
| `朱雀院红叶_e8_s38928.pth` | `f8bd92196175f435ae9df2bc01e87b0119a5d94c0af81ada4573a9720536ab38` |

SoVITS 实际识别为 **V2Pro**，输出 32 kHz。GPT 配置为 24 层、隐藏维度 512、16 个注意力头、FFN 2048、语义词表 1025、音素词表 732。不能按文件名把它当成普通 V2 模型。

参考使用“开心”条目 `VO02_0204.OGG`，44.1 kHz 单声道、3.927 秒，日文文本为：`じゃあ、私、もっと悪い子になっちゃおうな〜`。

固定目标文本：

- 日文：`こんにちは。今日はいい天気ですね。よろしくお願いします。`
- 中文：`你好，欢迎使用樱花语音。现在正在测试苹果电脑上的语音合成。`

## 耗时与音频

时间由主机单调时钟测量，生成前后调用 MPS 同步，包含上游 `infer` / `run` 内的文本处理、参考准备和音频生成，不包含写 WAV 文件或播放。每条文本的第二次执行记为热运行；每种配置只有一个热样本，不计算 p50 / p95。

| 路径 | 语言 | 首次调用 / 秒 | 第二次调用 / 秒 | 音频时长 / 秒 | 第二次 RTF |
|---|---|---:|---:|---:|---:|
| Lite | 日文 | 19.011 | 1.476 | 3.308 | 0.446 |
| Lite | 中文 | 6.459 | 2.406 | 5.604 | 0.429 |
| 官方 | 日文 | 23.959 | 2.217 | 5.100 | 0.435 |
| 官方 | 中文 | 59.890 | 2.986 | 6.140 | 0.486 |

`RTF = 合成耗时 / 音频时长`。本次各热样本的 RTF 小于 1，表示完整生成耗时短于音频播放时长，不代表已经实现低首包延迟或流式播放。

官方日文首次调用包含语言识别模型下载，中文首次调用包含 G2PW 下载、解压和初始化。Lite 的模型加载耗时 151.195 秒也包含缺失资源下载；官方成功运行记录的加载耗时为 5.004 秒，当时共享模型已经存在。这些数字不是同条件冷启动对照。

同一路径、同语言的两次 WAV 哈希相同；官方与 Lite 的音频长度和哈希不同。尚未对齐两者的文本前端、中间特征、采样实现与停止行为，不能直接给出加速倍数或质量等价结论。

本地样音：

| 路径 | 中文 | 日文 |
|---|---|---|
| Lite | `runs/20260919T094124.872669Z-lite-mps/zh-2.wav` | `runs/20260919T094124.872669Z-lite-mps/ja-2.wav` |
| 官方 | `runs/20260919T094645.650008Z-official-mps/zh-2.wav` | `runs/20260919T094645.650008Z-official-mps/ja-2.wav` |

## 内存与磁盘记录

下表取最后一次中文执行结束时的内存快照。`allocated` / `driver` 是执行边界值，不是连续采样的峰值；RSS 是进程从启动至该时刻的高水位。

| 路径 | MPS allocated / MiB | MPS driver / MiB | 进程 peak RSS / MiB |
|---|---:|---:|---:|
| Lite | 1983.84 | 2187.00 | 1738.73 |
| 官方 | 2653.01 | 3903.33 | 2351.17 |

Mac 使用统一内存，以上指标口径不同，不能相加，也不能当成 NVIDIA 独立显存峰值。当前只说明两个参考进程在这些边界上的状态，没有建立资源下降比例或无泄漏结论。

本地已保存开发目录清单 `macos-workspace.json`、两份环境版本清单和关键模型文件哈希 `model-files.json`。辅助资源包括 CNHuBERT、中文 RoBERTa、V2Pro 说话人编码器、语言识别模型及官方 G2PW。

当前目录还包含两个 Python 环境、下载缓存、上游自带的默认模型和 G2PW 压缩包。这是开发对照环境，尚未裁成发行包；不能把目录占用与用户之前的整合包直接比较，也没有证明最终安装体积已经降到目标值。

## 本次处理的问题

| 问题 | 原因与处理 | 证据 |
|---|---|---|
| 官方导入报缺少 `matplotlib` | 推理模型的导入链包含 Lightning scheduler，补齐独立官方环境依赖 | `logs/official-import.log` |
| 官方首次合成报 `fast-langdetect: Cache directory not found` | 上游相对缓存目录不存在；创建共享目录并链接到官方预期位置，再运行 | `logs/official-mps-first.log`；失败的 `result.json` 已保留 |
| 中文首次触发 G2PW 下载 | 官方中文前端需要该资源，按原实现下载并运行 | `logs/official-mps-second.log` |

未人工修改上游 Python 源码。官方执行后自动更新 `GPT_SoVITS/text/g2pw/polyphonic.md5`，生成 `text/ja_userdic/user.dict` 和 `userdict.md5`，因此运行后的官方工作区不是完全干净；这几个生成文件保留在参考目录。Lite 和 Genie 工作区检查为干净。最终状态另存于 `verification.json`，未改写原始运行结果。

## 复现

本次测试的 Mac 已准备好环境和资源；仅克隆本仓库不会自动安装它们。两个后端顺序运行，避免彼此争用 GPU。从 SakuraTTS 仓库根目录执行以下命令，每次创建新的结果目录，不覆盖历史结果：

```sh
TTS_PROJECT_DIR="$PWD"
TTS_REF_DIR="$(dirname "$TTS_PROJECT_DIR")/SakuraTTS-References"

env -u PYTORCH_ENABLE_MPS_FALLBACK \
  "$TTS_REF_DIR/.venv-macos/bin/python" -u \
  "$TTS_PROJECT_DIR/harness/reference_smoke.py" \
  --references "$TTS_REF_DIR" --backend lite --device mps \
  --languages ja zh --repeat 2

env -u PYTORCH_ENABLE_MPS_FALLBACK \
  "$TTS_REF_DIR/.venv-official-macos/bin/python" -u \
  "$TTS_PROJECT_DIR/harness/reference_smoke.py" \
  --references "$TTS_REF_DIR" --backend official --device mps \
  --languages ja zh --repeat 2
```

[Harness](../../harness/reference_smoke.py) 保存 `result.json`、模型来源、官方配置和每轮音频；发生错误时保留完整 traceback 并返回失败。本次已完成的两个运行目录还保存了执行脚本副本及 `verification.json`，用于核对 WAV、脚本哈希和源仓库状态。依赖与资源的详细布局保存在本机参考目录的 `README.md`。

## 原始结果与验证边界

| 本地证据 | 路径 |
|---|---|
| Lite 结果与 WAV 校验 | `runs/20260919T094124.872669Z-lite-mps/` 下的 `result.json` 和 `verification.json` |
| Lite 日志 | `logs/lite-mps-first.log` |
| 官方结果与 WAV 校验 | `runs/20260919T094645.650008Z-official-mps/` 下的 `result.json` 和 `verification.json` |
| 官方日志 | `logs/official-mps-second.log` |
| 官方首次失败记录 | `runs/20260919T094526.697377Z-official-mps/result.json` |

已验证：8 个 WAV 均为 32 kHz、单声道、PCM16，样本非空且有限，存在非零信号，文件哈希与运行记录一致；13 个角色资源与原压缩包一致；两个环境的实际安装包分别为 55 个和 107 个，均与冻结清单一致。

尚未验证：人工听感、文字内容完整性及 ASR、跨后端数值等价、流式首包、长句与混合语言、取消及角色切换、长期资源稳定性、纯离线干净安装、原生运行时和 Windows / CUDA。完整 [基准与验收协议](../specs/benchmark-protocol.md) 仍待执行，M0 不因本次冒烟通过而视为完成。

Mac 上的下一步是核对样音内容，导出固定文本前端及中间张量作为行为对照，再测量各阶段开销。先找到这组真实模型的计算和资源成本，随后决定转换与优化的切入点。
