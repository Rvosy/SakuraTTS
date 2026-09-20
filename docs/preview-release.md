# 开发者预览版

SakuraTTS `0.1.0a1` 提供 Python 包和源码，当前重点是 Windows / NVIDIA 上的日文完整 WAV 推理。GitHub 保存源码；发行者手动构建 ZIP，再把 ZIP 与校验文件上传 ModelScope。项目自身采用 MIT，第三方代码与资源保留各自许可。

这是开发者预览版。安装 wheel 后仍需准备运行依赖、模型、日文前端、参考条件和独立声学工作进程。发行 ZIP 不包含 Python、CUDA/cuDNN、模型权重、参考音频或词典，不是离线整合包。普通推理不导入 PyTorch；模型转换与参考准备使用单独的开发环境。

## 下载内容

解压手动发布的 `sakuratts-0.1.0a1-preview.zip`：

- `dist/sakuratts-0.1.0a1-py3-none-any.whl`：安装到 Python 环境的运行代码。
- `dist/sakuratts-0.1.0a1.tar.gz`：源码包，含转换脚本、测试、验证工具及文档。
- `requirements-windows-runtime.txt`：本轮 Windows 主环境的固定依赖。
- `QUICKSTART.md`：本说明；源码包中的 `docs/` 含详细指南。
- `release-manifest.json`、`SHA256SUMS`：构建版本、源码身份及包内文件校验。

ZIP 旁的 `.sha256` 文件用于下载后的完整性核对。PowerShell 可用 `Get-FileHash <下载的ZIP> -Algorithm SHA256` 计算哈希，与该文件比较。文件名中的版本号不替代哈希。

## 安装主环境

当前实测使用 Windows x64、Python 3.11、RTX 5060 8 GB。先安装 NVIDIA 驱动和 [uv](https://docs.astral.sh/uv/getting-started/installation/)，把运行环境放在只含 ASCII 字符的路径（例如 `D:/SakuraTTS/`），然后在解压目录运行：

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-windows-runtime.txt
uv pip install --python .venv/Scripts/python.exe --no-deps dist/sakuratts-0.1.0a1-py3-none-any.whl
uv pip check --python .venv/Scripts/python.exe
.venv/Scripts/sakuratts.exe --version
.venv/Scripts/sakuratts.exe doctor --japanese
```

以上安装命令允许从包索引获取依赖；缓存齐全时，可在每条 `uv` 命令中加 `--offline`。首次安装不要默认自己已有发行者的缓存。不要直接复制开发机的 `.venv`，也不要把 `onnxruntime-gpu` 覆盖安装进含 CPU `onnxruntime` 的主环境。

实际安装验证发现，CuPy / NVRTC 的 CUDA 首次编译无法读取非 ASCII 路径中的 `cupy/complex.cuh`。需要将 Python 环境及 CUDA 头文件放在 ASCII 路径，重新创建环境后安装 wheel；不要仅搬移现有 venv。空格与中文/日文字符是不同情况，限制针对非 ASCII 头文件路径，不限制日文文本或模型参考名称。

也可以从源码安装。将源码包解压到新目录，在其中执行相同的 requirements 安装，最后改为 `uv pip install --python <环境中的python.exe> --no-deps .`。需要改代码时再使用 `-e .`。`.[japanese,nvidia]` 只声明主环境的直接依赖，复现本轮环境优先使用固定清单。

## 模型、前端和声学工作进程

先解开 `dist/sakuratts-0.1.0a1.tar.gz`，其中的 `docs/setup-windows-nvidia.md` 给出各准备工具及完整命令，也可阅读 [GitHub 上的 Windows 指南](https://github.com/Rvosy/SakuraTTS/blob/feat/reference-parity/docs/setup-windows-nvidia.md)。独立安装后的普通请求需要：

| 资源 | 准备方式 |
| --- | --- |
| GPT / SoVITS 推理包 | 从自己的原始权重转换，保留模型身份与校验信息 |
| 日文前端和参考条件 | 从同一官方版本及参考音频准备；经典版和 plus 前端不能互换 |
| 声学工作进程 | 目前使用 CPython 3.9、NumPy 1.23.4、ORT CUDA 1.19.2 及匹配的 CUDA DLL |
| `runtime.json` | 填写上述资源路径，路径相对配置文件解析 |

当前 `scripts/prepare_ort_worker_runtime.py` 从已有官方运行目录及本地 NVIDIA wheel 文件导出独立组件，不是一个从空机器自动安装全部资源的工具。首次使用者仍需这些准备输入；它们不在本次开发者 ZIP 中。导出完成后，普通推理读取独立组件与包内资源，不读取原官方项目。

`examples/runtime.windows.example.json` 提供配置格式。示例中的路径和 `neutral` 只是占位，必须指向自己的真实产物。不要沿用文档中开发机的绝对路径。开发转换依赖按 Windows 指南安装 `.[japanese,nvidia,dev]`；日常主环境只装 runtime 清单。

## 命令行和 Python 接口

原 FP32 整图配置可先检查，再生成音频：

```powershell
.venv/Scripts/sakuratts.exe doctor --nvidia --config <runtime.json>
.venv/Scripts/sakuratts.exe synthesize --config <runtime.json> `
  --text "おはよう。今日もよろしくね。" --seed 1234 --output outputs/hello.wav
```

输出目录可以不存在，但 WAV 和同名 JSON 文件必须尚不存在。退出码 `0` 表示正常停止；`2` 表示触及生成上限，不能当作文本完整的证明。其他非零退出码表示配置或运行失败。

连续请求应在同一进程内复用引擎：

```python
from sakuratts.nvidia import NVIDIAEngine, write_wav

engine = NVIDIAEngine("runtime.json")
try:
    engine.load()
    pcm, report = engine.synthesize("おはよう。今日もよろしくね。", seed=1234)
    write_wav("hello.wav", pcm, report["sample_rate"])
    # 下一次 synthesize 可复用已加载模型；同一实例一次处理一个请求。
finally:
    engine.close()
```

实测的低显存配置为 GPT FP16 / baseline attention，加已筛查的声学 FP16 分块 256；速度配置改用 GPT FP32 / split-KV 256。两者都需专用声学包、`allow_experimental_acoustic_fp16=True`、`acoustic_arena_shrink=True`、`acoustic_chunk_frames=256`，CLI 对应参数见 Windows 指南。默认 FP32 整图行为保留。未通过完整检查的 GEMV 实验没有进入运行包的执行路径。

`doctor --nvidia --config` 暂只覆盖原 FP32 整图包，不接受声学 FP16 / 分块包。这些候选需实际合成检查，不能用一次依赖检查替代推理验收。

## 当前范围

已在本机验证 Sakura V2ProPlus、日文单请求、完整 WAV 和模型生命周期。速度配置自然生成 25.90 秒完整 PCM 的热请求中位数为 1557.08 ms；低显存配置生成 25.46 秒 PCM 为 2373.28 ms。两者工作量不同，不作为严格横向加速比。资源轮测得的全卡增量分别为 1108 MiB 和 799 MiB，包含桌面负载，也可能漏采瞬时峰值，不是进程独占显存。

预览版尚未承诺其他 GPU、最低显存或其他模型家族兼容；不提供中文整链、量化、流式播放、并发服务或 Sakura 宿主集成。声学 FP16 保留相对官方 FP32 的严格数值失败。24 条保存音频已做辅助 ASR 检查，部分内容仍待复听，音色和自然度未获人工验收。详见源码包中 `docs/experiments/` 下的 ASR 和分块入口记录，或 GitHub 上的 [ASR 记录](https://github.com/Rvosy/SakuraTTS/blob/feat/reference-parity/docs/experiments/2026-09-20-windows-asr.md) 与 [分块入口记录](https://github.com/Rvosy/SakuraTTS/blob/feat/reference-parity/docs/experiments/2026-09-20-windows-vocoder-public.md)。

## 手动构建和上传

在源码根目录、Python 3.11 环境中执行：

```powershell
python scripts/build_preview.py --output dist/preview-0.1.0a1
```

本机缓存齐全时加 `--offline`。输出目录必须是新目录。脚本需要 `uv`，在临时源码目录构建 wheel 与 sdist，再生成 ZIP、源码哈希清单与 ZIP 校验文件，不修改现有环境，不联网上传。它按明确目录和文件类型收集源码，不复制模型、实验输出、虚拟环境或本机构建缓存。构建清单保留 Git commit 与工作区是否有修改；从不含 `.git` 的源码包重建时，Git 状态记为不可用。

发布前从该 wheel 在独立环境安装并生成音频，检查源码包能重新构建，核对 ZIP 清单。确认产物后，在 ModelScope 自己选定的仓库中手动上传 ZIP 与 `.sha256`，仓库说明可使用 `docs/releases/0.1.0a1.md`。GitHub 只提交源码与文档；`dist/` 已忽略，不把 ZIP 或 wheel 加入 Git。

下载页应标注版本、源码 commit、Python 和系统范围、包内不含的运行资源以及已知限制。若日后另行发布模型、词典或运行库整合包，需要为那份产物单独记录来源和许可，不能套用本项目 MIT 许可。
