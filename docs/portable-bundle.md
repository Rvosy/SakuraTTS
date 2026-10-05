# Windows 整合包

Apple silicon 的独立运行组件、构建与验收见 [Mac 整合包指南](portable-macos.md)。

整合包可以只用本地文件构建。默认发行组合包含 HTTP、CLI、日英混合推理及私有 Python 和所选后端的运行库；完整包另外带独立准备组件，让用户提供原始权重和新参考音频后直接调用 HTTP。PyTorch、原版准备源码和公共辅助模型放在准备组件中，按需启动 CPU 进程，完成后退出。HTTP 和准备组件可以分别省略。

提供两个主要发行组合：CPU/AMD 共用 DirectML ONNX Runtime，可用配置或 `--backend cpu|directml` 切换；NVIDIA 使用独立 CUDA 组合。无需按显卡代际拆包。带准备组件的完整包可处理原始权重和新参考；精简包仅适合已有转换资源，按实际分发需求选择，不必同时发布两种大小。

解压后使用 `check-runtime.bat` 检查默认设备，再用 `start-server.bat` 启动服务。CPU/AMD 包默认检查 CPU，AMD 另运行 `sakuratts.bat check-runtime --backend directml`。无需系统 Python、Git、编译器或 CUDA Toolkit；GPU 模式仍需要系统驱动。NVIDIA 包要求 ASCII 安装路径（可含空格），CPU/AMD 包不限制中文目录，须对最终解压位置做验收。CPU 首次 INT8 转换支持安装目录和 TEMP/TMP 同时包含中文，无需手动修改系统临时目录。完整安装路径为 ASCII 时，启动器沿用包内 cache/tmp；中文安装路径保留外部 TEMP/TMP。没有模型时服务保持未加载状态，不选择默认角色。

目标系统为 Windows 10/11 x64。当前 NumPy 2.4.6 轮子的 CPU 基线是 x86-64-v2，见 [NumPy 2.4 发布说明](https://numpy.org/doc/2.4/release/2.4.0-notes.html#modulate-dispatched-x86-cpu-features)；不能将“x64”视为所有旧 CPU 均可运行，也不要求为此拆出一个相同依赖的 CPU 包。准备组件的 PyTorch/MKL 完整链路尚未在旧 CPU 验收。DirectML 官方 Windows 10 1903 / DirectX 12 要求只是执行提供程序的基础条件，不等于整个包的最低系统验收结论。

`check-runtime` 的 DirectML 检查按现有 DXGI 编号选择设备，并执行 FP32、FP16 小图；失败保留原始异常，不重试 CPU。`doctor --backend directml` 列出同一编号的适配器，`software: true` 表示软件渲染器，不能据此报告 GPU 通过。代码没有 AMD/Intel 厂商白名单；目前只有 Radeon 780M 的真实执行证据，其他设备仍需验收。缺少包内 Python 时 bat 会提示重新完整解压；DLL 导入异常保留在诊断报告中，系统无需预装 Python 或 Git。

## 内容与边界

```text
SakuraTTS-Windows/
  start-server.bat          启动 HTTP 服务
  sakuratts.bat             CLI
  check-runtime.bat         实际 CPU/GPU 运算检查
  runtime/
    main/                  私有 Python、前端、所选后端、HTTP
    acoustic/              仅 CUDA 组合：独立 ORT GPU ABI 与经典日语前端
    preparation/           完整包：准备 Python、必要上游源码、公共辅助资源
    bin/                   FFmpeg，音频读取与 HTTP 编码
    portable.json          当前发行组合、前端和声学解释器路径
  models/                  由用户导入模型
  configs/                 所选后端的配置模板与推理档位
  licenses/
  logs/
  cache/
  bundle-manifest.json      版本、文件哈希与共享库清单
```

发声模型全部排除：开发机角色、GPT/SoVITS 官方底模、转换结果、参考音频、参考条件、生成音频和个人配置均不进入整合包。“模型包”是用户明确选择的 GPT 语义模型与 SoVITS 声学模型转换后的部署目录，可来自官方底模或用户微调权重，不是 Python/CUDA 运行环境。

完整包的 `runtime/preparation/` 携带 HuBERT、Pro 说话人编码器、语言识别模型、日文字典、离线英语资源和必要的原版源码。用户不需要配置内部解释器或源码路径。精简包省略该目录，支持已有转换模型和预先准备的参考条件；尝试转换原始权重或处理新参考时，会提示缺少准备组件。

NVIDIA 包的 `configs/low-vram.json` 和 `configs/minimum-vram.json` 保留低显存选择，需要匹配的 FP16 chunk256 声学包，不能通过切换配置自动转换模型。极限档支持 CLI/Python 和 managed HTTP；FP16 听感验收未完成，默认仍保持 FP32。详见[推理档位](inference-profiles.md)。

## 首次使用完整包

1. 将自己的 GPT `.ckpt`、SoVITS `.pth` 放在 `models/`，或保留在其他目录。
2. 复制 `configs/tts_infer.example.yaml` 为 `configs/tts_infer.yaml`，填写 `t2s_weights_path` 和 `vits_weights_path`。
3. 双击 `start-server.bat`。终端依次显示前端准备、GPT 转换、SoVITS 转换；完成后开始接收请求。
4. 按[原版 HTTP 字段](api-v2-guide.md)提交 `/tts`，包含 `ref_audio_path`、`prompt_text`、`prompt_lang` 等。首次参考编码自动完成，返回音频后缓存可复用。

整个过程使用用户选定的两份权重，不需要手工执行转换命令或提供内部模型目录。当前范围为 V2ProPlus、日文，尚未接入的原版参数按 API 支持表报错。没有 `configs/tts_infer.yaml` 时保持未配置状态，不选择任何默认角色。

后台应用可加 `--runtime-mode managed`。首次转换也计入唤醒等待，慢机器可显式设置 `--wake-timeout-seconds 900`；普通重启命中缓存后不需要再次导出。第一次处理参考的时间也包含在请求时限内。准备默认使用 CPU FP32，主推理默认模式和精度不变。

模型转换缓存和参考缓存保存在 `cache/`。整包搬迁会绑定新位置的内部资源；准备组件或转换器改变时会生成相应的新缓存。升级时先解压到新目录，再复制自己的模型、参考、配置及需要保留的缓存，不用模板覆盖个人配置。

## 原版带的模型与本项目的边界

核对 [GPT-SoVITS 固定版本配置](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/GPT_SoVITS/configs/tts_infer.yaml) 和[预训练资源说明](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/README.md#pretrained-models)，可以区分以下资源。具体文件以所用上游版本的资源清单为准。

| 类别 | 例子与作用 | SakuraTTS 整合包的处理 |
| --- | --- | --- |
| 官方发声底模 | V2ProPlus 配置中的 `s1v3.ckpt` 与 `v2Pro/s2Gv2ProPlus.pth`，分别负责语义生成和声学生成 | 另行准备与转换，按具体权重验证 |
| 用户微调模型 | 用户自己的 GPT `.ckpt`、SoVITS `.pth`，以及转换后的推理图和权重 | 用户自行导入，不进入发行清单 |
| 公共辅助模型与字典 | HuBERT 用于参考音频特征，Pro 系列的说话人编码器；中文 BERT、G2PW 用于中文链路；语言识别模型和日文字典用于前端 | 按已支持功能建立版本、来源、哈希和许可清单，独立获取；当前日文路径不因上游包含中文模型就全部捆绑 |
| 个人参考与缓存 | 参考录音、转写、提取出的语义与音色条件、已生成音频 | 不分发；在用户设备上显式选择并生成 |
| 训练与数据处理资源 | 训练用判别器、ASR、UVR 人声分离等 | 不属于推理整合包的默认范围 |

底模与微调模型不是每次都必须同时加载的两层。当前转换器读取用户明确选择的 GPT、SoVITS 检查点，并检查其中的结构与权重；是否需要额外基础权重取决于具体模型格式。当前支持范围不据此扩展到所有增量或 LoRA 模型。

## 离线构建

发行组合在 [packaging/recipes/windows-nvidia-ja.toml](../packaging/recipes/windows-nvidia-ja.toml) 中声明：

```toml
target = "windows-x64"
backend = "cuda"
languages = ["ja"]
services = ["http"]

[workers]
acoustic = "runtime/acoustic/python.exe"
frontend = "runtime/acoustic/python.exe"
```

`--recipe` 选择发行组合，省略时使用上面的 CUDA recipe。推理档位与 direct/managed 仍由运行时选择：CPU 默认 INT8 GPT / FP32 声学，DirectML 默认 FP16，CUDA 默认 FP32；默认生命周期仍是 direct。

| 选择 | 构建结果 |
| --- | --- |
| `services = ["http"]` | 包含 HTTP 依赖和 `start-server.bat`，保留 SDK/CLI |
| `services = []` | 只保留 SDK/CLI，省去 HTTP 依赖和启动脚本 |
| 提供 `--preparation LOCAL_COMPONENT` | 带原始权重转换、新参考编码所需的独立 CPU 准备组件 |
| 省略 `--preparation` | 使用已有转换结果和已准备的参考条件 |

CPU/AMD 使用 [windows-cpu-amd-ja.toml](../packaging/recipes/windows-cpu-amd-ja.toml)，`backend = "directml"` 表示装入 DirectML ORT 及其 CPU provider。它仅选择 `japanese-text`，不会混装 CPU ORT，也没有独立前端或声学 worker。NVIDIA 继续使用上面的 recipe。`languages` 支持 `ja` 和可选的 `en`；这只是依赖选择，首次转换当前生成日文前端，英文需另行准备并验收资源。

`workers` 分别描述前端和声学运行角色，路径相对于包根目录，必须属于所选发行内容。CUDA 组合中二者共享一套 Python 3.9，配置为两个角色不会多复制一套环境。CPU/AMD 留空，在主环境执行。构建器把角色路径写入 `runtime/portable.json`，推理时再绑定到当前安装位置；模型本身不负责选择安装目录。

语言、设备和 HTTP 依赖仍在 `pyproject.toml` 中分别维护，recipe 选择它们的组合，不复制一份依赖版本表。准备组件沿用独立构建流程，上游源码中的提前导入仍会带入部分其他语言依赖；修改 `languages` 不会自动裁剪这些依赖。

先用本地缓存构建产品 wheel。历史构建留下的 `build/lib` 可能包含已删除模块，因此应在干净的源码快照中构建，或先清理仓库内已确认的 `build/` 目录。

```powershell
uv build --offline --wheel --no-python-downloads --out-dir dist/portable-wheel
python scripts/build_preparation.py `
  --python-base LOCAL_PREPARATION_PYTHON `
  --vc-runtime LOCAL_MICROSOFT_VC143_CRT `
  --site LOCAL_PREPARATION_SITE_PACKAGES `
  --runtime-site LOCAL_CPU_TORCH_SITE_PACKAGES `
  --official-source LOCAL_SUPPORTED_GPT_SOVITS_SOURCE `
  --language-model LOCAL_LID_176_BIN `
  --english-resources LOCAL_ENGLISH_RESOURCES `
  --output dist/preparation `
  --audit tmp/preparation-inputs.json
python scripts/build_portable.py `
  --recipe packaging/recipes/windows-cpu-amd-ja.toml `
  --python-base LOCAL_CPYTHON_BASE `
  --vc-runtime LOCAL_MICROSOFT_VC143_CRT `
  --main-site LOCAL_RUNTIME_SITE_PACKAGES `
  --ffmpeg LOCAL_FFMPEG_EXE `
  --wheel dist/portable-wheel/sakuratts-0.1.0a1-py3-none-any.whl `
  --preparation dist/preparation `
  --output dist/SakuraTTS-Windows-CPU-AMD `
  --audit tmp/portable-inputs.json
```

构建脚本需要 `packaging`，使用已有开发 Python 即可；不执行安装或下载。`--plan-only` 只校验并列出构建输入。输出目录必须不存在。源文件按安装包 RECORD 和声学组件清单选取，不整体复制 venv，不复制可编辑安装、`.pth`、字节码或开发配置。绝对来源路径只写在包外的本地 audit；发行清单不包含这些路径。

省略 `--preparation` 可构建精简包。解释器版本从输入中的 `python3X.dll` 读取，主环境与准备环境分别选择，并与各自依赖的 ABI 匹配；支持标准和嵌入式布局。前端扩展模块使用所选环境对应的 Python ABI，不单独限制版本号。准备组件只使用 CPU PyTorch / torchaudio；`--runtime-site` 可覆盖来源目录中的对应包，不修改原环境。源环境本身就是 CPU 版时可省略该参数。构建器不会联网补装依赖。

构建器检查主环境和准备环境的 wheel 标签，拒绝不同系统、架构或 Python ABI 的依赖。准备组件的目标平台也必须与主包一致；旧版未记录目标字段的准备清单按 Windows 处理。平台检查不能代替目标设备上的启动与模型验收。

准备组件不带头文件、静态链接库和依赖的测试目录。若本地只有 GPU 版 ORT，保留其 CPU 核心，排除准备阶段不会使用的 CUDA / TensorRT provider。原版 TTS 导入时仍会加载部分训练相关库，目前保留这些实际依赖。主推理环境保留 NVRTC 所需的 NVIDIA、CuPy 和 NumPy 头文件，不能套用准备环境的裁剪规则。

准备组件分别生成 `preparation.json`、`preparation-manifest.json` 和 `licenses.json`。源码及辅助资源采用白名单，开发者配置不复制；官方发声底模、用户角色和参考均排除。`licenses.json` 会标明本地输入缺少的辅助权重许可信息，不能用上游代码的 MIT 许可代替权重许可。公开分发前仍需补齐这些来源声明。

构建使用当前本地文件，允许打包修改过的依赖；RECORD 用于确定文件范围，不比对原始校验和。发行清单记录实际产物的哈希。

两个解释器共享内容完全一致的 NVIDIA DLL，声学 worker 显式从包内主环境加载，不同版本分别保留。构建产物的实际文件和体积由 `bundle-manifest.json` 记录。

启动器绑定包内 Python，清理外部 Python/CUDA 路径，把缓存放在包内。模型中旧的声学、前端 Python 路径由 `workers` 替换；没有 worker 时使用主环境。NVIDIA 组合保留 CuPy/NVRTC 的 ASCII 安装路径限制，CPU/AMD 组合允许中文和空格。最终包仍需验证启动、转换、重启及搬迁。

FFmpeg 保留二进制版本、编译选项和实际许可说明；GPL 构建不能按 LGPL 分发。公开分发时需提供所携带第三方二进制要求的来源、对应源码与许可材料。

CPU/AMD 主环境使用 `.[directml,japanese-text,server]`，仅包含 `onnxruntime-directml`；它同时提供 CPU 执行。源码安装的 `cpu` extra 继续使用独立 CPU ORT，不改变原有安装契约。NVIDIA 构建改用其 recipe 并提供 `--worker LOCAL_VERIFIED_ORT_WORKER`。

`--vc-runtime` 指向已获准再分发的 Microsoft VC CRT x64 目录，例如 Visual Studio 的 `VC/Redist/MSVC/<版本>/x64/Microsoft.VC143.CRT`。构建器将其 DLL 放到主解释器和准备解释器旁，不依赖开发机的 MSVCP140 安装。CPython 自带的 VCRUNTIME140 不能代替完整 C++ 运行库。目录和解释器均是明确本地输入，不搜索系统、不下载。

`--language-model` 允许从已有缓存选择 `lid.176.bin`；省略时使用上游预训练目录。OpenJTalk 字典按实际安装的 classic 或 plus 前端选择。准备组件携带固定来源、哈希、上游许可声明和 fastText CC-BY-SA 3.0 正文；HuBERT/ERes2Net 的权重归属材料及 FFmpeg 对应源码提供方式仍须在公开分发前核对。来源清单见 [auxiliary-model-sources.json](../packaging/auxiliary-model-sources.json)，不同本地权重不能沿用这份归属结论。

`bundle-manifest.json` 的 `release` 记录产品版本、源码提交、工作树是否有改动、Python 版本和可用后端，`components` 记录实际依赖版本，wheel 与所有文件均有哈希。正式发布应从确定的提交重新构建；工作树试验包会明确标为 `source_dirty: true`。

## 7z 压缩

```powershell
python scripts/archive_portable.py --bundle dist/SakuraTTS-Windows-CPU-AMD `
  --sevenzip "C:/Program Files/7-Zip/7z.exe" --output tmp/7z-benchmark --benchmark
python scripts/archive_portable.py --bundle dist/SakuraTTS-Windows-CPU-AMD `
  --sevenzip "C:/Program Files/7-Zip/7z.exe" --output dist/portable-release --profile maximum
```

采样比较 LZMA2 solid 的 `mx=5/7/9`、32/64/128 MiB 字典，固定 2 个压缩线程，记录压缩大小、耗时、校验和解压耗时。样本取自体积最大的 12 个文件的多个位置，结果只用于选参，不代表完整包的压缩率。正式压缩只读取发行清单中的文件；验收产生的缓存、日志、音频及用户后来放入的模型都不会收录。输出 `.7z`、SHA256 和压缩报告。

压缩选项由 [archive_portable.py](../scripts/archive_portable.py) 的 `PROFILES` 定义。`balanced` 适合开发打包，`maximum` 用于体积优先的发行物；本机样本比较与完整压缩结果分别保存，具体结果见下方记录。

## 验证范围

首次使用验收使用新的完整包副本和空的模型／参考缓存：

```powershell
python scripts/verify_portable_first_use.py `
  --bundle dist/SakuraTTS-Windows-CPU-AMD `
  --backend cpu --profile int8 `
  --gpt D:/Voices/voice.ckpt --sovits D:/Voices/voice.pth `
  --reference D:/Voices/reference.wav --prompt-text "参考音声です。" `
  --output outputs/portable-first-use
```

AMD 验收改用 `--backend directml --profile fp16`，使用独立的空缓存包；NVIDIA 可用 `--backend cuda`。同一次验收比较同一后端的重复 PCM，不要求不同 ORT 或不同后端逐位相同。

运行验收脚本的开发解释器需要 `psutil`，被测服务及准备、推理子进程都使用包内解释器。脚本保留已有缓存，发现模型或参考缓存非空就拒绝首次使用测试；报告和音频写到包外。验收覆盖原始模型转换、新参考编码、重复请求、重启缓存、默认 direct、managed 休眠和进程退出，并比较返回的 PCM。它不代替干净机器、峰值显存和人工听音测试。

搬迁保留缓存的整包后，可用相同参数加 `--reuse-cache`，并指定新的输出目录。该模式只验证已有缓存的复用，不把它计为首次转换通过。

## 验证记录

| 产物与日期 | 记录 |
| --- | --- |
| 2026-09-22 完整包 | [缩减、首次使用、搬迁与解压验收](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/portable-compact-20260922.md) |
| 早期精简推理包 | [压缩、解压与单机运行检查](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/portable-runtime-bundle-20260922.md) |

这些记录中的体积、路径和设备对应当时的构建。查询手头产物应读取 `bundle-manifest.json`、压缩报告及 SHA256 文件。构建和本机验证结果与实际发布状态分别记录。

硬件记录集中于 RTX 5060。其他 Windows 设备、干净机器、最低驱动、长句、多轮请求、取消恢复、完整显存测量和人工听音仍需验收。
