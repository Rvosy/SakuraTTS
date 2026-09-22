# Windows / NVIDIA 整合包

整合包可以只用本地文件构建。默认发行组合包含 HTTP、CLI、日文推理及私有 Python/CUDA 运行库；完整包另外带独立准备组件，让用户提供原始权重和新参考音频后直接调用 HTTP。PyTorch、原版准备源码和公共辅助模型放在准备组件中，按需启动 CPU 进程，完成后退出。HTTP 和准备组件可以分别省略。

下载并解压到 ASCII 路径（允许空格），双击 `check-runtime.bat` 检查 GPU，双击 `start-server.bat` 启动服务。无需系统 Python 或 CUDA Toolkit，系统仍须安装兼容的 NVIDIA 驱动。没有模型时服务保持未配置状态，不选择本机角色。

## 内容与边界

```text
SakuraTTS-Windows-NVIDIA/
  start-server.bat          启动 HTTP 服务
  sakuratts.bat             CLI
  check-runtime.bat         实际 GPU 运算检查
  runtime/
    main/                  Python 3.11、前端、CuPy、HTTP
    acoustic/              Python 3.9、独立 ORT GPU ABI；当前也供经典日文前端使用
    preparation/           完整包：准备 Python、必要上游源码、公共辅助资源
    bin/                   FFmpeg，音频读取与 HTTP 编码
    portable.json          当前发行组合、前端和声学解释器路径
  models/                  由用户导入模型
  configs/                 中性模板、四个推理档位
  licenses/
  logs/
  cache/
  bundle-manifest.json      版本、文件哈希与共享库清单
```

发声模型全部排除：开发机角色、GPT/SoVITS 官方底模、转换结果、参考音频、参考条件、生成音频和个人配置均不进入整合包。“模型包”是用户明确选择的 GPT 语义模型与 SoVITS 声学模型转换后的部署目录，可来自官方底模或用户微调权重，不是 Python/CUDA 运行环境。

完整包的 `runtime/preparation/` 携带 HuBERT、Pro 说话人编码器、语言识别模型、日文字典和必要的原版源码。用户不需要配置内部解释器或源码路径。精简包省略该目录，支持已有转换模型和预先准备的参考条件；尝试转换原始权重或处理新参考时，会提示缺少准备组件。

`configs/low-vram.json` 和 `configs/minimum-vram.json` 保留低显存选择，需要匹配的 FP16 chunk256 声学包，不能通过切换配置自动转换模型。极限档支持 CLI/Python 和 managed HTTP；FP16 听感验收未完成，默认仍保持 FP32。详见[推理档位](inference-profiles.md)。

## 首次使用完整包

1. 将自己的 GPT `.ckpt`、SoVITS `.pth` 放在 `models/`，或保留在其他目录。
2. 复制 `configs/tts_infer.example.yaml` 为 `configs/tts_infer.yaml`，填写 `t2s_weights_path` 和 `vits_weights_path`。
3. 双击 `start-server.bat`。终端依次显示前端准备、GPT 转换、SoVITS 转换；完成后开始接收请求。
4. 按[原版 HTTP 字段](http-api.md)提交 `/tts`，包含 `ref_audio_path`、`prompt_text`、`prompt_lang` 等。首次参考编码自动完成，返回音频后缓存可复用。

整个过程使用用户选定的两份权重，不需要手工执行转换命令或提供内部模型目录。当前范围为 V2ProPlus、日文，原版尚未支持的参数仍按兼容清单报错。没有 `configs/tts_infer.yaml` 时保持未配置状态，不选择任何默认角色。

后台应用可加 `--runtime-mode managed`。首次转换也计入唤醒等待，慢机器可显式设置 `--wake-timeout-seconds 900`；普通重启命中缓存后不需要再次导出。第一次处理参考的时间也包含在请求时限内。准备默认使用 CPU FP32，主推理默认模式和精度不变。

模型转换缓存和参考缓存保存在 `cache/`。整包搬迁会绑定新位置的内部资源；准备组件或转换器改变时会生成相应的新缓存。升级时先解压到新目录，再复制自己的模型、参考、配置及需要保留的缓存，不用模板覆盖个人配置。

## 原版带的模型与本项目的边界

核对 [GPT-SoVITS 固定版本配置](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/GPT_SoVITS/configs/tts_infer.yaml) 和[预训练资源说明](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/README.md#pretrained-models)，可以区分以下资源。本机现有原版目录也包含这些预训练资源，但不能据此推断每个官方发行版本都包含相同文件。

| 类别 | 例子与作用 | SakuraTTS 整合包的处理 |
| --- | --- | --- |
| 官方发声底模 | V2ProPlus 配置中的 `s1v3.ckpt` 与 `v2Pro/s2Gv2ProPlus.pth`，分别负责语义生成和声学生成 | 默认不捆绑，可另行准备和转换；尚未验证的底模不能直接宣称兼容 |
| 用户微调模型 | 用户自己的 GPT `.ckpt`、SoVITS `.pth`，以及转换后的推理图和权重 | 用户自行导入，不读取或打包开发机当前模型 |
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

`--recipe` 选择这份配置，省略时使用上述默认文件。它只决定发行内容，FP32/FP16 和 direct/managed 仍由推理配置选择，默认 direct + FP32 不变。

| 选择 | 构建结果 |
| --- | --- |
| `services = ["http"]` | 包含 HTTP 依赖和 `start-server.bat`，保留 SDK/CLI |
| `services = []` | 只保留 SDK/CLI，省去 HTTP 依赖和启动脚本 |
| 提供 `--preparation LOCAL_COMPONENT` | 带原始权重转换、新参考编码所需的独立 CPU 准备组件 |
| 省略 `--preparation` | 使用已有转换结果和已准备的参考条件 |

当前实现的目标只有 `windows-x64` + `cuda` + `ja`。其他语言、CPU、AMD 和 Apple 后端需先完成对应实现和验证，再增加发行组合；构建器会直接拒绝未实现的组合。准备组件使用 CPU 不代表支持 CPU 语音推理。

`workers` 分别描述前端和声学运行角色，路径相对于包根目录，必须属于所选发行内容。当前二者共享一套 Python 3.9，配置为两个角色不会多复制一套环境。以后若需要不同的解释器，可以更换对应组件和路径。构建器把角色路径写入 `runtime/portable.json`，推理时再绑定到当前安装位置；模型本身不负责选择安装目录。

语言、设备和 HTTP 依赖仍在 `pyproject.toml` 中分别维护，recipe 选择它们的组合，不复制一份依赖版本表。准备组件沿用独立构建流程，上游源码中的提前导入仍会带入部分其他语言依赖；修改 `languages` 不会自动裁剪这些依赖。

先用本地缓存构建产品 wheel。历史构建留下的 `build/lib` 可能包含已删除模块，因此应在干净的源码快照中构建，或先清理仓库内已确认的 `build/` 目录。

```powershell
uv build --offline --wheel --no-python-downloads --out-dir dist/portable-wheel
python scripts/build_preparation.py `
  --python-base LOCAL_PREPARATION_PYTHON `
  --site LOCAL_PREPARATION_SITE_PACKAGES `
  --runtime-site LOCAL_CPU_TORCH_SITE_PACKAGES `
  --official-source LOCAL_SUPPORTED_GPT_SOVITS_SOURCE `
  --output dist/preparation `
  --audit tmp/preparation-inputs.json
python scripts/build_portable.py `
  --recipe packaging/recipes/windows-nvidia-ja.toml `
  --python-base LOCAL_CPYTHON_311 `
  --main-site LOCAL_RUNTIME_SITE_PACKAGES `
  --ffmpeg LOCAL_FFMPEG_EXE `
  --worker LOCAL_VERIFIED_ORT_WORKER `
  --wheel dist/portable-wheel/sakuratts-0.1.0a1-py3-none-any.whl `
  --preparation dist/preparation `
  --output dist/SakuraTTS-Windows-NVIDIA `
  --audit tmp/portable-inputs.json
```

构建脚本需要 `packaging`，使用已有开发 Python 即可；不执行安装或下载。`--plan-only` 只校验并列出构建输入。输出目录必须不存在。源文件按安装包 RECORD 和声学组件哈希清单选取并复核，不整体复制 venv，不复制可编辑安装、`.pth`、字节码或开发配置。绝对来源路径只写在包外的本地 audit；发行清单不包含这些路径。

省略 `--preparation` 就构建精简包。准备组件支持本地 CPython 3.9 嵌入式解释器和 CPython 3.11，依赖必须与其 ABI 匹配。它使用 CPU 版 PyTorch / torchaudio；`--runtime-site` 可指定已有 CPU 组件，覆盖源环境中的对应依赖，不修改原环境。如果源环境本来就是 CPU 版，可以省略此参数。构建器不联网安装缺失依赖。

准备组件不带头文件、静态链接库和依赖的测试目录。若本地只有 GPU 版 ORT，保留其 CPU 核心，排除准备阶段不会使用的 CUDA / TensorRT provider。原版 TTS 导入时仍会加载部分训练相关库，目前保留这些实际依赖。主推理环境保留 NVRTC 所需的 NVIDIA、CuPy 和 NumPy 头文件，不能套用准备环境的裁剪规则。

准备组件分别生成 `preparation.json`、`preparation-manifest.json` 和 `licenses.json`。源码及辅助资源采用白名单，开发者配置不复制；官方发声底模、用户角色和参考均排除。`licenses.json` 会标明本地输入缺少的辅助权重许可信息，不能用上游代码的 MIT 许可代替权重许可。公开分发前仍需补齐这些来源声明。

本地安装文件若与 wheel 的 RECORD 不同，构建默认失败。只有核对过的修改才能通过 `--record-overrides` 显式提供相对文件名、原始 `record_sha256`、当前 `sha256` 和 `reason`；这些信息写入准备组件清单。不能用该选项批量忽略校验。

两个解释器共享内容完全一致的 NVIDIA DLL，声学 worker 显式从包内主环境加载，保留不同版本的库。共享与去重将本次运行输入从约 4.84 GB 减为 3.85 GB；这也包含移除未使用的 ffprobe。余下体积主要来自 ORT CUDA、cuDNN、cuBLAS 和日文字典。不能仅凭单卡冒烟通过就删除其他架构或算子需要的库。

启动器绑定包内 Python，清理外部 Python/CUDA 路径，将缓存放在包内。模型中旧的声学和前端 Python 路径在运行时按 `workers` 改为当前包内解释器；个人权重路径不变。准备路径不会回退到开发环境。暂不支持非 ASCII 的解压目录；模型和参考路径可含中文。

FFmpeg 保留本地二进制的版本、编译选项和 LGPL 说明；组件许可证随包携带。本地拼装与运行验证不等于完整公开发行合规审计，正式对外分发前仍需核对所携带第三方二进制的对应源码及再分发要求。

## 7z 压缩

```powershell
python scripts/archive_portable.py --bundle dist/SakuraTTS-Windows-NVIDIA `
  --sevenzip "C:/Program Files/7-Zip/7z.exe" --output tmp/7z-benchmark --benchmark
python scripts/archive_portable.py --bundle dist/SakuraTTS-Windows-NVIDIA `
  --sevenzip "C:/Program Files/7-Zip/7z.exe" --output dist/portable-release --profile maximum
```

采样比较 LZMA2 solid 的 `mx=5/7/9`、32/64/128 MiB 字典，固定 2 个压缩线程，记录压缩大小、耗时、校验和解压耗时。样本取自体积最大的 12 个文件的多个位置，结果只用于选参，不代表完整包的压缩率。正式压缩只读取发行清单中的文件，并重新校验哈希；验收产生的缓存、日志、音频及用户后来放入的模型都不会收录。输出 `.7z`、SHA256 和压缩报告。

早期精简包使用约 302 MB 样本，实测如下。最高档比快速档小约 1.6%，多耗时约 16 秒，解压耗时相近。考虑下载流量，最终选择 `mx=9`、128 MiB 字典；快速开发打包可使用 `balanced`。

| 档位 | 压缩大小 | 压缩耗时 | 解压耗时 |
| --- | ---: | ---: | ---: |
| mx=5 / 32 MiB | 99.19 MB | 36.90 秒 | 2.42 秒 |
| mx=7 / 64 MiB | 98.70 MB | 51.12 秒 | 2.94 秒 |
| mx=9 / 128 MiB | 97.57 MB | 53.33 秒 | 2.93 秒 |

该轮精简运行包为 1,313,042,840 字节（约 1.31 GB），清单内文件解压后为 3,851,802,388 字节（约 3.85 GB，不含清单自身）。完整压缩耗时 598 秒，`7z t` 校验耗时 24 秒；这些是历史单机记录，不代表本轮完整包。

该轮压缩包重新解压耗时 31.6 秒，6,361 个文件的 SHA256 全部匹配，没有清单外文件；解压后的副本再次通过 GPU 检查。历史报告保存在本地 `dist/portable-release/`。

本轮完整包改用 CPU 准备组件，并清理重复的开发文件和校验步骤；解压后的清单文件从 10.90 GB 降到 5.51 GB。最终 7z 为 **1,960,862,106 字节（1.961 GB）**，低于 2 GB，`7z t` 校验通过。实际首次使用、解压验收及继续缩减的方向见[完整包缩减记录](../research/notes/portable-compact-20260922.md)，本地发行物位于 `dist/compact/release/`。

## 验证范围

首次使用验收使用新的完整包副本和空的模型／参考缓存：

```powershell
python scripts/verify_portable_first_use.py `
  --bundle dist/SakuraTTS-Windows-NVIDIA `
  --gpt D:/Voices/voice.ckpt --sovits D:/Voices/voice.pth `
  --reference D:/Voices/reference.wav --prompt-text "参考音声です。" `
  --output outputs/portable-first-use
```

运行验收脚本的开发解释器需要 `psutil`，被测服务及准备、推理子进程都使用包内解释器。脚本保留已有缓存，发现模型或参考缓存非空就拒绝首次使用测试；报告和音频写到包外。验收覆盖原始模型转换、新参考编码、重复请求、重启缓存、默认 direct、managed 休眠和进程退出，并比较返回的 PCM。它不代替干净机器、峰值显存和人工听音测试。

搬迁保留缓存的整包后，可用相同参数加 `--reuse-cache`，并指定新的输出目录。该模式只验证已有缓存的复用，不把它计为首次转换通过。

以下体积、解压与 GPU 运算记录属于早期精简包，不代表新完整包已通过同样的全部验收。

本机 RTX 5060、驱动 610.62 已通过 NVRTC、FP32/FP16 GEMM、CUDA Graph 和独立 ORT CUDA 实际运算；也用包外模型完成了日文短句合成。GPU 检查不代表音质验收，冷启动短句耗时也不能作为稳定性能指标。

包目录复制到带空格的新位置后，同样通过 GPU 检查与无模型 HTTP `/health` 检查。测试故意设置了错误的外部 `PYTHONHOME`、`PYTHONPATH`、`CUDA_PATH`、`CUDA_HOME`；主 Python 的文件审计同时阻止读取原开发环境。服务返回 `ready`、`model_loaded=false`。这项测试在同一台 Windows 上进行，不能替代干净机器验收。

搬迁后的包还通过了 FP16 低显存档短句合成，使用包外的 chunk256 声学模型与已准备参考。FP32 和低显存档都生成了 32 kHz、单声道、4.54 秒的 WAV；本轮未重新测量峰值显存或评价音质。模型、参考与输出均未进入发行清单。

其他 Windows 设备，尤其 Turing、Ampere、Ada、笔记本和低显存型号仍需测试。同一 NVIDIA 包不承诺覆盖 CPU、AMD、Intel GPU 或其他操作系统。干净 Windows 上无开发目录的安装、最低驱动、长句、多轮请求、取消恢复、完整显存测量和人工听音仍属于发行验收范围。
