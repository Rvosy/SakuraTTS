# Windows / NVIDIA 整合包

当前提供无模型的推理预览包，可以只用本地文件构建。面向桌宠，优先控制显存和分发体积；主包包含 HTTP、CLI、日文推理及私有 Python/CUDA 运行库，不带 PyTorch、训练工具、转换环境或开发环境。

下载并解压到 ASCII 路径（允许空格），双击 `check-runtime.bat` 检查 GPU，双击 `start-server.bat` 启动服务。无需系统 Python 或 CUDA Toolkit，系统仍须安装兼容的 NVIDIA 驱动。没有模型时服务保持未配置状态，不选择本机角色。

## 内容与边界

```text
SakuraTTS-Windows-NVIDIA/
  start-server.bat          启动 HTTP 服务
  sakuratts.bat             CLI
  check-runtime.bat         实际 GPU 运算检查
  runtime/
    main/                  Python 3.11、前端、CuPy、HTTP
    acoustic/              Python 3.9、独立 ORT GPU ABI
    bin/                   FFmpeg，仅用于 HTTP 音频编码
  models/                  由用户导入模型
  configs/                 中性模板、四个推理档位
  licenses/
  logs/
  cache/
  bundle-manifest.json      版本、文件哈希与共享库清单
```

发声模型全部排除：开发机角色、GPT/SoVITS 官方底模、转换结果、参考音频、参考条件、生成音频和个人配置均不进入整合包。“模型包”是用户明确选择的 GPT 语义模型与 SoVITS 声学模型转换后的部署目录，可来自官方底模或用户微调权重，不是 Python/CUDA 运行环境。

主包包含日文前端依赖需要的公共字典和辅助数据；HuBERT、Pro 说话人编码器及原版准备源码不在主包内。它支持已有转换模型和预先准备的参考条件。原始检查点转换、新参考编码需要另行准备独立组件；本次没有交付准备组件。HTTP 仍要求参考音频路径和转写，只有命中已有参考条件才不需要准备环境。

`configs/low-vram.json` 和 `configs/minimum-vram.json` 保留低显存选择，需要匹配的 FP16 chunk256 声学包，不能通过切换配置自动转换模型。极限档当前仅支持 CLI/Python；FP16 听感验收未完成，默认仍保持 FP32。详见[推理档位](inference-profiles.md)。

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

先用本地缓存构建产品 wheel。历史构建留下的 `build/lib` 可能包含已删除模块，因此应在干净的源码快照中构建，或先清理仓库内已确认的 `build/` 目录。

```powershell
uv build --offline --wheel --no-python-downloads --out-dir dist/portable-wheel
python scripts/build_portable.py `
  --python-base LOCAL_CPYTHON_311 `
  --main-site LOCAL_RUNTIME_SITE_PACKAGES `
  --ffmpeg LOCAL_FFMPEG_EXE `
  --worker LOCAL_VERIFIED_ORT_WORKER `
  --wheel dist/portable-wheel/sakuratts-0.1.0a1-py3-none-any.whl `
  --output dist/SakuraTTS-Windows-NVIDIA `
  --audit tmp/portable-inputs.json
```

构建脚本需要 `packaging`，使用已有开发 Python 即可；不执行安装或下载。`--plan-only` 只校验并列出构建输入。输出目录必须不存在。源文件按安装包 RECORD 和声学组件哈希清单选取并复核，不整体复制 venv，不复制可编辑安装、`.pth`、字节码或开发配置。绝对来源路径只写在包外的本地 audit；发行清单不包含这些路径。

两个解释器共享内容完全一致的 NVIDIA DLL，声学 worker 显式从包内主环境加载，保留不同版本的库。共享与去重将本次运行输入从约 4.84 GB 减为 3.85 GB；这也包含移除未使用的 ffprobe。余下体积主要来自 ORT CUDA、cuDNN、cuBLAS 和日文字典。不能仅凭单卡冒烟通过就删除其他架构或算子需要的库。

启动器绑定包内 Python，清理外部 Python/CUDA 路径，将缓存放在包内。模型中旧的声学 Python 路径在运行时改为当前包内解释器；个人权重路径不变。准备路径不会回退到开发环境。暂不支持非 ASCII 的解压目录；模型和参考路径可含中文。

FFmpeg 保留本地二进制的版本、编译选项和 LGPL 说明；组件许可证随包携带。本地拼装与运行验证不等于完整公开发行合规审计，正式对外分发前仍需核对所携带第三方二进制的对应源码及再分发要求。

## 7z 压缩

```powershell
python scripts/archive_portable.py --bundle dist/SakuraTTS-Windows-NVIDIA `
  --sevenzip "C:/Program Files/7-Zip/7z.exe" --output tmp/7z-benchmark --benchmark
python scripts/archive_portable.py --bundle dist/SakuraTTS-Windows-NVIDIA `
  --sevenzip "C:/Program Files/7-Zip/7z.exe" --output dist/portable-release --profile maximum
```

采样比较 LZMA2 solid 的 `mx=5/7/9`、32/64/128 MiB 字典，固定 2 个压缩线程，记录压缩大小、耗时、校验和解压耗时。样本取自体积最大的 12 个文件的多个位置，结果只用于选参，不代表完整包的压缩率。正式压缩只读取发行清单中的文件，并重新校验哈希；验收产生的缓存、日志、音频及用户后来放入的模型都不会收录。输出 `.7z`、SHA256 和压缩报告。

本次约 302 MB 样本的实测如下。最高档比快速档小约 1.6%，多耗时约 16 秒，解压耗时相近。考虑下载流量，最终选择 `mx=9`、128 MiB 字典；快速开发打包可使用 `balanced`。

| 档位 | 压缩大小 | 压缩耗时 | 解压耗时 |
| --- | ---: | ---: | ---: |
| mx=5 / 32 MiB | 99.19 MB | 36.90 秒 | 2.42 秒 |
| mx=7 / 64 MiB | 98.70 MB | 51.12 秒 | 2.94 秒 |
| mx=9 / 128 MiB | 97.57 MB | 53.33 秒 | 2.93 秒 |

最终运行包为 1,313,042,840 字节（约 1.31 GB），清单内文件解压后为 3,851,802,388 字节（约 3.85 GB，不含清单自身）。完整压缩耗时 598 秒，`7z t` 校验耗时 24 秒；这些是本机单次构建记录，不是其他设备的性能保证。

最终压缩包重新解压耗时 31.6 秒，6,361 个文件的 SHA256 全部匹配，没有清单外文件；解压后的副本再次通过 GPU 检查。压缩报告和验收报告保存在本地 `dist/portable-release/`，与 `.7z` 和 `.sha256` 同目录。

## 验证范围

本机 RTX 5060、驱动 610.62 已通过 NVRTC、FP32/FP16 GEMM、CUDA Graph 和独立 ORT CUDA 实际运算；也用包外模型完成了日文短句合成。GPU 检查不代表音质验收，冷启动短句耗时也不能作为稳定性能指标。

包目录复制到带空格的新位置后，同样通过 GPU 检查与无模型 HTTP `/health` 检查。测试故意设置了错误的外部 `PYTHONHOME`、`PYTHONPATH`、`CUDA_PATH`、`CUDA_HOME`；主 Python 的文件审计同时阻止读取原开发环境。服务返回 `ready`、`model_loaded=false`。这项测试在同一台 Windows 上进行，不能替代干净机器验收。

搬迁后的包还通过了 FP16 低显存档短句合成，使用包外的 chunk256 声学模型与已准备参考。FP32 和低显存档都生成了 32 kHz、单声道、4.54 秒的 WAV；本轮未重新测量峰值显存或评价音质。模型、参考与输出均未进入发行清单。

其他 Windows 设备，尤其 Turing、Ampere、Ada、笔记本和低显存型号仍需测试。同一 NVIDIA 包不承诺覆盖 CPU、AMD、Intel GPU 或其他操作系统。干净 Windows 上无开发目录的安装、最低驱动、长句、多轮请求、取消恢复、完整显存测量和人工听音仍属于发行验收范围。
