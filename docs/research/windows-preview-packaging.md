# Windows 预览版的源码与 Python 包分发

日期：2026-09-20。首版面向开发者：GitHub 保存源码，由维护者在本地构建 Python 安装包和源码包，压缩后手动上传 ModelScope。运行环境、模型和参考资源单独准备。本轮不设置自动发布，也不把整合环境当作默认交付物。

Genie 和 Lite 都有可借鉴的 Python 库封装。最适合 SakuraTTS 的部分是稳定的库入口、可安装的 wheel、独立模型目录，以及将界面和转换工具放在普通推理依赖之外。上游的下载交互、宽松依赖版本和整合包体积不直接沿用。

## 查阅范围

通过 GitHub API 固定了两个默认分支的当前提交，并读取了打包配置、公开入口、资源准备、下载代码和许可；另从 PyPI 下载对应版本的 wheel / sdist，核对 SHA256 后只枚举内容，未安装或执行上游代码。

| 项目 | 源码提交 | 源码声明版本 | 本轮核对的 PyPI 版本 |
| --- | --- | --- | --- |
| [Genie-TTS](https://github.com/High-Logic/Genie-TTS/tree/d347fd0f8683e9a362b69f59fa0a4799ddb5e828) | `d347fd0f8683e9a362b69f59fa0a4799ddb5e828` | `2.0.2` | `2.0.2` |
| [GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite/tree/6c049397142f4c9147a85f86b6ba37546e93a188) | `6c049397142f4c9147a85f86b6ba37546e93a188` | `0.4.7` | `0.4.7` |

同一个版本号不表示 Git HEAD 与 PyPI 产物相同。Genie 当前源码要求 Python `>=3.10`，提供 `gui` extra；已发布的 `2.0.2` wheel 元数据写的是 `>=3.9`，没有这个 extra，多份源码也不同。Lite 的 `0.4.7` wheel 与当前源码的 `Config.py` 不同。以下将源码和已发布产物分开描述，不用其中一份代替另一份。

本机 `D:/Project/Genie-TTS` 停在 `0ad8b386`，且有本地修改；其中 `build_sakura_bundle.ps1`、`install_deps.ps1`、`start_test.ps1` 等是未跟踪文件。它们不是这两个上游固定提交的打包脚本，未作为上游证据。

## 两个项目实际怎样封装

| 项目 | Python 包 | 使用入口 | 资源与模型 | 附加功能 |
| --- | --- | --- | --- | --- |
| Genie | setuptools，`src/genie_tts`，包内附带 ONNX 图模板和权重名称表 | `import genie_tts` 后调用加载角色、设置参考、合成、卸载等函数 | `GenieData` 和 `CharacterModels` 在包外；支持指定资源目录 | 当前源码将 PySide6 放在 `gui` extra；FastAPI / uvicorn 是基础依赖，转换时才导入 Torch |
| Lite | setuptools，根目录下的 `gsv_tts` | `from gsv_tts import TTS`；实例管理模型并执行合成 | 默认缓存为 `~/.cache/gsv`，也接受 `models_dir` | API 和 WebUI 在仓库独立目录，各有 requirements；Torch / torchaudio 需要使用者按平台另装 |

两份 `pyproject.toml` 都没有 `[project.scripts]`，本轮检查的 wheel 也没有 `entry_points.txt`。它们主要按 Python API 分发；Genie 的仓库 `Main.py` 是 GUI 启动脚本。SakuraTTS 已有 `sakuratts` 命令行入口，首版应保留这一入口和 Python API，无需为了与上游一致而移除 CLI。

Genie 的 [pyproject.toml](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/pyproject.toml) 明确配置包内 ONNX 模板；[公开入口](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/__init__.py)导出库函数。[转换入口](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/Internal.py#L326)在调用时检查 Torch。[当前 ModelManager](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/ModelManager.py#L189)设置 `CPUExecutionProvider`，依赖也是 CPU `onnxruntime==1.22.1`。因此不能用这份默认安装配置推算 SakuraTTS CUDA / cuDNN 运行环境大小。

Lite 的 [pyproject.toml](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/pyproject.toml)没有声明 Torch，但 [TTS.py](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/gsv_tts/TTS.py#L12)会导入 Torch 和 torchaudio；[README](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/README.md#L93)要求先按 CUDA / CPU / MPS 安装。这是显式的环境准备步骤，小 wheel 本身并不包含推理所需的全部依赖。

两个上游大部分基础依赖没有锁定版本。SakuraTTS 已有固定的 Windows 运行依赖，应继续保留，并将模型转换、ASR、实验 Harness 的依赖留在开发环境。把 API 服务、GUI 或测量工具加进默认安装，只会扩大首版必须维护的范围。

## 模型下载与运行库分离

Genie 的 [Resources.py](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/Core/Resources.py)用 `GENIE_DATA_DIR` 指定外部资源目录。资源目录不存在时，导入链会询问是否从 Hugging Face 下载；完整下载函数在当前源码中还会追加 Chinese RoBERTa。角色资源由 [PredefinedCharacter.py](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/src/genie_tts/PredefinedCharacter.py)按角色路径下载到 `CharacterModels`。两处 `snapshot_download` 都没有固定 revision。

Lite 在创建 TTS 实例时检查外部资源。其 [Download.py](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/gsv_tts/Download.py)通过连通性和延迟选择 ModelScope 或 Hugging Face：前者下载 `pretrained_models5.zip`，后者下载 `pretrained_models6.zip` 并另取 GitHub 的 `g2p.zip`。下载地址使用 `master` / `main`；代码检查下载字节数，没有绑定这些压缩包的 SHA256。

SakuraTTS 可以采用同样的代码与资源分开存放方式，保留现有包 manifest、相对配置路径和哈希校验。首版由使用者显式准备资源，普通导入和合成不应自动选镜像、下载模型或发起交互。ModelScope 上传内容应以文件名、版本和 SHA256 标明实际产物，不能只给一个会变化的分支地址。

本轮检查的两份完整 Git tree 都没有 Windows 嵌入式 Python 准备脚本、`.bat` / `.cmd` / `.ps1` 打包脚本或 `.github/workflows`。这只说明所查源码没有提供这些内容。Genie 有整合包，不代表其完整生成过程已随当前源码公开；本轮未下载或解包大型 GUI 整合包，不能确认其中的解释器布局、DLL 去重、安装器或裁剪方法。

## 体积需要分别报告

PyPI 的精确版本元数据和本轮实际读取的文件一致：

| 已发布文件 | 压缩字节 | 解包内容字节 | 文件数 |
| --- | ---: | ---: | ---: |
| `genie_tts-2.0.2-py3-none-any.whl` | 489,887 | 4,574,195 | 73 |
| `genie_tts-2.0.2.tar.gz` | 444,102 | 4,587,230 | 81 |
| `gsv_tts_lite-0.4.7-py3-none-any.whl` | 109,537 | 399,665 | 54 |
| `gsv_tts_lite-0.4.7.tar.gz` | 97,031 | 427,647 | 59 |

来源为 [Genie PyPI 2.0.2](https://pypi.org/pypi/genie-tts/2.0.2/json) 和 [Lite PyPI 0.4.7](https://pypi.org/pypi/gsv-tts-lite/0.4.7/json)。解包内容字节是归档内普通文件长度之和，不含文件系统分配开销。这些文件都没有捆绑完整 Python、依赖或外部模型。

GitHub Releases 的资产是另一种交付物：

| Release | 资产 | API 报告的下载字节 |
| --- | --- | ---: |
| [Genie v2.0.2](https://github.com/High-Logic/Genie-TTS/releases/tag/v2.0.2) | `Genie-TTS.GUI.7z` | 1,931,591,446 |
| [Genie v1.0.2](https://github.com/High-Logic/Genie-TTS/releases/tag/v1.0.2) | `Genie.High_Logic.7z` | 594,776,693 |
| [Lite g2p](https://github.com/chinokikiss/GSV-TTS-Lite/releases/tag/g2p) | `g2p.zip` | 13,004,922 |

Genie v2.0.2 的说明称整合包适合 Windows 10/11、无需 GPU、预装三组角色模型，模型一项标为 963 MB。README 则将约 200 MB 运行时、约 230 MB 角色模型和首次约 391 MB 资源分别描述，未给出可重建的 200 MB 文件清单。当前源码的资源下载还增加了可选 RoBERTa，所以旧 README 的资源数字也不能覆盖所有当前下载路径。

因此首版应分别报告 Python 包大小、资源包大小和用户准备好的环境大小。SakuraTTS 的 wheel 小，不会自动把现有 Windows 双解释器环境变成几百 MB；此前的 [4.314 GiB 运行环境清点](../../research/experiments/2026-09-20-windows-runtime-inventory.md)继续有效，不能从安装成本中删去未随 wheel 分发的依赖。

## SakuraTTS 首版应交付什么

建议保持现有 `src/sakuratts` 布局，直接完善 setuptools 分发，不为首版引入冻结可执行文件、安装器或一套新服务框架。GitHub 中保留产品源码、测试、转换脚本、Harness 和文档；构建目录、模型、第三方运行库及实验输出仍留在 Git 之外。

维护者本地生成的上传材料至少应包含：

- 可安装的 `sakuratts-<version>-py3-none-any.whl`，含自身许可、适用的第三方许可和普通运行所需模块。
- 对应版本的 sdist，或附带所需脚本、依赖清单和文档的源码归档。应实际检查 sdist 内容，不能默认 setuptools 会收进仓库中所有文件。
- 简短安装说明、一个可改路径的配置示例、已知限制，以及产物文件名 / 字节数 / SHA256 / 源码 commit 的清单。

维护者将这些产物压缩后手动上传 ModelScope。模型、前端和参考包若另外上传，应使用独立清单并说明来源与许可；不能仅因为 Python 源码使用 MIT 就推断角色权重和参考音频也可同样分发。首版不需要 PyPI 发布、GitHub release asset 上传或发布 CI。

用户从下载的 wheel 安装后，应能离开源码目录运行 `python -m sakuratts --help` 和 `sakuratts --help`；Python API 导入也应使用安装产物。模型转换与打包脚本可以从源码归档调用，不必为了将它们塞进运行 wheel 而改写全部目录结构。

版本、源码 commit 和产物哈希必须同时记录。上游相同版本号对应不同源码的情况说明，只写 `0.1.x` 不能准确定位使用者实际安装的实现。维护者本地重新构建时也应使用新的输出目录，避免把旧产物混入上传压缩包。

## 发布范围和剩余验证

SakuraTTS 自身源码的 MIT 授权已由作者明确，本轮发布整理补入根 `LICENSE` 和包元数据；原有第三方许可继续保留。Genie 和 Lite 的所查源码也各自使用 MIT，见 [Genie LICENSE](https://github.com/High-Logic/Genie-TTS/blob/d347fd0f8683e9a362b69f59fa0a4799ddb5e828/LICENSE) 和 [Lite LICENSE](https://github.com/chinokikiss/GSV-TTS-Lite/blob/6c049397142f4c9147a85f86b6ba37546e93a188/LICENSE)。本轮只研究它们的分发方式，没有复制其产品代码。

实际上传前，需从新输出构建 wheel / sdist，检查文件清单，再用新环境安装非 editable wheel，离开 checkout 验证 CLI 和 Python API。sdist 应能重建 wheel；不能把源码目录中的成功导入当成安装包测试通过。

Windows 声学工作进程目前依赖单独准备的 CPython 3.9 / ORT CUDA 1.19.2 组件，已有准备命令使用维护者本机的运行文件。开发者版说明必须列出该前置条件，不能声称 `pip install` 后即可生成音频。若计划随后提供运行组件包，其解释器、DLL、词典许可及搬迁验证需单独完成，不属于本次小 wheel 的验证结果。

预览版的实测范围仍是 Windows / NVIDIA、日文、已验证的 Sakura V2ProPlus 组合、完整 WAV 和单活动请求。速度与低显存配置可以作为显式试用选项；声学分块包的 `quality_accepted=false` 不因打包而变成 `true`。人工试听、其他机器部署、其他权重和语言兼容性应继续按各自记录说明。本轮研究没有运行 GPU、复测音频或创建外部发布。

## 原始证据

本地 `outputs/windows-preview-research/manifest.json` 记录检索时间、源码提交、31 份源码的 URL / 字节数 / SHA256、PyPI 产物哈希和证据文件清单。每个项目目录包含 GitHub repo / commit / 完整 tree / releases API 响应、固定源码、PyPI 元数据、四个小型分发归档中的对应两份，以及逐文件归档清单。GitHub Release API 本次返回 Genie 两项、Lite 一项，均少于每页 30 项；两份 Git tree 都没有被截断。

这些原始材料位于忽略目录，不随源码包分发。本文中的 commit 链接和精确版本 PyPI URL 可用于重新查询。大型 Windows GUI 资产只核对了 API 元数据，未下载；其资产大小不能作为本地安装验收结果。
