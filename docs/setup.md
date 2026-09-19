# 环境准备与项目结构

SakuraTTS 是独立推理项目。产品源码在 `src/sakuratts/`，通过根目录的 `pyproject.toml` 安装，不要求旁边存在 `SakuraTTS-References`。上游代码仅用于当前转换工具和开发对照，不是普通合成的运行依赖。

当前已有 Apple Silicon / MLX 和 Windows / NVIDIA 两条日文完整 WAV 路径。Windows 普通运行使用无 Torch 的环境；具体安装、模型转换、参考准备和命令见 [Windows 独立推理指南](setup-windows-nvidia.md)。本页保留开发环境和项目目录说明，依赖检查与模型、质量验收分别记录。

## Windows / NVIDIA 开发环境

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

`Bypass` 只作用于这次脚本进程，不修改系统执行策略。脚本在项目内创建 Python 3.11 的 `.venv`，安装 CUDA 12.8 版 PyTorch / torchaudio 2.7.1，再按 `requirements-windows-dev.txt` 安装日文前端与开发依赖，并将当前项目安装为可编辑包。已存在的非 3.11 环境会被拒绝，不会被删除。

脚本只使用本机 uv 缓存，所有 uv 命令带 `--offline`；缺少 Python 或依赖缓存时停止。它不修改系统 Python，不安装官方或 Lite 软件包，也不下载角色模型。PyTorch 用于模型转换、回归测试和 CUDA 开发，日常运行另用独立环境。Windows 的声学解码使用 ORT CUDA；使用预编译运行库不需要额外安装完整 CUDA Toolkit。

无需激活环境即可检查或运行测试：

```powershell
.venv\Scripts\python.exe -m sakuratts doctor --japanese --cuda
.venv\Scripts\python.exe -m unittest discover -s tests
```

也可以运行 `.venv\Scripts\Activate.ps1` 后使用 `sakuratts doctor --japanese --cuda`。这项开发检查包含依赖导入、日文 Nani ONNX 会话可用性和实际 CUDA 矩阵运算，不加载 TTS 模型。普通运行环境使用 `doctor --nvidia --config <runtime.json>`，检查依赖、资源哈希、参考身份和独立工作进程导入；它不执行 GPU 推理，也不评估音质。输出分别报告依赖与资源是否通过，未执行的推理和质量检查不会标记为通过。

无符号链接创建权限的 Windows 会跳过一项符号链接逃逸测试；普通相对路径、盘符和目录穿越检查继续执行。无需为安装环境更改系统权限。

`pyproject.toml` 声明直接依赖；Windows 清单固定本轮解析的传递依赖。更新直接依赖后，用下面的命令重新生成清单并复验：

```powershell
uv --offline pip compile pyproject.toml --extra japanese --extra dev --python-version 3.11 --python-platform windows --output-file requirements-windows-dev.txt
```

CUDA 包由安装脚本单独从 PyTorch 官方 cu128 索引安装，再安装其余依赖；清单中的 `torch==2.7.1` 接受对应的 `2.7.1+cu128` 构建。不要在 Windows 使用 `requirements-mlx-japanese.txt`。

### 本机验证

2026-09-20 在 Windows / RTX 5060 上运行安装脚本成功：Python 3.11.15、Torch / torchaudio 2.7.1+cu128，CUDA 矩阵运算通过，日文依赖与 Nani 会话可用；另外实际执行了 OpenJTalk 音素转换和 Sudachi 分词。`uv pip check` 未发现依赖冲突。

最初环境准备阶段运行 64 项单元测试，63 项通过，符号链接权限测试跳过；独立 wheel 在只有 NumPy 的新环境中完成基础检查。这些是当时的环境记录。后续 Sakura V2ProPlus 的真实转换、官方对照与 Windows 运行结果单列在 [Windows 实测记录](experiments/2026-09-20-windows-nvidia-backend.md)，不能用旧测试数量代替本轮验收。Mac 路径未在本机复验。

## Apple Silicon 日文运行环境

保留原先实测的完整依赖清单：

```sh
uv --offline venv --python 3.11 .venv
uv --offline pip install --python .venv/bin/python -r requirements-mlx-japanese.txt
uv --offline pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m sakuratts doctor --japanese
```

准备 GPT、SoVITS、日文前端和参考条件四个包后，按[日文运行入口](japanese-runtime.md)生成 WAV。资源目录由参数指定，不要求旧实验目录或时间戳命名。Mac 依赖的既有验证记录仍有效，本次 Windows 环境整理没有重新执行 Mac 合成。

## 代码和资源放在哪里

```text
SakuraTTS/
├── pyproject.toml                 # 自有 Python 包、命令与依赖
├── requirements-windows-dev.txt  # Windows 开发依赖清单
├── requirements-windows-runtime.txt # 无 Torch 的 Windows 日常环境
├── requirements-mlx-japanese.txt # 已验证的 Mac 日文依赖
├── src/sakuratts/                # 推理、文本、模型包与环境检查
├── scripts/                     # 安装、模型转换、参考准备、合成入口
├── models/                      # 本地权重与资源包，自行准备，不入 Git
├── outputs/                     # 生成音频与报告，不入 Git
├── tests/                       # 无需真实角色模型的回归测试
├── harness/                     # 上游对照、数值与性能实验
└── docs/                        # 使用说明、契约与实验记录
```

基础安装 `pip install -e .` 只安装 NumPy 和项目代码。`.[japanese]` 增加日文前端，`.[dev]` 增加转换与回归所需的 PyTorch 等依赖；Windows NVIDIA 开发建议使用上述脚本来选对 CUDA 构建。Mac 的 MLX 后端仍由既有 requirements 文件管理。

模型不随代码包上传。普通合成读取准备好的资源包；原始权重转换和参考准备需要明确指定的官方源码，仅开发阶段使用。Windows 已有独立准备工具，按 [Windows 指南](setup-windows-nvidia.md) 从原角色文件生成新包，不要求复制历史实验目录。

## 对照上游后的取舍

以下是最初安装整理时查看的固定版本；本轮 g50 文件身份和 Genie / Lite 定点核对见 [Windows 研究记录](research/windows-nvidia-reference-implementations.md)，不得把不同官方版本混入同一次数值对照：

| 项目 | 源码和安装方式 | 对 SakuraTTS 的用途 |
|---|---|---|
| [GPT-SoVITS `48b1a016`](https://github.com/RVC-Boss/GPT-SoVITS/tree/48b1a0169a28582a8984402f82cf438d3bfa6aca) | 仓库内维护 `GPT_SoVITS`、推理/API/WebUI 入口和 requirements；模型另行准备 | 借鉴项目内入口、明确资源位置的方式，保留行为对照 |
| [GSV-TTS-Lite `6c049397`](https://github.com/chinokikiss/GSV-TTS-Lite/tree/6c049397142f4c9147a85f86b6ba37546e93a188) | `pyproject.toml` 声明 `gsv-tts-lite` 包；CUDA Torch 单独安装；API/WebUI 使用各自依赖；文档说明首次下载公共模型 | 借鉴可安装的引擎包、独立的开发依赖和可选产品入口 |

SakuraTTS 保留已有 `src` 布局与独立 Python 包。Windows 完整准备流程已单列，公共模型仍由用户提供本地文件，不自动下载。WebUI、训练工具和全部上游依赖不进入日常安装。
