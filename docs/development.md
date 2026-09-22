# 开发指南

先读与改动相关的 [Spec](specs/)、[ADR](adr/) 和测试。依赖声明在 [pyproject.toml](../pyproject.toml)；`requirements/` 是此前实测环境的冻结快照，不能当作额外功能声明。运行、转换与私有声学解释器仍分开安装。

```powershell
python -m pip install -e ".[japanese,english,dev,server]"
python -m unittest discover -s tests
python research/run_tests.py  # 仅 Git checkout，验证实验工具
python benchmarks/run.py models/mika --text "こんにちは。" --output outputs/benchmark.json
python scripts/build_preview.py --output dist/preview-0.1.0a1
```

单元测试使用小型资源和替身验证代码行为。CUDA 冒烟需要模型与目标运行环境，MLX 验证需要 Mac；音质通过实际音频另行检查。

## 环境准备

### Windows / NVIDIA 开发环境

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

`Bypass` 只作用于这次脚本进程，不修改系统执行策略。脚本在项目内创建 Python 3.11 的 `.venv`，安装 CUDA 12.8 版 PyTorch / torchaudio 2.7.1，再按 `requirements/windows-dev.txt` 安装日文前端与开发依赖，并将当前项目安装为可编辑包。已存在的非 3.11 环境会被拒绝，不会被删除。

脚本只使用本机 uv 缓存，所有 uv 命令带 `--offline`；缺少 Python 或依赖缓存时停止。它不修改系统 Python，不安装官方或 Lite 软件包，也不下载角色模型。PyTorch 用于模型转换、回归测试和 CUDA 开发，日常运行另用独立环境。Windows 的声学解码使用 ORT CUDA；使用预编译运行库不需要额外安装完整 CUDA Toolkit。

无需激活环境即可检查或运行测试：

```powershell
.venv\Scripts\python.exe -m sakuratts doctor --japanese --cuda
.venv\Scripts\python.exe -m unittest discover -s tests
```

也可以运行 `.venv\Scripts\Activate.ps1` 后使用 `sakuratts doctor --japanese --cuda`。这项开发检查包含依赖导入、日文 Nani ONNX 会话可用性和实际 CUDA 矩阵运算，不加载 TTS 模型。普通运行环境使用 `doctor --nvidia --config <runtime.json>`，检查依赖、资源哈希、参考身份和独立工作进程导入；它不执行 GPU 推理，也不评估音质。输出分别报告依赖与资源是否通过，未执行的推理和质量检查不会标记为通过。

无符号链接创建权限的 Windows 会跳过一项符号链接逃逸测试；普通相对路径、盘符和目录穿越检查继续执行。无需为安装环境更改系统权限。

`pyproject.toml` 声明直接依赖；Windows 清单固定已解析的传递依赖。更新直接依赖后，用下面的命令重新生成清单并复验：

```powershell
uv --offline pip compile pyproject.toml --extra japanese --extra dev --python-version 3.11 --python-platform windows --output-file requirements/windows-dev.txt
```

CUDA 包由安装脚本单独从 PyTorch 官方 cu128 索引安装，再安装其余依赖；清单中的 `torch==2.7.1` 接受对应的 `2.7.1+cu128` 构建。不要在 Windows 使用 `requirements/mlx-japanese.txt`。

### Apple Silicon 日文实验环境

保留原先实测的完整依赖清单：

```sh
uv --offline venv --python 3.11 .venv
uv --offline pip install --python .venv/bin/python -r requirements/mlx-japanese.txt
uv --offline pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m sakuratts doctor --japanese
```

准备 GPT、SoVITS、日文前端和参考条件四个包后，按[日文运行入口](japanese-runtime.md)生成 WAV。资源目录由参数指定；历史模型和运行结果见该指南链接的研究记录。


## 旧目录对应关系

| 原位置 | 当前位置 |
| --- | --- |
| `nvidia.py` 的实现 | `backends/cuda/engine.py`；原导入保留兼容入口 |
| `cuda_*` / `ort_*` / `mlx_*` | 对应 `backends/cuda/`、`onnx/`、`mlx/` |
| 文本、日文、中文、G2PW 模块 | `frontend/` |
| `tone_sandhi` / `zh_normalization` | `frontend/_vendor/`，保留许可与来源 |
| worker、参考和采样模块 | `_internal/` |
| `harness/` | `research/tools/`，常用入口提到 `benchmarks/` |
| `docs/experiments/` | `research/experiments/` |
| GPT / ONNX 导出和 Windows 参考准备 | `_internal/conversion/`，由 `sakuratts convert` 编排 |
| 其余模型与资源脚本 | `tools/` |
| 根目录 `requirements-*.txt` | `requirements/*.txt` |

旧 `sakuratts synthesize --config ...` 和 `sakuratts.nvidia.NVIDIAEngine` 继续可用。内部模块没有逐个添加别名；仓库内调用方与测试已迁移，外部研究脚本需按表更新。

新实验输出写到被 Git 忽略的 `outputs/`、`results/` 或 `artifacts/`。`research/experiments/data/` 保留已引用的历史证据，避免丢失数值失败与性能数字的来源。

## 产品与研究的边界

`tests/` 验证包、API、转换和运行生命周期；`research/tests/` 验证历史测量与实验工具。两组测试都保留，发布源码包只带产品测试。研究测试入口会加载少量产品测试夹具，需要完整 Git checkout。

研究总结、原始测量和失败记录由 [research/README.md](https://github.com/Rvosy/SakuraTTS/blob/main/research/README.md) 导航。产品支持范围由[兼容矩阵](specs/compatibility-matrix.md)维护。

`python scripts/build_preview.py --offline --output dist/preview-check` 构建 wheel、源码包和校验清单。源码包包含维护工具、产品测试、使用文档与许可，排除整个 `research/`。源码包应能在没有研究目录的环境中重建并运行产品测试；研究脚本从对应 Git 提交获取。
