# Windows / NVIDIA 环境与模型准备

本页说明源码安装和离线资源准备。自带 Python 的版本见[整合包指南](portable-bundle.md)，设备与模型范围见[兼容矩阵](specs/compatibility-matrix.md)。

普通推理使用自有 CUDA GPT、日文前端和独立 ORT 声学工作进程。原始权重、官方源码及 PyTorch 用于转换和参考准备，运行主环境单独安装。

## 安装主环境

需要 Windows x64、NVIDIA 驱动、Python 3.11、uv 和 Windows C/C++ 运行库。Python 与 CUDA 头文件放在 ASCII 路径；模型和参考名称可含日文或中文。

以下命令在仓库根目录执行。`--offline` 读取本机缓存，缺少依赖时停止；需要联网安装时移除该选项。

```powershell
uv --offline venv .venv-windows-runtime --python 3.11
uv --offline pip install --python .venv-windows-runtime\Scripts\python.exe -r requirements/windows-runtime.txt
uv --offline pip install --python .venv-windows-runtime\Scripts\python.exe --no-deps -e .
uv --offline pip check --python .venv-windows-runtime\Scripts\python.exe
.venv-windows-runtime\Scripts\sakuratts.exe doctor --japanese
```

已创建的环境可以直接复用。直接依赖在 [pyproject.toml](../pyproject.toml)，冻结环境在 [windows-runtime.txt](../requirements/windows-runtime.txt)。主环境使用日文前端所需的 CPU ORT，声学 CUDA ORT 由独立解释器加载。

`doctor --japanese` 检查前端依赖；`doctor --nvidia --config PATH` 检查原 FP32 整图配置的依赖、资源完整性、参考身份和工作进程导入。实际 GPU 推理使用下方生成命令验证。`doctor --cuda` 检查的是带 PyTorch 的开发环境。

## 准备声学工作进程

[离线 ORT 组件工具](setup-ort-worker-offline.md)使用本地 CPython 3.9、NumPy、ORT CUDA 和匹配的 NVIDIA DLL，导出独立运行目录。完整输入要求和命令由该指南维护。输出中 `python.exe` 的路径用于模型或服务配置的 `acoustic_python`。

经典日文前端可以与声学 worker 共用这份解释器。`frontend_python` 可单独指定，省略时沿用 `acoustic_python`。这两个字段的路径规则见[模型格式](model-format.md)。

## 转换原始模型

在独立的开发环境安装转换依赖，并准备受支持的官方源码、HuBERT、Pro 说话人编码器、完整语言识别模型和日文字典：

```powershell
python -m pip install ".[convert,japanese]"
sakuratts convert `
  --gpt D:/Voices/character.ckpt `
  --sovits D:/Voices/character.pth `
  --reference D:/Voices/reference.wav `
  --reference-text "参考音声です。" `
  --official-source D:/Resources/GPT-SoVITS `
  --python D:/Resources/GPT-SoVITS/runtime/python.exe `
  --acoustic-python D:/Runtime/ort-worker/python.exe `
  --output models/mika
```

将路径和转写替换为实际资源。`--python` 选择准备阶段的解释器，`--acoustic-python` 选择普通推理的声学解释器。目标目录必须不存在；转换成功后生成 `model.json` 与资源包。只需模型与前端时，同时省略 `--reference` 和 `--reference-text`。

参考准备使用实际选定的前端实现、字典与精度。经典 `pyopenjtalk 0.3.4` 和 plus 前端在部分输入上会生成不同音素；CPU 与 CUDA 准备的声学条件也可能有浮点差异。固定数值对照应使用同一套准备结果，来源信息保存在包内 manifest。

服务也可按[配置示例](../examples/tts_infer.example.yaml)自动转换原始权重并准备新参考。源码安装需填写独立准备环境；完整整合包已绑定这些位置。用法见 [API V2 使用说明](api-v2-guide.md)。

## 生成 WAV

对带参考条件的模型目录执行：

```powershell
.venv-windows-runtime\Scripts\sakuratts.exe tts models/mika `
  --text "おはよう。今日もよろしくね。" `
  --seed 1234 `
  --output outputs/hello.wav
```

旧 `runtime.json` 可以直接作为模型参数，也可以用 `sakuratts convert --config OLD --output NEW` 整理为模型目录。使用其他参考时传 `--reference NAME`。

输出 WAV 和同名报告须使用新路径。退出码 `0` 表示正常停止，`2` 表示达到生成上限，报告状态为 `stopped_at_limit`；其他异常保留具体错误。NumPy 与官方 Torch 的相同 seed 使用不同随机数序列，数值对照使用固定随机输入。

默认 FP32。显存策略、FP16 包要求与调用示例集中在[推理档位](inference-profiles.md)，底层选项见 [Python API](python-api.md)。`Engine`、CLI 和 HTTP 默认启用声学空闲 arena 收缩；对照旧策略时可显式关闭，具体作用见 [Python API](python-api.md)。

### 独立声码器分块包

声学 FP16 和分块包需要对应的工程筛查，`doctor --nvidia` 目前只覆盖原 FP32 整图配置。高级转换与筛查流程见[声学 FP16 实验](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-acoustic-fp16.md#使用与证据)和[分块包入口](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-vocoder-public.md)。

## 验证记录

[Windows 后端实测](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-nvidia-backend.md)保存已运行的文本、参考切换、冷启动、热请求、显存和失败项。[运行环境清点](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/2026-09-20-windows-runtime-inventory.md)保存早期独立环境的逐文件体积；整合包后续去重和压缩结果见[整合包指南](portable-bundle.md#验证记录)。
