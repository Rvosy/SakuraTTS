# Linux CPU 运行环境

Linux 当前使用源码安装路径。依赖锁文件包含 Linux，CPU 后端使用 NumPy 和 ONNX Runtime；DirectML 只适用于 Windows，现有 NVIDIA 安装、CUDA 动态库装配和整合包构建也以 Windows 为目标。Linux 尚无整合包 recipe、启动器和完整准备组件，不能直接运行 Windows 压缩包。

## 安装与诊断

以下命令在仓库根目录执行，使用独立的 Python 3.12 环境。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，并让 `ffmpeg` 可从 `PATH` 找到。

```sh
UV_PROJECT_ENVIRONMENT=.venv-linux-cpu uv sync --locked --python 3.12 \
  --extra cpu --extra japanese-text --extra server
uv pip check --python .venv-linux-cpu/bin/python
.venv-linux-cpu/bin/python -m sakuratts doctor --backend cpu --japanese
```

诊断检查依赖与前端资源，不代表已经完成模型合成或音质验收。不要添加 `directml` 或 `nvidia` extra 来启用 Linux GPU；当前配置没有提供这条交付路径。

## 模型与服务

已有匹配 CPU 后端的模型和预制参考包时，使用公共入口：

```sh
.venv-linux-cpu/bin/python -m sakuratts doctor models/voice-cpu --backend cpu
.venv-linux-cpu/bin/python -m sakuratts tts models/voice-cpu --backend cpu \
  --text "こんにちは。" --output outputs/linux.wav
.venv-linux-cpu/bin/python -m sakuratts serve models/voice-cpu --backend cpu
```

CPU 模型需要 INT8 GPT sidecar 与 FP32 声学资源，配置见 [CPU 指南](cpu-amd.md)。经典日文前端的 `frontend_python` 必须指向具有匹配 classic 前端和字典的 Linux 环境，不能沿用 Windows 的 `.exe` 路径，也不能直接换成上述 plus 前端。模型和资源使用可搬迁的相对目录，见[模型格式](model-format.md)。原始权重转换和新参考编码仍需独立的准备环境、官方源码及辅助模型，仅安装上述运行依赖不包含这些资源。

当前实测条件、失败记录及未验证项见 [Windows / Linux 检查记录](../research/notes/cross-platform-portability-20261004.md)。其他 glibc 版本、旧 CPU 和 Linux ARM 尚未验收。
