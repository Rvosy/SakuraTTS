# 离线准备 Windows ORT 运行组件

当前机器已有的 ORT 1.19.2 CUDA 包使用 CPython 3.9 ABI，而 SakuraTTS 开发环境使用 Python 3.11。本地准备工具把必要文件导出到独立目录，供声学工作进程使用。它不下载文件，不修改源环境，也不复制 PyTorch 包。

这是一项离线依赖适配。双 Python 进程的启动、通信与 CPU 内存成本必须计入完整请求，不能只比较声学算子时间。它尚未通过另一台干净机器的安装验证，也不是通用的 Windows 发行包构建器。

## 准备

在项目根目录执行：

```powershell
.venv\Scripts\python.exe -B tools\prepare_ort_worker_runtime.py `
  --source-runtime D:\Project\sakura\tts\g50\runtime `
  --nvidia-root .venv\Lib\site-packages\nvidia `
  --output data\windows-ort-runtime
```

工具要求本地已存在以下版本与资源；缺失时直接报错：

- CPython 3.9 的解释器、标准库和本机扩展。
- NumPy 1.23.4 和 ONNX Runtime GPU 1.19.2。
- `nvidia-root` 中的 CUDA 12 cuBLAS、CUDA runtime、NVRTC 和 cuDNN 9 DLL，以及各自的本地许可文件。
- 源环境内携带的 `cufft64_11.dll`、`nvJitLink_120_0.dll` 和 `zlibwapi.dll`。这三个文件单独复制，并记录原始来源；不会因此复制或导入 Torch。

已有输出目录会被拒绝，工具不会覆盖或删除其中的文件。若准备失败，查看输出目录的日志，修正缺失资源后另选一个输出目录。

## 运行时边界

输出中的 `python39._pth` 只允许访问该目录内的标准库、NumPy 和 ORT。没有启用 `site` 自动发现，也不继承原环境的 `PYTHONPATH`。源目录路径只保留在 `runtime-manifest.json` 中作为来源信息，普通推理不会使用它们。

CUDA DLL 位于运行组件的 `cuda` 子目录。启动声学工作进程时需要同时：

1. 将这个子目录放在该进程 `PATH` 的前面。
2. 调用 `os.add_dll_directory()`，并在 Session 存活期间保留返回的句柄。

本机的 ORT 1.19.2 使用了旧式 DLL 查找行为，单独调用 `add_dll_directory` 不足以加载 CUDA provider。不要把 g50 或其 `torch/lib` 放进普通运行进程的路径。

若模型配置位于 `models/windows-sakura/runtime.json`，声学解释器路径为：

```json
{
  "acoustic_python": "../../data/windows-ort-runtime/python.exe"
}
```

工作进程业务代码由 SakuraTTS 提供，不在此运行组件中复制上游推理入口。系统仍需要合适的 NVIDIA 驱动与 Windows C/C++ 运行库。

## 已执行的验证

2026-09-20 在当前机器完成了本地导出。清单包含 504 个文件，合计 3,069,259,906 字节，约 2.86 GiB；主要体积来自 ORT CUDA provider 和 NVIDIA DLL。这是新增运行目录的逻辑文件大小，不包括仍保留的开发环境、模型和参考包。

导出后把子进程的 `PATH` 限制为新运行目录与 Windows System32，再实际启动新解释器。结果为 CPython 3.9.13、NumPy 1.23.4、ORT 1.19.2；两个包与全部 `sys.path` 均位于新目录中，`find_spec("torch")` 返回空。每个复制文件均保存源路径、字节数和 SHA-256，并核对复制前后源内容不变。

此工具只做导入验证，不创建 GPU Session。`get_available_providers()` 的列表来自 ORT 构建信息，不能证明某个 provider 已成功加载；列表中仍出现 TensorRT 也不表示工具交付了 TensorRT。实际声学与整链验证由对应 Harness 完成，不能用这次导入结果替代。

三个单元测试覆盖原目录内输出拒绝、已有输出不覆盖，以及保留原生库同时排除测试和字节码。实际运行日志与逐文件清单保存在 `data/windows-ort-runtime/`。

## 许可记录

输出保留 Python、NumPy、ORT 许可与第三方声明，以及本机 NVIDIA wheel 的许可文件。NVIDIA 库不适用 ORT 的 MIT 许可。从现有环境补齐的 cuFFT、nvJitLink 和 zlib 文件单独记录来源及源包声明；这次本地导出没有完成面向公众发行的再分发条款核对。
