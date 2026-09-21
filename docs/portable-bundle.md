# Windows / NVIDIA 整合包

状态：交付目标与待办，尚未提供整合包。当前 `scripts/build_preview.py` 只构建 wheel、产品源码包和安装说明，不携带 Python、CUDA 运行库或模型。

目标是在支持的 Windows x64 / NVIDIA 设备上，下载、解压、选择模型后直接启动 HTTP 推理服务。用户不需要安装 Python、pip、uv、PyTorch 或 CUDA Toolkit；系统仍须有兼容的 NVIDIA 驱动。Windows 版本、最低驱动与显卡范围需由最终同一份 ZIP 的实测确定。

## 交付内容

整合包由固定版本的产品 wheel 和锁定的运行资源组装，与开发者源码包分别发布。产品源码继续留在 `src/sakuratts/`，构建与安装工作放在 `scripts/`；研究材料只用于开发验证，不作为整合包运行依赖。

建议目录如下，具体解释器布局在 ABI 验证后确定：

```text
SakuraTTS-Windows-NVIDIA/
  start-server.bat
  runtime/                 私有 Python、依赖和 CUDA 运行库
    runtime-manifest.json
  models/                  模型包
  configs/                 相对于整合包解析的配置
  licenses/                实际捆绑组件的许可与来源
  logs/
  cache/                   可删除并重建的缓存
```

权重转换仍在开发环境进行。运行环境与角色模型分开管理，可以提供基础运行包及可选模型包；只有已获再分发许可的模型和参考资源才能随整合包发布。准备新参考音频目前依赖额外准备环境与编码模型，必须在交付前明确：随包提供独立准备组件，或明确限定为已准备的参考。不能把主推理进程无 Torch 等同于任意新参考也无需准备依赖。

## 当前阻塞与实施顺序

1. **可搬迁路径。** `converter.package_model()` 会把 `acoustic_python`、`main_dictionary` 写成绝对路径；服务示例也引用本机准备环境。整合包需统一从自身位置解析运行资源，移动目录后不再读取开发机路径。调整模型与运行环境的绑定时，保留旧配置读取能力，并补模型、前端、参考准备的迁移验证。
2. **私有运行环境。** 根目录启动脚本当前选择本地 venv 或系统 Python。整合包启动器必须使用包内解释器，在自己的进程及 worker 中确定 DLL、头文件和缓存位置。当前 `configure_cuda()` 会保留外部 `CUDA_PATH`；整合包需要隔离系统 CUDA 和用户 Python 环境，不能只增加几个 PATH 项就算完成。
3. **运行组件来源。** `prepare_ort_worker_runtime.py` 是本机导出工具，仍从 `torch/lib` 补 DLL，并查找固定的 Torch 2.7.0 许可目录。正式构建要从固定来源的组件组装，记录版本、下载来源、SHA256、许可和目标路径，检查 VC Runtime 等系统依赖。不能把任意开发 venv 直接压缩发布。
4. **路径与 ABI。** 当前 CuPy/NVRTC 头文件要求 ASCII 路径，需要验证带空格、中文用户名及中文解压目录的启动方案。`LOCALAPPDATA` 本身也可能含中文，不能作为未经检查的 ASCII 保证。经典 `pyopenjtalk 0.3.4` 仍依赖现有 Python 3.9 组件；是否统一到 Python 3.11，要同时验证前端扩展、字典和 ORT，不能仅凭存在 CP311 ORT wheel 决定。
5. **实际 GPU 检查。** `doctor --nvidia` 当前检查导入和资源身份，不执行 GPU 推理。整合包应另有实际分配、NVRTC 编译、cuBLAS、所选精度、CUDA Graph 与声学 CUDA 请求检查，并输出 GPU、驱动、组件版本及失败阶段。检查成功与音质验收分别记录。

这些项目需要运行代码与构建工具的后续实现。本轮仓库整理不改变现有模型字段、解释器版本或 CUDA 搜索策略。

## 兼容验收

先以 Windows / NVIDIA 为范围，分别测试 Turing、Ampere、Ada 和 Blackwell 的代表设备。GTX 16、RTX 20/30/40/50 属于拟验证范围；当前已有证据主要来自 RTX 5060，不能据此宣布整代或所有型号通过。再补低显存及笔记本样本，记录实际型号和驱动。CPU、AMD、Intel GPU 和其他操作系统需要独立后端与发行验收，不承诺由同一个 NVIDIA 包直接覆盖。

每台设备使用相同 ZIP，保留以下记录：

- 干净 Windows 上无 Python、CUDA Toolkit 和开发工程时的首次启动、短句及长句请求。
- FP32 与所选 FP16 档位、多轮请求、取消、失败恢复、卸载重载和显存峰值。
- 整个目录搬迁、空格与非 ASCII 路径，以及已有其他 CUDA / Python 环境时的隔离结果。
- 音频内容、停止原因、时长、有限值检查、数值误差和人工抽听；跨 GPU 不以 PCM 哈希完全相同作为唯一门槛。

运行包大小、首次启动后缓存大小和准备新参考的依赖要分别记录。Linux 云卡测试可补充算子与架构证据，不能替代 Windows 整合包验收。
