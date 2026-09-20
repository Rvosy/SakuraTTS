# Windows 开发者预览版的安装验证

日期：2026-09-20。用户确认当前性能足够进入预览阶段，本轮转向源码整理和独立安装，暂停新增内核优化。GitHub 只推送源码；压缩包后续由维护者手动上传 ModelScope。本轮没有上传二进制、模型或运行环境。

## 源码与包的改动

版本设为 `0.1.0a1`，按用户选择补充 MIT 许可。README、安装指南、配置示例和发布说明区分运行代码、开发工具及外部资源。Windows 开发依赖清单已补齐当前转换工具所需依赖。运行 wheel 不包含 Torch、实验 harness、模型或 DLL；sdist 保留转换脚本、测试、文档和实验数据说明。

`scripts/build_preview.py` 从明确的源码目录构建 wheel / sdist，再生成 ZIP、源码哈希与产物校验清单。它拒绝覆盖已有输出，不上传文件。归档生成前，逐文件检查 sdist 是否包含完整选中源码，以及 wheel 的产品代码和许可证是否与源码快照一致。首次候选漏掉两份 `harness/cases/*.json`，已修复 MANIFEST；该失败记录仍保留。

独立 CPython 3.9 工作进程原先会把主包的父目录加入 `sys.path`。wheel 安装后，这个目录是主 Python 3.11 的整个 `site-packages`，可能错误加载不同 ABI 的 NumPy。现在通过标准库按目录加载指定的 SakuraTTS 包，保留 worker 自己的 NumPy / ORT 搜索路径；前端、声学和诊断入口共用这项修复。

含日文字符的独立环境完成导入与 worker 检查后，首次 CUDA 编译失败，NVRTC 报告无法读取 `cupy/complex.cuh`。因此增加实际头文件路径的提前检查；非 ASCII 安装路径给出重建环境提示，不改变模型路径和日文文本的处理。后续 ASCII 且含空格的环境已完成真实推理。

## 实际验证

完整测试在开发环境执行：

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

共 293 项，291 项通过，2 项因 Windows 符号链接权限跳过。开发环境依赖检查通过。曾误用无 Torch / ONNX 导出器的运行环境执行完整开发测试，6 个测试模块因缺开发依赖导入失败；该日志保留，没有把缺依赖视为产品通过。

第二版构建候选的 ZIP 为 1,640,117 字节，其中 wheel 为 174,113 字节、sdist 为 1,451,118 字节。这只是代码与文档的压缩体积，不包括 Python、CUDA、cuDNN、词典或模型。368 个选中源码文件全部进入 sdist 且哈希一致。wheel 的 64 个成员包括 48 份产品 Python 源码、11 份许可证及 5 份元数据；未包含模型或运行库二进制。

源码包解压到无 `.git` 的独立目录后，使用离线 `uv build --wheel` 成功重建。两个 wheel 的 64 个成员内容逐字节相同，归档文件哈希因 ZIP 元数据不同而变化；本轮不宣称压缩文件完全可重现。

在仓库外两个新建环境中，以非 editable 方式安装实际 wheel。ASCII 且含空格的环境通过 `--version`、依赖检查及原 FP32 配置的 `doctor --nvidia`。含日文路径的环境在新增检查中明确失败，诊断未声称已经执行推理。worker 的 NumPy / ORT 来自自身 CPython 3.9 组件，SakuraTTS 业务代码来自已安装 wheel。

第二版候选在新的 CuPy 编译缓存下完成 18 次 Python API 请求：默认 FP32、低显存分块、速度分块各 6 次，包括四类文本、重复短句和卸载后长句。主进程拒绝读取仓库 `src`、`scripts`、`harness` 和旧运行环境，并拒绝网络连接；实际 worker 命令也指向已安装包。模型和独立 worker 组件仍复用本机已准备的文件，这不是另一台干净机器的部署验收，也不是新增性能基准。

两种分块配置的 12 条 PCM 与既有同条件输出逐位一致，音素、完整采样 token、返回语义 token、停止原因及长度也一致。原 FP32 整图存在小量重复输出波动：首轮短句 110,720 个采样中有 19 个相差 1 个 PCM 量化单位，逐位断言失败；用原 `atol=1e-4, rtol=1e-5` 检查归一化 PCM 时，超差数为 0。历史源码路径也有同量级差异，四类首次请求与历史同条件输出的完整契约相同。本轮保留逐位失败，未将它解释成音质验收。

## 证据与边界

本地原始记录在 `outputs/windows-preview-build/` 和仓库外安装验证目录；位置记录为 `outputs/windows-preview-validation-location.json`、`outputs/windows-preview-validation-ascii-location.json`。独立归档审计为 `candidate-v1-audit.json`、`candidate-v2-audit.json`；FP32 重复差异审计为 `outputs/windows-preview-validation/baseline-repeat-audit.json`。完整测试日志为 `outputs/windows-preview-final-unit-tests.txt`。这些本地运行产物不进入 GitHub。

当前可先推送已验证源码，打包产物仍为本地候选。人工听音、其他机器 / GPU、原始资源的通用安装器及外部发布尚未完成。辅助 ASR 和历史精度失败继续按[音频检查记录](2026-09-20-windows-asr.md)保留。
