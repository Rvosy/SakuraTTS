# Windows / Linux 便携性检查（2026-10-04）

本轮检查基于 `30ab4d5` 及当前工作树的 Mac 交付修复。Windows 和 Linux 均在独立目录安装测试环境，未覆盖原项目、系统 Python、模型或运行记录。代码尚未发布。

## 已修复的问题

| 问题 | 复现与修复 |
| --- | --- |
| Linux 无法使用锁文件安装 | 原 `uv.lock` 只接受 Windows 和 macOS，Linux 上 `uv sync --frozen` 直接拒绝平台。已补充 Linux 环境并重新锁定；Windows 已有锁定版本保持不变 |
| Windows 包可能混入 Mac 准备组件 | 装配器此前接受 Mac 的准备清单。现在核对主包与准备组件的目标平台，旧版无目标字段的准备清单继续按 Windows 处理 |
| Windows wheel 未核对平台与 ABI | 主环境和准备环境现在检查 wheel 标签，拒绝其他系统、架构或不匹配的 CPython ABI；仍保留既有准备环境的目录 egg 路径 |
| Windows 检查程序在 Unicode 路径下崩溃 | 在包含 `樱花`、空格和 `🧪` 的安装目录中，CPU 运算通过，但打印结果抛出 `UnicodeEncodeError: 'gbk' codec`。隔离启动忽略了 `PYTHONUTF8` 环境变量，三个批处理入口已改用显式 `-X utf8` |
| Mac 新增测试不能在 Windows 运行 | 修正模拟路径的分隔符断言；执行权限检查只在 POSIX 文件系统执行，归档测试比较原始文件实际权限。Windows 继续检查内容、目录布局和归档完整性 |

Windows 批处理回归实际启动私有解释器，覆盖三个入口的 Unicode 重定向输出。`check-runtime.bat` 的系统 `pause` 提示可能采用控制台本地编码，测试只核对 Python 输出，避免把系统提示误判为产品编码失败。

## 实测环境与产品回归

| 项目 | Windows | Linux |
| --- | --- | --- |
| 系统 | Windows 10 x64，10.0.19042 | x86_64，Linux 6.8.0-40，glibc 2.39 |
| 硬件条件 | Ryzen 7 5700G，32 GB 内存；GT 730 | 约 1.6 GiB 内存的虚拟机 |
| Python | 独立 CPython 3.12.13 | Python 3.12.3 的独立 venv |
| CPU provider | DirectML ORT 1.24.4 的 CPU provider | CPU ORT 1.30.0 |
| 依赖一致性 | `pip check` 通过，53 个安装包 | `pip check` 通过，49 个安装包 |
| 依赖与日文前端诊断 | 通过 | 通过 |
| 实际 CPU ONNX 矩阵运算 | 通过；运行检查未导入 Torch | 通过；运行检查未导入 Torch |
| 产品测试 | 389 项：388 通过、1 项 POSIX 测试跳过 | 389 项：385 通过、4 项 Windows 测试跳过 |

两个环境均未安装 PyTorch，因此未加载 `test_bert_features` 的 3 项测试。测试包含小型真实 ONNX 图和计算资源，也使用替身验证进程、配置与恢复；这些结果不是完整语音模型或音质验收。Mac 本地补充执行 39 项相关打包测试，38 通过、1 项 Windows 批处理测试跳过。

Windows 在线下载解释器返回网络错误 `WinError 10013`，随后从本地传入固定解释器和 wheel，核对 SHA256 后离线安装。Linux 从锁文件完成运行依赖安装；附加测试依赖下载重试后改用离线 wheel。下载失败日志保留，不计作产品测试失败或成功。

## Windows 运行包验收

使用 CPU/DirectML recipe、独立 CPython 3.12.13、VC143 CRT 和独立 FFmpeg 构建运行包。包中没有准备组件、发声模型或个人参考。

- 在中文、空格和特殊字符路径下，CPU 检查和日文依赖诊断正常退出。
- 归档后解压到另一处 Unicode 目录，4,191 个清单文件的 SHA256 全部一致，解压后的 CPU 检查通过。
- 将外部 `PYTHONHOME`、`PYTHONPATH` 指向不存在的目录，限制外部 `PATH` 后，包内解释器仍能启动 HTTP。`/health` 返回 `ready`、`model_loaded: false`；退出控制返回成功，进程以 0 退出。
- 以上验证覆盖解释器、依赖、归档、搬迁和无模型服务启动；没有执行原始权重转换、新参考编码或语音合成，也没有验证缺少系统 VC 运行库的干净机器。

归档大小为 189,349,229 字节，包含清单共 4,192 个文件，SHA256 为 `22201884b83a7752c3836c529bba6273eaf1fdc2425ace5ba61abfcac70d5918`。这是验收用运行包，不是已发布的完整整合包。

GT 730 的 DirectML 检查选择 DXGI `device_id=0`。FP16 探针未完全在 `DmlExecutionProvider` 执行，检查保留失败并返回非零退出码；该设备不能据此标为 DirectML FP16 通过。CPU 路径已通过上述检查，GPU 路径应另行验收。本轮没有复验当前 CUDA 包，也没有更新显卡驱动。

## Linux 交付缺口

Linux 的 CPU 运行依赖和产品回归已通过，但尚未完成完整语音模型、原始权重准备与新参考编码验收。构建器仍只提供 Windows 与 Apple silicon 的发行目标；Linux 没有 recipe、启动器或完整准备组件。

NVIDIA extra 和 CUDA 动态库装配仍面向 Windows。仅取消依赖中的平台条件，不能得到可搬迁的 Linux CUDA 包。Linux 后续需补齐解释器与动态库装配，确定 glibc 和 CPU 基线，再验证实际模型、断网首次使用与搬迁。Linux ARM、其他 glibc 版本、旧 CPU，以及更多 Windows GPU 均未验收。

当前用法见 [Linux 指南](../../docs/linux.md)和 [Windows 整合包指南](../../docs/portable-bundle.md)。

## 原始证据

本地证据目录为 `outputs/cross-platform-audit/`，保留源码快照、输入哈希、安装日志、失败结果和最终报告：

- `linux-uv-before.log`、`linux-uv-sync-after.log`：Linux 锁文件拒绝与修复后的解析结果。
- `windows-passed-windows-report-final.json`、`linux-linux-report-final.json`：两台实际主机的最终测试结果；对应日志包含逐项结果和跳过原因。
- `windows-bundle-report.json`、`bundle-cpu-check.log`：原始 GBK 编码失败；`windows-bundle-utf8-report.json` 记录修复后的通过结果。
- `windows-acceptance.json`：归档、解压后文件校验、CPU 检查、DirectML FP16 失败及 HTTP 启停。
- `source-before-files.json`、`source-final-files.json`、`source-linux-final-files.json`、`source-windows-fixed-files.json`：对应阶段的源码文件身份；测试断言修订与原始失败分别保留。
- `windows-tested-files.json`、`windows-source-verified.json`：最终 Windows 测试源码与本地 254 个文件的逐项 SHA256 核对结果。

Windows 验收目录中的 Git 提交只记录临时测试源码快照，未改动主仓库历史，未推送或发布。
