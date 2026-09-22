# 语言、执行后端与发行组合拆分验收

日期：2026-09-22。此次只使用本机文件和构建缓存，没有联网或下载依赖。当前公开支持范围仍是 Windows / NVIDIA、V2ProPlus、日文；默认 direct 和 FP32 保持不变。

## 实现与本地原版对照

本机原版 `D:/Project/sakura/tts/g50` 根据平台安装不同 PyTorch 构建，语言模块按语言分派，但训练、WebUI 和多语言依赖仍集中安装。它不是一个可直接跨系统使用的 Windows 包。具体文件和后续 CPU、AMD、Apple、中文的缺口见[架构说明](../../docs/architecture.md)和 [ADR 0004](../../docs/adr/0004-composable-components.md)。

本轮分开了以下职责：

- Model 描述资源、语言和建议后端；实际执行支持由后端 factory 和前端装配判断。
- CUDA 专属参数归 CUDA adapter，公共 Engine 通过固定实现表选择后端。未实现的设备明确报错。
- 语言处理器负责音素与 BERT 特征，FrontendRuntime 管理资源和生命周期。日文的零 BERT 行为保持不变。
- 前端与声学有各自的解释器设置，旧配置仍可共用。服务中的显式设置覆盖缓存模型的旧安装路径，不改写模型文件；整合包最终使用发行清单的位置。
- recipe 选择平台、后端、语言、HTTP 服务和 worker 位置。HTTP 与离线准备组件可独立省略，当前只有已实现的 Windows CUDA / 日文目标。

检查转换、诊断、启动、参考、推理、权重切换和睡醒链路时，还修复了诊断错误地用声学 Python 检查前端的问题。共用解释器只探测一次，独立前端不会被要求导入 ORT/CUDA。删除了重复的格式、默认引用和已准备文本语言检查；保留资源身份、哈希、IPC 和进程所有权检查。

## 实际验证

完整单元回归通过 332 项，日志位于 `tmp/composition-final-tests.log`。wheel 和源码包使用 `uv build --offline --no-python-downloads` 构建，212 个源码快照文件已与构建时的工作区核对一致；wheel 含 77 个 Python 文件和 13 个许可文件。构建报告位于 `dist/composition/python/build-report.json`。提交前只清理了 `frontend/processors.py` 末尾的多余空行，发布物的运行逻辑未变。

真实整合包验收记录位于 `outputs/portable-composition-20260922/report.json`，从空模型与参考缓存开始：

- 原始 GPT / SoVITS 权重转换、新参考音频准备成功。
- managed 首次与重复请求、重启复用缓存、默认 direct 复用缓存均完成。
- 休眠后推理进程及子进程退出，三个服务实例正常结束。
- 四份 PCM 均为 110,720 个样本，最大差异不超过 1 个 int16 LSB；不是逐位相同。
- 与重构前保存的音频相比，长度一致，最大差异也是 1 LSB，见同目录 `previous-version-comparison.json`。

另将经典日文前端绑定到 `runtime/preparation/python.exe`，声学绑定到 `runtime/acoustic/python.exe`，通过了分别诊断和实际合成。两处诊断进程及主推理进程均未导入 Torch，生成音频与基线最大相差 1 LSB。记录位于 `outputs/portable-composition-workers-20260922/report.json`。这验证了独立解释器组合，不代表新增 CPU 推理后端。

新整合包仅替换已有解压目录中的产品文件、模板与清单，未再次复制整套运行环境。发行有效负载为 5,512,352,446 字节、18,575 个文件，比之前多 9,829 字节。压缩只包含清单文件，角色模型、参考音频、测试缓存和日志均不进入发行包。

最终归档 `dist/composition/release/SakuraTTS-Windows-NVIDIA.7z` 为 **1,960,852,931 字节（1.961 GB）**，低于 2,000,000,000 字节目标。SHA256 为 `2678eae3b5b60ecb4d1a0d729f1bce3155e1801be6c0f74ee84537a351282de9`。7z 完整性测试通过；另外提取并比对了内嵌 manifest，归档的 18,576 个成员与发行清单完全一致，见 `outputs/composition-archive-verification.json`。此次没有另解压一份完整运行目录，真实推理使用的是与归档清单一致的构建目录。

此次没有进行干净机器、跨设备、听感或显存峰值验收。进程退出证据也不等于重新测得控制进程显存为 0。当前测试记录主要用于确认装配边界变化后，原有默认链路和资源回收行为仍成立。

## 副本清理

用户授权后，先删除三份过时构建目录：`dist/first-use/preparation`、`dist/first-use/complete/SakuraTTS-Windows-NVIDIA`、`dist/first-use/SakuraTTS-Windows-NVIDIA`，合计回收逻辑文件大小 28,337,483,565 字节。

新版归档通过后，又删除 12 项旧运行副本、旧发行归档、压缩样本和 wheel 测试目录，共 19,447,065,708 字节。两批合计 **47,784,549,273 字节（约 47.8 GB）**，这是已删除文件的逻辑大小，不是 NTFS 实际分配簇的测量。五份构建 audit 均未引用这 12 项作为输入；三个旧运行副本没有用户模型或个人配置，当前保留包与 compact 的 18,466 个输入条目哈希一致。

原版目录、用户原始模型、当前构建输入及全部验收报告保留。盘点和删除结果分别位于 `outputs/composition-cleanup-audit.json`、`outputs/composition-cleanup-result.json`。历史报告引用的旧副本和旧压缩包已清理，当前交付物使用上面的 `dist/composition/release/` 路径。
