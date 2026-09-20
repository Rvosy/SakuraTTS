# ONNX Runtime 遥测退出崩溃

日期：2026-09-19。环境为 macOS 26.5.2 / arm64、Python 3.11.15、PyTorch 2.7.1、ONNX Runtime 1.30.0。ORT 的实际 build 信息为 `git-commit-id=f2c39fe`。

声学生命周期实验的两个控制进程都完成了八个 WAV、保存卸载边界并写入 `status=completed`，随后以 shell 退出码 134 中止。首个相关崩溃栈位于 ONNX Runtime 的遥测退出流程，没有指向 MPS 析构。音频和数值记录可以继续核验，但这两次进程不能记为正常完成。

## 原始现象与栈

首个运行是 `runs/20260919T111917.646673Z-official-mps/`，日志为 `runs/acoustic-lifetime-control-20260919T1119.log`。日志在 `COMPLETED=...` 后出现：

```text
libc++abi: terminating due to uncaught exception of type std::__1::system_error: recursive_mutex lock failed: Invalid argument
```

对应的 macOS 报告是 `~/Library/Logs/DiagnosticReports/python3.11-2026-09-19-191953.ips`，捕获时间为 `2026-09-19 19:19:49.1185 +0800`，PID 15225，异常为 `EXC_CRASH / SIGABRT`。只读取了这个相关 Python 报告，并提取主线程与触发异常的线程；没有收集其他应用日志。

主线程已进入 C++ 静态对象退出：

```text
exit
  → __cxa_finalize_ranges
  → onnxruntime::(anonymous namespace)::PosixEnv::~PosixEnv
  → onnxruntime::PosixTelemetry::~PosixTelemetry
  → onnxruntime::PosixTelemetry::Shutdown
  → Microsoft::Applications::Events::LogManagerImpl::FlushAndTeardown
  → Microsoft::Applications::Events::HttpClientManager::cancelAllRequests
```

触发 SIGABRT 的后台线程仍在处理 HTTP 回调：

```text
Microsoft::Applications::Events::PlatformAbstraction::WorkerThread::threadFunc
  → HttpClientManager::onHttpResponse
  → HttpResponseDecoder::handleDecode
  → HttpResponseDecoder::DispatchEvent
  → LogManagerImpl::DispatchEvent
  → DebugEventSource::DispatchEvent
  → std::__1::recursive_mutex::lock
  → std::__1::__throw_system_error
  → abort
```

这支持“遥测清理与后台回调发生竞争”的判断。具体是哪个对象先销毁、哪个锁失效，尚未用调试器复现。不能仅凭退出发生在 MPS 推理后就归因 MPS，也没有证据把日志中的 tokenizers fork 提示认定为这次中止的原因。

第二个控制运行 `runs/20260919T112123.173815Z-official-mps/` 也观察到退出码 134；其间的声学预计算候选退出码为 0。原始结果和日志均保留，工具观察到的退出码另存于 `runs/20260919T112420.880704Z-acoustic-lifetime-observation/result.json`，没有回写历史 JSON。

## 为什么采用导入前环境设置

官方中文前端通过 `GPT_SoVITS/text/g2pw/onnx_api.py` 导入 ORT 并创建 G2PW session。当前安装包公开了 `onnxruntime.disable_telemetry_events()`，但精确 build commit 的实现存在两个不同的关闭层次：

| 入口 | `f2c39fe` 的实际行为 | 对当前崩溃路径的影响 |
|---|---|---|
| `disable_telemetry_events()` | Python binding 取得 `Env::Default()`，`DisableTelemetryEvents()` 只执行 `enabled_.store(false)` | 停止后续事件，不销毁已经创建的上传器；退出仍可能执行 `FlushAndTeardown` |
| 导入 ORT 前设 `ORT_DISABLE_TELEMETRY=1` | `PosixTelemetry::Initialize()` 检查环境后直接返回，并锁存关闭状态 | 不创建 log manager 和上传器，避开本次栈中的清理对象 |

因此本轮选择进程局部、导入前的 `ORT_DISABLE_TELEMETRY=1`，并把实际设置写入测量元数据。它只控制 ORT 遥测，不改变模型权重、中文特征或推理精度。没有用 `os._exit`、额外 sleep、重试或吞掉异常来掩盖退出问题；也没有修改用户 shell 配置和原上游代码。该结论针对当前 POSIX ORT 1.30.0，不能假定旧版本、Windows ETW 或其他构建具有相同环境变量语义。

源码依据为以下固定提交文件，下载副本与 SHA-256 已保存在证据目录：

- [telemetry.cc](https://github.com/microsoft/onnxruntime/blob/f2c39fe/onnxruntime/core/platform/posix/telemetry.cc)：`Initialize()`、`Shutdown()` 和 `DisableTelemetryEvents()`。
- [telemetry_environment.h](https://github.com/microsoft/onnxruntime/blob/f2c39fe/onnxruntime/core/platform/telemetry_environment.h)：`ORT_DISABLE_TELEMETRY` 的解析与作用范围。
- [onnxruntime_pybind_state.cc](https://github.com/microsoft/onnxruntime/blob/f2c39fe/onnxruntime/python/onnxruntime_pybind_state.cc)：公开 Python API 的绑定。

## 纯 CPU 检查

使用安装包自带的 `mul_1.onnx`，固定 `CPUExecutionProvider`，intra/inter-op 线程数均为 1。分别运行“导入后只调用公开 API”和“导入前设置环境变量”两组，每组启动 3 个独立短进程；没有加载 TTS 模型或使用 MPS。

| 条件 | 正常退出 | 输出 |
|---|---:|---|
| API only | 3 / 3 | 与环境变量组三次逐位相同 |
| Environment before import | 3 / 3 | 与 API 组三次逐位相同 |

六次进程退出码均为 0，stderr 为空。所有输出 SHA-256 为 `ea172dd92c2034fe14450ee25238aa574d07e94d6b6b496e3f1a53c579354357`。检查时没有触发 ORT 的 CI 或 `ORT_RUNNING_UNIT_TESTS` 自动抑制条件。

短进程没有复现原竞争，因此这些结果只证明公开 API 可调用、环境设置下基本 CPU 计算和退出正常。它们不能证明 API-only 足以修复原问题，也不能代替完整 TTS 控制/候选两组的正常退出回归。后续测量需要在相同遥测设置下运行，并由外层进程记录实际退出码；子进程写出的 `completed` 不能单独作为进程成功凭据。

## 证据与复现

证据目录为 `runs/20260919T112548.123957Z-ort-teardown-cpu/`，相对于 `/Users/beyondpower/Documents/Projects/SakuraTTS-References`。其中保存了首个崩溃栈摘录、原报告 SHA-256、固定提交源码、CPU 子进程脚本、每次 stdout/stderr、父进程返回码、环境抑制标志与文件清单。未复制完整系统崩溃报告中的其他进程状态。

复现单个 CPU 条件时可运行证据中的脚本；它正常返回，不会强制结束解释器：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
PROBE="$REF/runs/20260919T112548.123957Z-ort-teardown-cpu/source/child.py"
"$REF/.venv-official-macos/bin/python" "$PROBE" api
"$REF/.venv-official-macos/bin/python" "$PROBE" environment
```

重新做成组实验时应复制脚本到新目录，保留本轮六次原始结果。

## 完整 TTS 回归补充

随后控制组 `runs/20260919T112504.624812Z-official-mps/`、声学预计算候选 `runs/20260919T112618.136695Z-official-mps/` 各完成 8 份 WAV，扩展候选 `runs/20260919T112741.072548Z-official-mps/` 完成 10 条输入。三次都在导入前设置相同的 opt-out，并由 `research/tools/run_reference_process.py` 的父进程保存实际退出码 0。两个 8 份 WAV 组逐字节相同，10 条候选与既有对照也逐字节相同。当前检查支持保留该进程局部设置，但有限次正常退出不等于证明所有退出竞争均已消除。
