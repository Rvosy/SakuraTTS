# 后台驻留与提前唤醒

长期待机、间歇生成语音的应用可以选择 `managed` 模式。空闲时退出推理进程树，HTTP 控制服务继续接收请求；调用方也可以提前唤醒，让加载与其他工作并行。

默认 `direct` 模式在 HTTP 进程的专用线程中持有模型，适合持续请求。两种模式共用推理实现、参考准备和音频格式。模式选择与显存档位相互独立。

## 选择运行方式

```powershell
.\start-server.bat -c configs/tts_infer.yaml --runtime-mode managed --idle-sleep-seconds 60
```

启动参数、接口和状态字段集中在 [HTTP API](http-api.md#可选后台控制模式)。无须先调用唤醒接口：休眠时直接提交 `/tts` 会自动启动推理进程；提前唤醒只是减少文本就绪后的等待。

| 使用场景 | 选择 |
| --- | --- |
| 持续生成，优先减少每次加载 | `direct`，保留所选档位允许驻留的资源 |
| 间歇生成，空闲时释放进程级资源 | `managed`，根据请求间隔设置空闲时间 |
| 显存紧张，可接受片段重载 | `managed` 配合 H 极限档及匹配的 FP16 chunk256 模型包 |

H 档示例：

```powershell
sakuratts serve MODEL --experimental examples/minimum-vram.json --runtime-mode managed
```

各档位的资源策略、适用模型和实测取舍见[推理档位](inference-profiles.md)。H 在片段内部交替加载 GPT 和声学模型，提前唤醒只准备进程、前端和配置；片段重载时间仍计入合成延迟。A / C / E 可以提前完成各自的模型加载步骤，首次执行、CUDA Graph 捕获和新参考编码仍可能增加首句开销。

## 桌宠与 LLM API 的接入流程

```text
开始请求大模型 ────────────────── 获得可朗读文本
     └─ 同时调用 wake ── 准备 TTS ──────┘
                                      └─ /tts ── 首段 PCM
```

桌宠收到用户消息后，按以下顺序安排语音：

1. 并行发起 LLM API 请求和 `POST /runtime/wake`，无需等 TTS 加载完成才请求 LLM。唤醒返回 202 表示已开始准备，不表示模型已加载。
2. LLM 返回可朗读文本后提交 `/tts`。如果唤醒尚未完成，请求会等待同一次准备；不必通过轮询 `/runtime` 才能开始提交。
3. 流式 LLM 由桌宠把增量文本组成句子，再串行提交语音请求，并按顺序播放。当前服务只有一个活动操作槽位，重叠的合成会返回 409。
4. 对话结束后停止续期，让空闲计时和保活窗口自然到期。LLM 失败或本轮无需语音时同样如此，无需主动关闭整个服务。

唤醒请求可以是：

```http
POST /runtime/wake
Content-Type: application/json

{"keep_alive_seconds":60}
```

如果 LLM 等待时间较长，可再次调用 wake 延长保活；具体范围见 [HTTP API](http-api.md#可选后台控制模式)。查询状态不会续期。默认 `direct` 模式没有这些生命周期接口，桌宠须显式启用 `managed`。

若提前量为 `T_lead`，可复用的准备耗时为 `T_wake`，准备后到首段 PCM 为 `T_speech`，没有其他资源争用时：

```text
文本就绪后的等待 ≈ max(0, T_wake - T_lead) + T_speech
```

新参考编码和执行时才创建的资源计入剩余等待。若大模型也使用本机 GPU，调用方应结合显存和并发负载选择提前时机。

保活窗口到期且没有活动操作时，服务按空闲规则休眠。大模型失败或最终不需要语音时，客户端可以让保活自然过期。`/runtime/sleep` 会影响整个服务，适合管理操作；它在忙碌时返回 409。

## 提前加载与执行预热

`wake` 完成的是所选档位的准备步骤，不会自动运行一条合成请求。A / C / E 返回 `preparation="model_load"`；H 返回 `preparation="runtime_init"`。精确状态含义由 [HTTP API](http-api.md#可选后台控制模式)维护。

对于桌宠，可先使用提前加载：它不需要知道 LLM 会返回什么文本，也不会生成无用音频。若需要进一步转移首次执行开销，宿主可在同一段 LLM 等待时间内安排一次**可选试合成**：

1. 使用当前角色实际的参考音频、转写和语言，提交一条短文本 `/tts`，设置 `parallel_infer=false`、`streaming_mode=false`。
2. 完整接收响应并检查是否成功，丢弃音频，不加入播放队列。提前断开连接会触发取消，不能据此认定预热完成。
3. 等试合成结束，再提交正式回复。若 LLM 文本先到，也要避免与试合成重叠；只想避免这段额外等待时，选择仅调用 wake。

试合成使用现有 `/tts` 接口，没有独立的 warmup 路由。它可能提前触发参考准备、内核执行和部分缓存建立，也会消耗算力和保活时间；不同文本长度、分句及资源回收策略会影响可复用范围，不能保证所有后续输入都已预热。H 档每个片段都重载模型，不适合用假句消除重载成本。

首次安装的模型转换、陌生参考编码应在正式交互前完成并验证。它们不能按普通睡醒的耗时估计。若 LLM 也在本机 GPU 上运行，还需实测同时加载与合成对 LLM 的影响。

### 已有睡醒延迟记录

2026-09-22 在 RTX 5060 / Windows WDDM、FP16 标准档 C、已有编译和参考缓存的短句测量中：

| 场景 | 耗时 |
| --- | ---: |
| 唤醒准备 | 4.172 s |
| 准备完成后提交第一句，到首 PCM | 1.078 s |
| 随后两句，到首 PCM | 0.239 / 0.234 s |
| 从休眠直接提交合成，到首 PCM | 4.976 s |

这些是各场景一到两个样本，不是 p95，也不是完整桌宠链路的实测。如果 LLM 返回文本前留出的时间足以覆盖加载，就能隐藏这部分等待；首次执行仍在文本就绪后的路径上，除非宿主另行安排试合成。表中的热句结果不代表已验证任意预热文本均有同样收益。模型、PCM 对照及采样条件见[原始报告](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-refinement-20260922.md)。

## 资源与测量

控制进程持有配置、状态和计时器；独立推理进程持有 Inference、GPT 及前端，并管理声学、经典日文前端和参考准备子进程。休眠会退出整棵自有推理进程树。实现取舍见 [ADR 0003](adr/0003-managed-runtime.md)，取消、代次和所有权要求见[推理契约](specs/inference-contract.md#资源生命周期)。

服务应分别测量控制进程主存、醒着空闲占用、活动峰值、准备耗时和首段音频延迟。RSS、私有提交使用主存指标；显存使用 GPU Dedicated / Shared Usage。进程退出后，计数实例消失记录为进程已退出。

已有 Windows 单机记录包括[真实模型睡醒与 PCM 对照](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-runtime-20260922.md)，以及[控制服务主存、H 档 HTTP 与标准档 100 次睡醒](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-refinement-20260922.md)。数小时待机、H 档长期反复睡醒和更多设备仍待验证，后续工作见[路线图](roadmap.md)。
