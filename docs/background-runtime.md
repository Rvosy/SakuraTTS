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

## 客户端如何提前唤醒

```text
开始请求大模型 ────────────────── 获得可朗读文本
     └─ 同时调用 wake ── 准备 TTS ──────┘
                                      └─ /tts ── 首段 PCM
```

调用方并行请求大模型和 `/runtime/wake`，文本就绪后提交 `/tts`。流式大模型的断句、播放和跨请求音频队列由调用方管理。

若提前量为 `T_lead`，可复用的准备耗时为 `T_wake`，准备后到首段 PCM 为 `T_speech`，没有其他资源争用时：

```text
文本就绪后的等待 ≈ max(0, T_wake - T_lead) + T_speech
```

新参考编码和执行时才创建的资源计入剩余等待。若大模型也使用本机 GPU，调用方应结合显存和并发负载选择提前时机。

保活窗口到期且没有活动操作时，服务按空闲规则休眠。大模型失败或最终不需要语音时，客户端可以让保活自然过期。`/runtime/sleep` 会影响整个服务，适合管理操作；它在忙碌时返回 409。

## 资源与测量

控制进程持有配置、状态和计时器；独立推理进程持有 Inference、GPT 及前端，并管理声学、经典日文前端和参考准备子进程。休眠会退出整棵自有推理进程树。实现取舍见 [ADR 0003](adr/0003-managed-runtime.md)，取消、代次和所有权要求见[推理契约](specs/inference-contract.md#资源生命周期)。

服务应分别测量控制进程主存、醒着空闲占用、活动峰值、准备耗时和首段音频延迟。RSS、私有提交使用主存指标；显存使用 GPU Dedicated / Shared Usage。进程退出后，计数实例消失记录为进程已退出。

已有 Windows 单机记录包括[真实模型睡醒与 PCM 对照](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-runtime-20260922.md)，以及[控制服务主存、H 档 HTTP 与标准档 100 次睡醒](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/managed-refinement-20260922.md)。数小时待机、H 档长期反复睡醒和更多设备仍待验证，后续工作见[路线图](roadmap.md)。
