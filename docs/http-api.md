# HTTP 服务配置与生命周期

本页说明服务启动、后台控制和诊断。客户端路由、请求字段、默认值与错误处理集中在 [API V2 使用说明](api-v2-guide.md)；字段实现见 [server.SpeechRequest](../src/sakuratts/server.py)。

## 启动

复制 [配置示例](../examples/tts_infer.example.yaml)，填写模型、转换解释器和源码路径：

```powershell
.\start-server.bat -c configs/tts_infer.yaml
# 同等入口
python api_v2.py -a 127.0.0.1 -p 9880 -c configs/tts_infer.yaml
python api.py -a 127.0.0.1 -p 9880 -c configs/tts_infer.yaml
```

路径按启动工作目录解析，和原版一样；建议使用绝对路径。脚本固定从仓库目录启动。配置提供 `custom.t2s_weights_path` / `vits_weights_path` 时，先转换或复用部署缓存，再加载权重。`sakuratts.model` 可引用已有部署包以跳过完整转换。原始模型和原版配置文件不会被修改。

无参数时依次检查 `configs/tts_infer.yaml`、原版目录下的 `GPT_SoVITS/configs/tts_infer.yaml`，不会自动发现角色包。显式模型或 `-c` 优先。没有配置时 HTTP 可以启动，但 `model_loaded=false`，合成会提示配置模型。默认 `direct` 模式在启动时加载 GPU 权重；第一次合成仍可能有内核编译等开销。低级 `Engine.load` 保留延迟加载。总体 `policy="staged"` 可用于低级接口或显式启用的 `managed` HTTP 模式，`direct` 仍拒绝该策略。

## 可选后台控制模式

需要长期待机的调用方可以显式选择 `managed`。省略选项或选择 `direct` 时，沿用原有启动加载、请求处理和驻留行为，不启用空闲休眠或 `/runtime` 接口。

```powershell
.\start-server.bat -c configs/tts_infer.yaml --runtime-mode managed --idle-sleep-seconds 60
# 已转换的模型包也可以直接使用
sakuratts serve models/sakura --runtime-mode managed
```

控制模式启动后处于 `sleeping`，收到提前唤醒或需要推理的请求后再启动完整推理进程。空闲期结束后退出该进程及其子进程，HTTP 控制服务继续接受请求。默认保留 FP32；运行模式与显存档位分别选择。A / C / E 可用于两种模式，H 极限档的总体 `policy="staged"` 仅在 `managed` 中开放。

显存优先时，可在控制模式中显式选择 H 档：

```powershell
sakuratts serve MODEL --experimental examples/minimum-vram.json --runtime-mode managed
```

将 `MODEL` 换成准备好的 FP16 chunk256 模型目录或配置文件。该选项不会转换权重。H 档按片段交替加载 GPT 和声学模型，降低活动峰值和醒着空闲时的显存，代价是每片段都要等待重载；提前唤醒只完成运行环境准备，不能消除这些重载。需要连续短句速度时，可选择 `examples/fp16.json`，再由控制模式在长时间空闲后退出推理进程。档位说明见[推理档位](inference-profiles.md)。

| 启动参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--runtime-mode` | `direct` | `managed` 启用进程生命周期控制 |
| `--idle-sleep-seconds` | `60` | 最后一次操作完成后，到自动休眠的空闲秒数 |
| `--wake-timeout-seconds` | `120` | 推理进程启动和所选档位准备的超时秒数 |
| `--operation-timeout-seconds` | `300` | 单次工作进程操作的总超时秒数，长文本或首次参考准备可按需调大 |

三个时长都必须是有限正数，仅控制模式使用。`api.py`、仓库启动脚本和整合包启动脚本都转发这些参数；不会因为配置了时长或选择低显存档就自动启用控制模式。

| 方法与路径 | 行为 |
| --- | --- |
| `GET /runtime` | 查询资源状态，不启动推理进程、不续期 |
| `POST /runtime/wake` | 开始或合并一次准备，准备中返回 202，按所选档位准备完成后返回 200 |
| `POST /runtime/sleep` | 空闲时退出推理进程；合成、准备或切换中返回 409；已经休眠时可重复调用 |
| 原有 `GET/POST /tts` | 休眠时自动唤醒，正在准备时等待同一次准备；第二条合成仍返回 409 |

`wake` 可以省略请求体，或传 `{"keep_alive_seconds": 60}`。保活时长允许 `0` 至 `3600` 秒，默认 `60`；重复唤醒可以延长保活，不会缩短已有窗口。正在准备时，窗口从准备完成开始计算。没有后续语音请求也会在保活和空闲期限都结束后自动休眠。`sleep` 是服务级操作，可以提前结束保活，但不会中断活动请求。

```powershell
# 开始请求大模型时并行发送，文本到达后照常调用 /tts
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:9880/runtime/wake -ContentType application/json -Body '{"keep_alive_seconds":60}'
Invoke-RestMethod -Uri http://127.0.0.1:9880/runtime
```

`/runtime` 返回 `state`（`sleeping`、`waking`、`awake`、`stopping`、`failed`）、`model_configured`、`model_loaded`、`model`、`worker_pid`、`busy`、`last_error`、`last_wake_ms` 和 `idle_sleep_seconds`，另有进程代次 `generation`、保活剩余秒数 `keep_alive_remaining_seconds` 及准备范围 `preparation`。唤醒和休眠接口返回相同的状态结构。`busy` 表示已有合成、切换或参考准备请求；单独提前唤醒时它可以是 `false`，此时仍须结合 `state` 判断，准备中不能休眠。

`awake` 表示所选档位的准备已经完成：

| 档位 | `preparation` | `awake` 时的保证 |
| --- | --- | --- |
| A / C / E | `model_load` | 模型加载步骤完成，`model_loaded=true`；E 的声学 Session 仍按原策略在执行时创建 |
| H | `runtime_init` | 推理进程、前端和配置已准备，GPU 权重留到执行时错峰加载；`model_loaded` 始终为 `false`，`model` 可以返回当前所选模型 |

Prefill、Decode、CUDA Graph 捕获和新参考编码在实际请求时执行。H 档应结合 `state` 与 `preparation` 判断准备情况，不要等待 `model_loaded` 变为 `true`。`last_wake_ms` 是最近一次成功唤醒的耗时，不是首段音频延迟。`/health` 使用相同的 `model_loaded` 含义；休眠或 H 档的 `false` 是正常状态。

控制模式不在启动时执行 GPU 加载。只有唤醒后才能发现的资源、运行库或 CUDA 加载错误会记录为 `failed`，等待加载的请求也会收到错误。未配置模型时不创建 worker。失败不会无限自动重启；清理完成后，下一次唤醒或合成可以重试。

休眠保留本次服务会话中成功选择的模型与参考准备设置；重启服务仍恢复启动配置。已准备参考不替代 `/tts` 的 `ref_audio_path`。控制服务不加载 NumPy、GPU 计算库或文本前端，但仍占用解释器和 HTTP 所需的主存。运行方式选择和资源测量见[后台驻留与提前唤醒](background-runtime.md)。

## 请求与模型切换

[API V2 使用说明](api-v2-guide.md#当前支持哪些调用)列出原版路由、参数表与调用示例。`/docs` 和 `/openapi.json` 提供本服务的 GET 查询字段与 POST JSON 模型。

嵌入 ASGI 应用时，进程控制由宿主提供回调。权重切换只影响当前服务会话；重启恢复启动配置。初始模型及准备环境通过配置提供，`/tts` 每次仍须传参考音频、转写与语言。

请求参数在提交推理工作前校验，managed 模式下被拒绝的请求不会唤醒推理进程。非流式分桶会按规范化文本长度排序并改变各句的随机数消耗顺序；显式 `split_bucket=false` 保留原文推理顺序。报告的 `execution_order` 记录实际执行顺序，`fragments` 与完整 PCM 按原文排列。

## 生命周期与输出

服务默认在每片声学计算结束后收缩 ORT 显存池，释放空闲工作区；长请求后无需重启服务来回收这些分配。模型权重和仍在使用的缓冲区继续保留。该策略也适用于逐片返回音频的模式 1，不改变精度、分句或采样参数。需要测量旧策略时，可通过 `sakuratts serve --experimental options.json` 传入 `{"acoustic_arena_shrink": false}`；这是启动选项，不是 `/tts` 请求参数。

终端用一条横线分隔请求，依次显示请求参数、参考音频、输入文本、推理过程和一行完成摘要。文本单独缩进，按终端宽度换行。推理过程保留原版的“提取文本Bert特征”“预测语义Token”“合成音频”阶段名称，下面缩进显示片段与音素数量、进度、停止状态及耗时；日语沿用原版的零 BERT 特征，不加载中文 BERT 模型。多片段请求显示 `[1/2]` 等片段编号，单片段省略编号。模型文件名只在启动和切换权重时显示；每次请求对应的模型、规范化文本和阶段耗时汇总仍写入详细日志。普通消息省略 `INFO`，成功的 HTTP 访问不重复显示；警告和错误保留明确标记。`seed=-1` 时显示本次实际使用的种子，参考音频可复用时统一显示“已准备”。

终端中的 GPT 行随实际采样更新，保留原版常用的 `it`、`it/s`、`EOS` 等标记。`it` 是采样迭代次数，包含最后预测 EOS 的一次，不等同于交给 SoVITS 的有效 Token 数。结束时输出 `T2S Decoding EOS [157 -> 410]` 这样的记录：左侧是参考语义前缀长度，右侧按原版 batch 日志计算，包含参考前缀及所有采样迭代，尚未剔除最后的 EOS。数字来自当前片段。仅因长度限制停止时显示 `STOP` 和警告，不冒充 EOS；EOS 与上限同时出现时保留两种状态。进度不以保护上限计算百分比或预计剩余时间。非终端输出每秒最多记录一次中间进度。取消或失败会关闭进度显示并标记状态，音频编码失败不会打印完成摘要。

完整诊断默认保存到 `logs/sakuratts.log`，包含时间、级别、请求编号、完整路径和模型哈希、输入与规范化文本、参考转写、实际参数、缓存来源、细分停止原因和异常堆栈。单文件达到 10 MiB 后轮转，最多保留 3 份备份。准备进程的原始输出也写入该文件；已知的 `enc_q` 训练模块缺失项不在终端展开，其他未匹配权重和警告仍显示。

`start-server.bat --log-level debug` 可在终端展开排错细节，`--log-file PATH` 可指定诊断文件。级别只控制终端显示，文件始终保留完整诊断；`warning` / `error` 级别不会显示动态进度。多个服务实例应指定不同日志文件。

总耗时包含参考准备，输出 RTF 按包含句间静音的音频时长计算。非流式总耗时不含音频压缩和网络发送；流式总耗时包含片段编码及等待发送队列的时间。日志包含输入文本和本地路径。嵌入 ASGI 时由宿主配置 `sakuratts` 日志级别为 `INFO` 以显示内部进度；低级 SDK 默认保持安静。

一个专用线程负责模型创建、推理、切换和关闭。正在计算或切换时，冲突操作返回 409；这是当前与原版不同的并发约束。模型格式检查失败时保留现有实例；旧 GPU 权重释放后发生的加载失败会使服务处于未加载状态，需要重新加载模型。

完整响应的 `X-SakuraTTS-Status` 区分 `completed` 和 `stopped_at_limit`。`X-SakuraTTS-Request-Ms` 包含本次参考解析和原生推理，音频压缩与网络发送不在其中。达到长度限制不表示内容完整。

流式队列最多保留两个片段，发送慢时阻塞生产。`direct` 和 `managed` 的完整响应及流式响应都监听客户端断开，包括首片音频产生之前。断开后发出取消信号，计算边界检查该信号；当前内核或参考准备尚未返回时不会提前释放忙碌状态。第一片之前失败可以返回 JSON 错误；已开始发送后失败会中断连接，不补成成功音频。模型加载、显存释放、句间 RNG 顺序沿用原生实现。

首次原始参考编码需要配置中的原始检查点、HuBERT / 说话人模型与准备环境；缓存命中后的普通推理不需要 PyTorch。缓存保存音频条件，参考文本特征按当前文本、语言和前端重新计算，修改转写不会误用上一次文本。
