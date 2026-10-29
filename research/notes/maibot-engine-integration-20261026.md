# MaiBot 与 CPU 引擎整合验证

日期：2026-10-26。部署方式和有效用法见 [Linux 整合指南](../../integrations/maibot_sakuratts/deploy/linux/README.md)，本页只保存验证条件与结果。

## 条件

使用当前工作区的 CPU 优化源码与 SakuraTTS 插件 0.2.0 候选，Sakura V2ProPlus、CPU INT8 GPT / FP32 分块声学包，`minimum-memory` 档，GPT / 声学各 2 线程，容量 2048。复用 ECS 已有的 Python 3.12.3、ORT 1.30.0、日文前端和中性参考条件，不在线安装引擎依赖。

引擎在独立 transient systemd 服务运行，私有网桥监听 `172.20.0.1:19880`。`managed` 空闲期限 60 秒，操作期限 110 秒。限制为 `MemoryMax=2200M`、`MemorySwapMax=512M`、`CPUQuota=150%`、`Nice=10`，运行上限 1800 秒，停止超时 30 秒。服务源码冻结在 `/opt/sakuratts-maibot-integration-20261026/source/`，未替换常驻服务。

客户端使用同一私有网桥上的一次性 MaiBot 镜像容器，调用真实插件的 `reply_voice(before_send)`，经过 HTTP、实际 CPU 推理、完整 WAV 检查、Base64 编码，得到标准语音消息段。该检查使用固定的中文和日语输入，不调用 LLM、不发送聊天消息。

文本为 `こんにちは。今日はいい天気ですね。`。请求保持插件现有行为：`all_ja`、非流式 WAV，未覆盖服务的 `cut5`、随机种子及其他采样默认值。因此输出时长和 PCM 可以不同，本轮不做逐字节音频对照，也不将其耗时与之前 `seed=1234 / cut0` 基准直接比较。

## 真实 HTTP 与休眠

| 请求 | 客户端完整耗时 | 音频长度 | 参考来源 | 结果 |
| --- | ---: | ---: | --- | --- |
| 冷请求 | 13.93 s | 4.28 s | package | HTTP 200 / completed |
| 热请求 | 8.46 s | 4.24 s | memory | HTTP 200 / completed |
| 自动休眠后请求 | 11.39 s | 4.00 s | package | HTTP 200 / completed |

纯文字 `voice_ja=null` 分支没有请求引擎，也没有唤醒推理进程。冷请求后的进程代次为 1；60 秒空闲后状态回到 `sleeping`，`worker_pid=null`；再次请求后代次为 2，使用新 PID，返回完整音频。极限档唤醒后 `model_loaded=false`、`preparation=runtime_init` 符合契约。

整个服务进程树的 cgroup `memory.peak` 为 **1129.18 MiB**，`memory.swap.peak` 为 **24.64 MiB**；`memory.events` 的 max、oom、oom_kill 均为 0。该峰值包括 HTTP 控制进程、推理进程、文件缓存及内核记账，不等同于单个推理进程 RSS，也不与之前单进程基准相加。

最后一次 Host 验证后、停止服务前的进程快照：HTTP 控制进程 RSS 20.45 MiB、VmSwap 23.93 MiB；该代推理进程 RSS 404.77 MiB、VmHWM 528.97 MiB、VmSwap 0。这个快照不是三代进程的 RSS 峰值测量。后台其他业务继续运行；本轮没有做背景负载一致的性能对照或长时间压力测试。

首次客户端尝试缺少 `[plugin].config_version`，被 SDK 配置边界拒绝，没有进行合成。补齐联调配置后重新运行，失败日志保留，不计入上述请求耗时。

## Host 候选

在服务器构建候选镜像 `maibot-sakuratts:1.3.14-integration-20261026`，基于现有 `maibot-sakura-support:1.3.14-memory-attribution`，把 7 行 Schema 补丁固化进镜像。没有发布镜像或替换生产容器。

从基础镜像提取原文件，在独立源码目录执行 `git apply --check` 和 `git apply`；结果与候选镜像构建输入逐字节一致。镜像 ID 与源码哈希见 `host-patch-verification.json`。

新的隔离实例复用冻结的角色交接候选，使用真实 Host、插件 Runner 和 CPU HTTP 服务，通过 `run_reply_replay.py --structured --validate-only --real-tts` 验证：

- auto 的纯文字与真实语音分支正常，标准语音消息经过 Runner RPC 返回 Host。
- 非 JSON 和字段类型错误在发送边界被拦截；日志中对应异常是预期的负例。
- 普通回复、auto、requested 的 Schema 选项传递正确，模型替身每轮调用一次。
- 角色交接 Hook 与扩展可以同时加载，未修改角色卡。

此轮没有重新调用真实 Planner / Replyer 模型，不能代替前一轮真实模型回放或日常回复频率验收。QQ 发送、适配器编码、人工听音仍未验证。

## 检查与清理

本地实际执行 28 项测试、18 个 subtest，覆盖插件 API 契约、错误处理、回复扩展和 CPU 档位 / CLI 组合；两份联调脚本通过 Python 编译检查，`git diff --check` 通过。服务模板替换为临时测试路径和现有账号后，通过 ECS 的 `systemd-analyze verify`。正式 `sakuratts` 服务账号与 `/opt/sakuratts/current` 布局没有安装，仍需按指南配置读写权限。

临时引擎服务已停止，`Result=success`、`ExecMainStatus=0`、`MainPID=0`；验证容器自动删除。`sakura-console.service` 与 `sakura-telemetry.service` 保持 active，PID 仍为 874 / 875；线上 `maibot-core` 未重启。候选镜像和隔离文件保留用于复核。

本地证据位于 `outputs/maibot-engine-integration-20261026/`：`smoke-r2/result.json` 与三份 WAV、`resources.json`、服务限制与退出记录、Host 补丁及验证标记、源码哈希清单。服务器原始记录位于 `/opt/sakuratts-maibot-integration-20261026/`。包含角色配置的 `host-check/` 留在服务器受限目录，不进入分发包。

## 待机占用补测

同日另启一次临时服务，沿用上述模型、档位与资源限制，仅绑定宿主机 `127.0.0.1:19881`。一次真实短句合成耗时 13.30 秒，随后等待 60 秒自动休眠。表中 RSS 为当时服务进程树的 RSS 合计，不是峰值，也不与 cgroup 相加。

| 状态 | 进程 RSS 合计 | cgroup memory.current | 其中匿名内存 | 其中文件缓存 | cgroup swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 刚启动，尚未唤醒 | 45.45 MiB | 37.52 MiB | 30.79 MiB | 6.41 MiB | 0 |
| 合成结束，等待休眠 | 440.50 MiB | 999.38 MiB | 397.04 MiB | 600.18 MiB | 0 |
| 自动休眠后 | 41.36 MiB | 576.95 MiB | 31.48 MiB | 544.89 MiB | 0 |

休眠后只剩 HTTP 控制进程，`worker_pid=null`。Linux 仍将读取过的模型文件保留在文件缓存中，内存有压力时可以回收；这解释了监控中的 cgroup 总量仍高于控制进程 RSS。未清空系统缓存或强制回收，因此不能把休眠后的整个服务记账描述为只有约 41 MiB。

临时服务退出成功、`MainPID=0`，没有安装常驻服务。原始证据在上述本地和服务器目录的 `standby-20261026/` 中。
