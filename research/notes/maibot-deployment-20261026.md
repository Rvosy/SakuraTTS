# MaiBot SakuraTTS 部署记录

日期：2026-10-26。通用安装、升级和回退方式见 [Linux 整合指南](../../integrations/maibot_sakuratts/deploy/linux/README.md)。

本文记录首次部署快照。后续已更换 N.A.V.I. 音色，并将命令改为 `/tts`，当前部署见 [音色修正记录](maibot-navi-voice-20261026.md)。

ECS 已部署 SakuraTTS 插件 0.2.0 候选，MaiBot 核心镜像为 `maibot-sakuratts:1.3.14-integration-20261026`，包含 Replyer JSON Schema Host 补丁。插件安装在 `/opt/maibot-snowluma/plugins/sakura_tts/`，线上组件查询确认插件 enabled、loaded，`sakura_tts` 命令已启用、权限为 public，load_error 为空。

引擎位于 `/opt/sakuratts/releases/20261026-integration/`，`current` 指向该版本。模型、日文前端和中性参考已复制到 `shared/`，运行环境使用系统 Python 3.12.3 新建 venv 并复用已验证的依赖文件。运行不再依赖临时基准目录。

`sakuratts.service` 已启用开机启动，以 `sakuratts` 专用账号运行，监听私有 Docker 网桥 `172.20.0.1:9880`。CPU 配置为 `minimum-memory`，GPT / 声学各 2 线程，容量 2048；空闲 60 秒后退出推理进程。资源限制与单次操作期限沿用整合指南。

## 验证

- 新目录、新运行环境和专用账号下完成真实合成、连续请求、自动休眠及再次唤醒。三次请求分别耗时 15.91、9.44、11.12 秒，均为 HTTP 200 / completed。这是部署检查，不是性能对照。
- 复制当前模型配置到隔离实例，实际调用线上使用的 `gpt-6-astra`，验证插件的 requested JSON Schema，返回的中文和日语通过发送边界解析。该调用使用固定测试输入，没有向聊天平台发送。
- MaiBot 核心容器重建成功，扩展 Runner 加载失败数为 0，QQ 适配器重新连接。语音插件通过线上 WebUI 的只读接口确认已加载。
- 当前角色文件及原有 TOML 配置 SHA256 均未改变。QQ 适配器容器 ID 和启动时间不变，console / telemetry 服务 PID 保持 874 / 875。
- 最终引擎处于 sleeping、worker_pid=null、last_error=null，服务 active 且 NRestarts=0。

MaiBot 启动时按原有依赖流水线补装了其他已安装插件声明的缺失 Python 依赖。引擎依赖离线复用，不与 MaiBot 共用环境。

QQ 实际发送和人工试听留待聊天中验收，可先发送 `/sakura_tts こんにちは。`。角色回复可用“请用日语语音跟我说晚安，并附中文。”触发；是否选择回复扩展由 Planner 判断。普通 auto 回复允许只有文字。

## 备份与回退

原 Compose、配置与受保护文件哈希保存在服务器 `/opt/maibot-snowluma/backups/sakuratts-deploy-20261026/`。该目录只允许管理账号访问。旧引擎基准文件和原 MaiBot 镜像保留。

需要回退本次部署时，在服务器执行 `python3 /opt/sakuratts-deploy-20261026/rollback.py`。脚本检查当前镜像是否仍匹配本次部署，将语音插件移入备份目录，恢复原核心镜像，停用 SakuraTTS 服务；不删除模型、日志或角色配置。后续如果已升级为其他镜像，应先重新检查回退目标。

本地验证证据在 `outputs/maibot-deploy-20261026/`；服务器部署脚本与完整结果在 `/opt/sakuratts-deploy-20261026/`。包含连接配置的备份和隔离实例留在服务器，不进入插件包。
