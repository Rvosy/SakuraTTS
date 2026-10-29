# Linux 服务与 MaiBot 插件整合

部署由三个部分组成：MaiBot 加载轻量语音插件，systemd 管理 SakuraTTS HTTP 服务，SakuraTTS 的 `managed` 模式管理推理进程。插件更新不会重装模型，引擎更新不要求重启 MaiBot。插件没有下载运行库、安装系统软件或执行 `systemctl` 的权限和职责。

```text
MaiBot Replyer → 回复扩展 → 私有网桥上的 POST /tts
                              ↓
                      systemd 管理 HTTP 服务
                              ↓
                 managed 按需启动 CPU 推理进程
                              ↓
                中文 + WAV 语音消息 → 适配器 → QQ
```

当前交付方式是源码安装 Linux 引擎、复制插件目录并安装配套 Host 补丁。尚无可下载的 Linux 整合包或市场发布版本；Windows 整合包不能在 ECS 上运行。模型、日文前端和参考条件应事先准备好，不能把安装 HTTP 依赖当作完成模型安装。

## 为什么采用独立服务

核对的 [gpt-sovits-voice-plugin](https://github.com/CGl123/gpt-sovits-voice-plugin) 将插件安装与 GPT-SoVITS 服务地址分开配置；[GPT-SoVITS-v2-TTS-Plugin](https://github.com/Shiroha135/GPT-SoVITS-v2-TTS-Plugin) 明确要求先运行 API；[maimbot_tts_adapter](https://github.com/Mai-with-u/maimbot_tts_adapter) 也要求另行安装、启动引擎。这些项目的使用说明没有提供“插件安装后自动下载并管理完整推理环境”的路径。

在容器中的插件里启动宿主机引擎，还需要跨容器进程控制、权限和文件映射。systemd 已能处理开机启动、日志和进程树回收，SakuraTTS 已能处理按需唤醒，无需再在插件中维护一套进程状态。决策见 [ADR 0008](../../../../docs/adr/0008-maibot-http-plugin.md)。

## 准备引擎

1. 将选定版本的源码放到 `/opt/sakuratts/releases/<版本>/`，按 [Linux CPU 指南](../../../../docs/linux.md) 在该目录建立 `.venv-linux-cpu`。保留版本目录，`/opt/sakuratts/current` 链接到这版源码。当前尚未发布的工作区改动需要随候选一起交付，直接下载旧版 main 不包含这些优化。
2. 将 CPU INT8 GPT、FP32 **分块声学包**、匹配的日文前端和预制参考条件放到 `/opt/sakuratts/shared/models/voice-cpu/`。检查 `model.json` 的路径；经典日文前端还需要有效的 Linux `frontend_python`。极限档遇到旧整图包仍可运行，但不会获得分块路径的内存收益。
3. 将与预制参考条件哈希匹配的原始音频放到 `/opt/sakuratts/shared/references/`。`/tts` 仍读取该音频；仅保留条件数组不够。更换参考音频之前，先离线准备对应条件，或按 HTTP 指南配置完整准备环境。

日语台词可能包含 `Navi`、`TTS` 等拉丁字母词，`all_ja` 也会将这些片段交给英文处理器。须同时准备[英文前端资源及依赖](../../../../docs/english-frontend.md)，并用日英混合文本验收；只有纯日语短句成功，不能证明实际角色回复可用。

这些资源的格式和转换方法由 [CPU 指南](../../../../docs/cpu-amd.md) 和 [HTTP 指南](../../../../docs/http-api.md) 维护。本部署示例不在 4 GB 共用服务器上执行原始权重导出。

## 安装服务

以下为首次安装步骤。以有管理权限的账号执行，先完成引擎和资源准备。已有部署应保留原配置，按下文更新流程处理。

```sh
sudo useradd --system --home-dir /opt/sakuratts --shell /usr/sbin/nologin sakuratts
sudo install -d -o sakuratts -g sakuratts /opt/sakuratts/shared/cache /opt/sakuratts/shared/logs
sudo install -m 0644 integrations/maibot_sakuratts/deploy/linux/service.yaml.example /opt/sakuratts/shared/service.yaml
sudo install -m 0644 integrations/maibot_sakuratts/deploy/linux/service.env.example /opt/sakuratts/shared/service.env
```

编辑这两个文件。源码、解释器、模型、参考及其所有父目录必须允许 `sakuratts` 账号读取或遍历；缓存与日志目录需要可写。不要递归更改其他业务目录的所有者。

`service.yaml` 是 2 vCPU 共用服务器示例，选择 `minimum-memory`，GPT / 声学各 2 线程，容量保持 2048。分块大小由正式档位根据模型决定，不在部署模板中复制。三档的测量依据见 [CPU 测量记录](../../../../research/notes/cpu-profiles-20261026.md)。这个部署选择不改变引擎的通用默认档位。

MaiBot 在宿主机上时，保留 `SAKURATTS_HOST=127.0.0.1`。在 Docker 中时，用以下命令查私有网桥网关，将 `SAKURATTS_HOST` 与插件 `engine.base_url` 同时改为该地址：

```sh
docker inspect maibot-core --format '{{json .NetworkSettings.Networks}}'
```

服务只绑定选定的本地地址，不开放公网端口。该接口没有认证，应只让可信客户端所在的私有网络访问。网桥重建后若地址改变，需要同步配置并重启引擎。

```sh
sudo install -m 0644 integrations/maibot_sakuratts/deploy/linux/sakuratts.service /etc/systemd/system/sakuratts.service
sudo systemctl daemon-reload
sudo systemctl enable --now sakuratts.service
systemctl status sakuratts.service
journalctl -u sakuratts.service -n 50
```

这一步才安装并启动常驻服务。服务故障由 systemd 有限重启；正常空闲由引擎在 60 秒后退出推理进程，HTTP 服务继续监听。不要用 `model_loaded=false` 判定故障，极限档和休眠时它本来就是 false；状态含义见 [HTTP 生命周期](../../../../docs/http-api.md)。

在 2 vCPU / 4 GB 共用 ECS 上，可用 `sudo systemctl edit sakuratts.service` 添加已用于测试的资源限制：

```ini
[Service]
MemoryMax=2200M
MemorySwapMax=512M
CPUQuota=150%
Nice=10
```

保存后重启引擎生效。`MemoryMax` 是整个服务进程树的上限，不是预留内存；这些值也不保证任意输入都不会触及限制。其他机器根据实际负载选择，不写入引擎通用默认值。

## 接入 MaiBot

按[插件安装说明](../../README.md#安装)复制插件，备份并填写配置。[plugin.toml.example](plugin.toml.example) 对应本服务模板：填写同一个地址、参考音频和真实日文转写。总超时为 300 秒，覆盖服务模板的唤醒及单次操作期限，并低于插件组件的 330 秒期限。配置路径属于引擎环境，MaiBot 无需挂载模型。

先在对应 MaiBot **源码副本**应用 [Host 补丁](../../host_patch/README.md)，运行 `git apply --check` 后再应用、构建候选镜像。正式运行应使用含补丁的版本化镜像；不要修改正在运行的容器文件。角色卡与角色交接内容沿用原配置。Replyer 的模型接口还须支持 JSON Schema，只有手动朗读成功不能证明这部分已就绪。

候选验证顺序：

1. 在隔离端口启动上述 `managed` 服务。使用插件环境运行真实引擎检查，输出目录必须不存在：

   ```sh
   python tools/check_engine.py --config config.toml --output /tmp/sakura-engine-check --wait-for-sleep
   ```

   检查经过真实插件的回复扩展、HTTP 合成与 WAV 校验，保存冷请求、热请求、自动休眠后的再次请求。纯文字分支不得唤醒引擎。工具不向 QQ 发消息，也不调用 LLM；中文和日语测试内容是固定输入。
2. 使用配套 Host 候选做回复回放，确认普通回复不受影响，中文与日语同次生成，格式错误不会发送，语音失败仍保留中文。既有回放入口见[插件验证说明](../../README.md#错误与验证)。
3. 在指定测试会话验证 `/tts` 和实际角色回复，试听音频，并检查 QQ 发送、适配器编码及历史记录。这一步会对外发消息，需要明确选定测试会话后执行。

## 更新与回退

保持插件、引擎、模型三个版本独立。新引擎安装到新的 `releases/<版本>`，新模型使用独立目录，先在隔离端口完成上述检查。验收后停止 `sakuratts.service`，切换 `current` 与对应配置，再启动服务。旧版本与配置保留；失败时停止服务、恢复旧链接和配置再启动。私有 HTTP 地址与契约保持兼容时，MaiBot 不需重装。

市场一键安装可以分发轻量插件；引擎完整离线包则需要另行制作并验证 Linux 运行库、前端资源与模型清单。当前没有发布地址，因此不在插件里加入下载按钮或自动升级代码。
