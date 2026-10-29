# SakuraTTS 的 MaiBot 插件

通过 MaiBot 的回复扩展，让 Replyer 在同一次生成中组织中文回复与可选日语台词。日语台词交给独立 SakuraTTS HTTP 服务合成，文字和语音由 MaiBot 统一发送。保留 `/tts 日语台词` 手动朗读命令，不再注册绕过 Replyer 的独立语音 Tool。

适配目标为 MaiBot 1.3.14 和 SDK 2.16.0，版本区间以 [_manifest.json](_manifest.json) 为准。配置、默认值和 WebUI 字段由 [plugin.py](plugin.py) 中的 `SakuraConfig` 定义。当前为 0.2.5 候选版，尚未发布到插件市场；部署与验收范围见 [N.A.V.I. 音色部署记录](../../research/notes/maibot-navi-voice-20261026.md)。

**此候选需要配套的 [Host 补丁](host_patch/README.md)。** 当前 Host 尚未把回复 Hook 指定的 JSON Schema 传给模型接口；只安装插件会缺少这项格式约束。补丁在隔离容器中验证，不自动安装，也不修改线上 Host。

## 回复链路

Planner 通过原有 `reply` 工具选择扩展，不填写台词：

```json
{"plugin_options":{"rvosy.sakura-tts.voice":{"mode":"auto"}}}
```

不选择扩展时沿用普通文字回复。`auto` 允许 Replyer 根据交流选择是否附语音；`requested` 用于用户明确要求语音的情况。参数不包含情绪、人物立场或台词，角色理解继续由 Replyer 根据共享人物卡和原作资料形成。

扩展的 `prepare` 阶段向 Replyer 提供中文与日语的语义要求：

```json
{"text":"完整中文回复","voice_ja":"对应其中一个意思的简短日语台词"}
```

`auto` 模式的 `voice_ja` 可以为 `null`，此时不调用引擎。`requested` 模式要求日语台词非空。两种语言由同一次 Replyer 生成，不增加翻译模型调用；日语延续同一角色的自称、称呼和表达习惯。插件不修改人物卡、原作资料或结构化交接字段。

`maisaka.replyer.before_request` Hook 为选中的回复设置 `response_schema`，配套 Host 补丁将它传入已有 `LLMGenerationOptions.response_format`。实际模型接口必须支持 JSON Schema 输出；`auto` 允许空台词，`requested` 要求字符串。格式约束与语义要求在同一次调用中生效，不改写角色提示或匹配替换 Host 文案。回放证明只靠提示词不足以可靠取得结构化结果，因此不保留这条兜底路径。

只有选用扩展的回复跳过普通文字分句等后处理，以保留 JSON。`before_send` 严格解析正文，替换为中文并保留附件、@ 和引用，再按需附加日语 `voice` 消息段。该消息段的 `data` 保存实际日语台词，供 Host 的历史记录使用。扩展不直接发送消息，也不缓存每轮的台词。

格式错误在回复扩展的发送边界抛出，Host 阻止该次回复发送。不能只依赖普通 Hook 抛错：当前 Host 的部分 Hook 错误会回退到默认处理。其他扩展若也改写完整正文，需要在隔离环境另行验证组合顺序。

## 安装

将本目录复制到 MaiBot 的 `plugins/sakura_tts/`，或将插件压缩包解压到该目录。`plugin.py` 和 `_manifest.json` 必须直接位于 `plugins/sakura_tts/` 下，不能再嵌套一层同名目录。Docker 部署时，复制到宿主机映射给 MaiBot `plugins/` 的目录。

在 WebUI 中加载插件并填写配置。MaiBot 读取 manifest 安装 HTTP 客户端依赖；不要在 MaiBot 环境中安装 SakuraTTS、模型或推理依赖。Runner 首次加载会生成 `config.toml`，也可以在生成后直接编辑：

```toml
max_text_length = 300

[plugin]
config_version = "1.0.0"

[engine]
base_url = "http://127.0.0.1:9880"
timeout_seconds = 120

[voice]
ref_audio_path = "/opt/sakuratts/shared/references/normal.wav"
prompt_text = "这里填写参考音频的真实转写"
speed_factor = 1.0
```

`voice.speed_factor` 为固定语速，手动命令与自动回复共用；例如 `0.9` 轻微放慢，默认 `1.0`。非原速需要引擎支持 HTTP 调速并安装 FFmpeg，先升级引擎再修改倍率。

参考音频使用日语，`prompt_text` 填写实际日文转写。请求固定 `text_lang=all_ja`、`prompt_lang=ja`；模型不能通过扩展参数更改合成语言。音频路径由**引擎进程**读取，不是 QQ 用户上传路径，也不是 MaiBot 容器中的路径。

在测试会话发一条 `/tts こんにちは。` 可检查手动朗读链路。命令名不区分大小写，也支持前置一个群聊 @ 名称，例如 `@N.A.V.I. /TTS こんにちは。`；命令后的文本直接交给引擎，不经过 Replyer 翻译或改写。角色回复另需验证 `reply.plugin_options` 中的扩展调用。组件权限沿用 MaiBot 的管理设置。插件使用标准 `voice` 消息和 Base64 音频，MaiBot 与消息适配器不需要共享临时 WAV 文件。

手动朗读通过 `send.hybrid` 发送单个语音段，段内同时保存请求者和朗读原文，并注明台词由请求者提供。Host 仅在发送成功后将这条消息写入历史并同步当前会话上下文；后续对话可以引用它，重启后也可随普通历史恢复。群里仍只发送语音，记录本身不触发额外回复。原始 `/tts` 指令继续按命令拦截，不交给 Replyer 改写。

## Linux 服务部署与引擎更新

使用独立 Linux CPU 环境运行 SakuraTTS，由 systemd 管理服务启动与故障重启，`managed` 模式负责空闲休眠和按需唤醒。插件通过私有网络调用 `/tts`。Windows 整合包不能在 Linux 上安装。

[Linux 整合指南](deploy/linux/README.md) 提供服务模板、2 vCPU 共用服务器的极限省内存配置、参考资源要求、真实联调命令，以及引擎更新和回退步骤。插件不会安装系统软件或控制宿主机进程；当前没有已发布的 Linux 一键整合包。

## 插件自身更新

更新前备份当前 `config.toml`，替换插件源码、manifest 和依赖文件，保留该配置，再通过 MaiBot 重载插件。Runner 会按配置模型补齐新增默认值；现有地址和音色不会由本插件重置。配置热更新对下一次请求生效，已开始的请求继续使用启动该请求时的配置。

本插件目前保存在 SakuraTTS 的 `integrations/maibot_sakuratts/` 子目录，不能将 SakuraTTS 整仓库 URL 当作市场插件安装地址。若要通过市场分发，应将本目录发布为独立公开仓库，让 `plugin.py`、`_manifest.json`、README 和 LICENSE 位于仓库根目录，更新 manifest 的仓库地址，再按 [MaiBot 发布指南](https://docs.mai-mai.org/plugin/submission) 发布版本并提交索引。插件版本和引擎版本分别维护。

## 错误与验证

手动 `/tts` 命令进入插件的内存队列，后台按接收顺序逐条合成、发送，不向聊天发送排队或忙碌提示。各聊天共用一个队列，每条任务保留自己的目标会话和请求者。命令入队即返回，排队时间不占用 Host 的命令调用期限；每条任务开始合成后才计算 HTTP 总超时。插件卸载或重启会取消正在处理及尚未完成的任务，队列不跨重启保留。

手动队列与回复扩展共用合成锁，一次只向引擎提交一条请求，不自动重试。连接失败、超时和 HTTP 错误保留在插件日志中。回复扩展的合成失败会保留已经生成的中文：`auto` 只发送中文，`requested` 同时附上语音未生成的说明。手动任务失败时提示原因并继续下一条，入队成功不代表语音已发送。队列中的编程异常记录完整堆栈，不阻塞后续任务。

收到 `stopped_at_limit`、无效 WAV 或截断数据时不发送音频。为限制 Base64 与 IPC 的内存复制，一条音频最多 4 MiB。字符上限可在配置中调整；总超时最多 300 秒。合成在整组消息发送前执行，中文也会等待合成结束或失败。实际发送仍依赖消息适配器及平台可用性，不承诺整组消息在平台上原子送达。

在 SakuraTTS 仓库根目录运行插件测试，无需安装引擎模型：

```sh
uv run --no-project --with maibot-plugin-sdk==2.16.0 --with httpx==0.28.1 --with pytest \
  python -m pytest integrations/maibot_sakuratts/tests -q
```

测试使用真实 SDK、HTTP 替身和 Host RPC 替身，覆盖请求、发送失败、错误音频、超时取消及配置变更。它们不向 QQ 发送消息，不能替代目标 ECS 的真实模型合成和语音发送验收。

在上述命令的 `uv run` 参数中增加 `--with fastapi==0.141.1 --with numpy==2.4.6`，还会运行当前 SakuraTTS ASGI 服务的接口契约测试：实际经过 `/tts` 路由、请求校验和 WAV 编码，推理计算使用替身。没有这两个依赖时该项跳过。

隔离回放工具在 [tools/run_reply_replay.py](tools/run_reply_replay.py)，基于角色交接候选的回放脚本适配，运行真实 Planner、Replyer、角色插件和 Host 回复扩展控制器。仅允许 `/eval/baseline` 与 `/eval/candidate`，TTS 使用容器内 HTTP 替身。环境配置和原始聊天留在受限的评估目录，不进入插件压缩包。具体回放结果单独记录，不能将合成替身视作真实音色或延迟验证。

完成隔离配置后，可加 `--real-tts` 使用插件配置中的真实引擎，跳过 HTTP 替身；与 `--validate-only` 组合时验证真实 Host / Runner / 引擎链路，模型使用固定输出替身，不调用外部模型或发送聊天消息。独立的 [check_engine.py](tools/check_engine.py) 还会保存 WAV、响应头和休眠恢复记录。两者的整合实测见 [2026-10-26 验证记录](../../research/notes/maibot-engine-integration-20261026.md)。

回复协议参考 [MaiBot 回复扩展](https://docs.mai-mai.org/plugin/reply-extensions)；首版 HTTP 客户端与标准语音消息设计参考 [GPT-SoVITS 语音插件](https://github.com/CGl123/gpt-sovits-voice-plugin)。本插件按 SakuraTTS 契约独立实现。
