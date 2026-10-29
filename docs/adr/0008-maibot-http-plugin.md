# ADR 0008：MaiBot 插件通过 HTTP 调用独立引擎

状态：已接受。日期：2026-10-26。

MaiBot 需要调用 SakuraTTS 发送语音，同时允许引擎继续优化、升级和回退。将推理依赖直接安装到 MaiBot 环境会把两者的 Python 版本、依赖和重启周期绑在一起，也使插件更新承担模型与运行库分发。

插件放在 `integrations/maibot_sakuratts/`，通过已有的 API V2 `POST /tts` 调用独立服务。插件不导入 SakuraTTS，不保存模型，不控制引擎进程。推理后端、模型加载与空闲休眠由引擎维护，现有 HTTP 契约保持不变。

Linux 部署采用 systemd 管理 HTTP 服务及其整个进程树，引擎的 `managed` 模式管理推理进程休眠与唤醒。插件中的进程管理会重复已有生命周期逻辑，并要求容器访问宿主机进程控制；因此不在插件中增加引擎下载、安装或启动功能。服务配置和真实联调工具随插件提供，安装与更新由部署方执行，见 [Linux 整合指南](../../integrations/maibot_sakuratts/deploy/linux/README.md)。

日常回复使用 `ReplyExtension`：Planner 选择是否启用及是否为用户明确要求，Replyer 根据同一人物卡与资料同时生成中文正文和可选日语台词。扩展在发送前解析、合成并返回消息组，由 Host 统一发送与记录。独立语音 Tool 会让 Planner 绕过 Replyer 编写台词，因此取消该入口，保留手动朗读 Command。语音协议不修改人物卡和角色交接契约。

结构化正文不进入普通文字后处理，最终格式校验放在 Host 会阻止发送的回复扩展边界。合成失败时保留中文，明确请求语音的回复附加失败说明；协议错误不伪装成普通正文。回复不维护额外台词缓存，重试与并发按 Host 的本次回复上下文处理。

隔离回放发现，仅靠提示词无法可靠取得 JSON；追加或替换 Host 的纯文字格式指令仍有失败。插件通过 `maisaka.replyer.before_request` 提供 `response_schema`，配套的最小 Host 补丁将它传入已有 `LLMGenerationOptions.response_format`。该方案依赖模型接口的 JSON Schema 支持，不另建格式修复器，也不增加模型调用。角色提示、交接资料和普通回复保持原有路径。候选包需要与 Host 补丁一起评估，不能按独立插件宣称已经完成部署。

保持请求字段和完整 WAV 响应兼容的引擎版本可独立替换；不兼容接口变更需要配套插件更新。插件版本与引擎版本分别发布。当前源码在主仓库维护，市场发布需另建根目录符合 MaiBot 要求的插件仓库。

代价是一次本机 HTTP 传输与 Base64 消息复制，以及中文随消息组等待合成。插件限制单次音频大小、拒绝并发合成和不完整音频；不添加重试或模型切换逻辑。引擎重启期间合成明确失败，当前方案不承诺无停机升级。

插件独立打包，不进入引擎的 wheel 或源码包。配置、部署、更新和验证方法集中在[插件说明](https://github.com/Rvosy/SakuraTTS/blob/main/integrations/maibot_sakuratts/README.md)。
