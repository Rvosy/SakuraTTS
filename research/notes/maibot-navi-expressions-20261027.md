# N.A.V.I. 表情工具部署记录

2026 年 10 月 27 日部署 `rvosy.navi-expressions` 0.1.0。源码和用法见 [插件目录](../../integrations/maibot_navi_expressions/README.md)。

## 行为与参考

模型在调用前看到三个独立工具的神情和适用语境，选择与当前话题及自身回应相符的一张。开心工具说明区分“笑一个”等明确请求和普通夸奖：前者使用 `requested`，后者自主判断是否使用 `auto`。不再提供先调用再搜索素材的通用表情入口。

参考 [AstrBot 表情包小偷](https://github.com/nagatoquin33/astrbot_plugin_stealer) 将情绪匹配、概率与冷却分开的思路，以及 [discord.py 的动态冷却](https://discordpy.readthedocs.io/en/stable/interactions/api.html#discord.app_commands.checks.dynamic_cooldown) 对不同调用条件分别处理的机制。没有复制这些项目的实现或引入依赖。

本次部署采用线性恢复的自主发送概率：距上次成功发送 1 分钟为 5%，3 分钟为 15%，5 分钟及以后为 25%；发送后 30 秒内不自主发送，两次抽取也至少相隔 30 秒。没有发送历史时按 25% 计算。用户主动请求跳过概率和间隔。此曲线是本场景的初始参数，实际规则和可调参数以插件源码为准。

## 素材与验证

附件与 `D:/Project/sakura/characters/N.A.V.I/portraits` 中的三张中文名 PNG 逐字节一致，名称按原目录保留：

| 文件 | SHA-256 |
| --- | --- |
| 无奈认真.png | `31e7e91426afbabc1ca15c8412955f676f7d1041fcf7fd2c6b03fa24a872a089` |
| 意料之外.png | `c8b796b1fae17fb27d75a6c472ce1dbbcf49ba7729aa87bd3bcd1b2ff0e99764` |
| 开心.png | `b93a044aa652e5e8fb68af77b844184f715309d9188a00b7014440f0e8913d4e` |

11 项局部测试通过，覆盖时间加权、上限、并发防重复、重启后延续记录、明确请求绕过限制、失败不伪报成功，以及三个独立工具与原图的对应关系。

使用部署镜像的真实 SDK、Host 发送服务、工具参数转换和历史转换，在断网容器中验证三张图片原始字节、发送成功后历史同步、平台回执 ID，以及发送失败不写历史。现有 `emoji_send_policy` 允许这些普通图片，并继续拦截通用 emoji。平台与数据库边界使用替身，没有向 QQ 发送测试消息或写入线上聊天记录。

另用当前 planner 模型配置尝试六个合成语境的工具选择回放，首个“笑一个”样例即被模型服务的“账号余额不足”错误中断，未取得有效语义选择结果。因此不能将机制测试等同于真实模型在所有语境下都能正确选择模式和表情，群聊体验仍需验收。

## 安装结果

通过新目录安装并由 WebUI 插件启用接口热加载。线上确认三个工具启用，参数 schema 的 `mode` 枚举为 `auto` / `requested`，插件版本为 0.1.0。安装位置 `/opt/maibot-snowluma/plugins/navi_expressions/`。MaiBot 核心容器启动时间未变，TTS 配置、TTS 插件和旧表情策略文件哈希未变。

安装包为 `dist/maibot-navi-expressions-0.1.0.zip`；脚本、包哈希、Host 验证与部署结果保存在本地 `outputs/navi-expressions-20261027/` 和服务器 `/opt/navi-expressions-20261027/`。如需停用，在 WebUI 关闭 `rvosy.navi-expressions` 即可；停用不影响 TTS。

## 0.1.1：表情包格式与回复语境

同日 00:26–00:34 的实际聊天确认，“笑一个”已选中开心素材，但消息仍为普通图片。随后询问可用表情时，回复声称能生成害羞、不满等新图片；询问插画背景时，回复先描述与会长待在一起，再解释没有实体。问题来自工具能力仅向 Planner 提供，Replyer 缺少这三张固定素材的说明。

0.1.1 将发送段改为 `emoji`，保留三张 PNG 原始字节。Replyer 请求加入固定素材范围、自然使用神情、固定背景与现实位置的区分；保留原有角色上下文、工具交接参数和语音输出 schema。发送历史记录简短的表情名称。频率计算与明确请求绕过限制的行为不变。

现有 `rvosy.emoji-send-policy` 同步升级至 1.1.0，按实际图片字节的 SHA-256 只放行三张原图，继续隐藏通用表情库入口并拦截其他 emoji，包括合并转发中的 emoji。源码与安装说明见 [固定表情发送策略](../../integrations/maibot_emoji_send_policy/README.md)。

12 项表情插件测试、5 项发送策略测试通过。断网容器中的真实 Host 验证通过，使用线上 SnowLuma 适配器编码器确认三张素材均输出 `sub_type=1`，内容与原 PNG 一致；发送成功才写历史，失败不伪造成功。验证拦截平台和数据库边界，没有发送 QQ 测试消息。

00:42 已由源码监视器重新加载插件运行时；核心容器未重启，原先加载的插件均恢复，TTS 源码与配置、两个表情插件的原配置保持不变。更新前语音引擎空闲且未加载模型。备份位于 `/opt/maibot-snowluma/backups/navi-expression-stickers-20261027/`。

安装包为 `dist/maibot-navi-expressions-0.1.1.zip` 与 `dist/maibot-emoji-send-policy-1.1.0.zip`，包与逐文件哈希保存在 `outputs/navi-expressions-20261027/sticker-packages.json`；验证和部署结果分别为同目录的 `sticker-host-check.json`、`sticker-deployment.json`。QQ 客户端最终呈现和新提示下的自然对话仍需实际聊天验收。

## 0.2.0：随回复选择表情

用户补充要求：表情应伴随说话，由模型在组织回复时选择是否配表情。0.2.0 使用现有 Host 的 `ReplyExtension`，将选择入口加入 `reply.plugin_options`。Planner 在调用 reply 前确定神情，Replyer 根据同一神情组织正文；不选扩展就只回复正文。没有增加模型调用或新的文本输出格式。

扩展将固定表情追加到整组回复末尾，复用 Host 的发送顺序、失败停止和历史同步。与已有语音扩展一起使用时，先转换中文正文和日语语音，再追加表情，不修改 TTS 的 JSON 契约或源码。概率未命中时只省略表情。发送时间统一由成功回执 Hook 更新，避免把组装附件误算成发送成功。

独立的三个表情工具仅接受用户明确请求的 `requested` 模式；普通夸奖等自主配表情通过回复扩展处理。原有三张素材、时间加权与直接请求绕过限制保持有效。

17 项表情插件测试通过。真实 Host 的回复扩展参数构造、参数校验、优先级执行及发送链路验证通过：省略扩展保持原回复，文字与引用保留，语音在前、QQ 表情包在后，原图字节和历史同步正确。平台、语音合成和数据库边界使用替身，未向 QQ 发送测试消息。

00:50 部署 0.2.0，插件运行时重新加载后，原先加载的插件均恢复。核心容器未重启，TTS 与发送策略源码、原配置哈希不变。备份位于 `/opt/maibot-snowluma/backups/navi-expression-reply-20261027/`。包为 `dist/maibot-navi-expressions-0.2.0.zip`；证据为 `outputs/navi-expressions-20261027/reply-packages.json`、`reply-host-check.json` 和 `reply-deployment.json`。模型是否在自然聊天中恰当地选择配表情，仍需群聊验收。
