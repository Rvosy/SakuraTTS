# 固定表情发送策略

用于已安装 `rvosy.emoji-send-policy` 的 MaiBot。1.1.0 保留通用表情库的禁用规则，允许 [N.A.V.I. 情绪表情插件](../maibot_navi_expressions/README.md) 的三张原图作为表情包发送。

更新原插件目录中的 `plugin.py` 和 `_manifest.json`，保留原 `config.toml`，再加载插件。不要以另一个目录重复安装相同插件 ID。未安装旧策略的部署无需为了发送表情额外安装本插件；如果也需要全局限制其他表情包，可安装到 `plugins/emoji_send_policy/` 并启用。

发送边界解码实际表情字节并计算 SHA-256，对照 `plugin.py` 中的 `ALLOWED_SHA256`。名称、说明或载荷自带的 hash 不作为放行依据；合并转发中的表情也受检查。普通图片、文本和语音不受此规则影响。更新角色素材时需要同步更换允许的哈希并验证。

在同时包含两个插件目录、已安装对应 MaiBot SDK 的环境中运行：

```shell
python -m unittest discover -s integrations/maibot_emoji_send_policy/tests -v
```
