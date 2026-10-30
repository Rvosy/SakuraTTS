# 固定表情发送策略

禁用 MaiBot 的通用表情库。Planner 不再看到通用表情工具和回复中的 `attach_emoji` 参数，其他入口触发通用表情选择时也会中止。

更新原插件目录中的 `plugin.py` 和 `_manifest.json`，保留原 `config.toml`，再加载插件。不要以另一个目录重复安装相同插件 ID。需要禁用通用表情库的新部署可安装到 `plugins/emoji_send_policy/` 并启用。

[N.A.V.I. 情绪表情插件](../maibot_navi_expressions/README.md) 负责从自己的三张固定素材中选择并发送表情。本策略只负责通用表情库入口，不检查其他插件或合并转发的图片内容。固定素材仅在角色表情插件中维护。

在已安装对应 MaiBot SDK 的环境中运行：

```shell
python -m unittest discover -s integrations/maibot_emoji_send_policy/tests -v
```
