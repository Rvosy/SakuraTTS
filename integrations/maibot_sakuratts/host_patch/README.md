# Replyer 结构化输出候选补丁

[response-schema.patch](response-schema.patch) 面向当前 ECS 镜像 `maibot-sakura-support:1.3.14-memory-attribution` 中的 `src/chat/replyer/maisaka_generator_base.py`。

补丁将 `maisaka.replyer.before_request` Hook 返回的 `response_schema` 传给 MaiBot 已有的 `RespFormat(JSON_SCHEMA)`，再交给当前 Replyer 模型。未提供 schema 时，原有生成选项保持不变；无效 schema 或模型接口拒绝请求时，沿用 Host 的生成失败路径。

插件提供的是 MaiBot 的 schema 包装对象，包含 `name`、`strict` 和对象级 `schema`。不新增角色配置、模型别名、二次翻译或自动重试。普通回复也不会被强制改成 JSON。

补丁已固化到 `maibot-sakuratts:1.3.14-integration-20261026` 镜像，并在 ECS 部署测试。构建前从基础镜像提取原文件检查补丁，应用结果与构建输入逐字节一致。部署通过 Compose 切换镜像，不覆盖运行中的容器文件，也不修改角色卡。后续升级仍需在对应 MaiBot 源码中检查补丁，再构建镜像验证。

在匹配的 MaiBot 源码副本中，可用 `git apply --check response-schema.patch` 检查上下文是否匹配。补丁随目标宿主源码维护，不能保证适用于任意 MaiBot 版本。回放工具的 `--validate-only` 会调用真实 `generate_reply_with_context`，以模型替身检查普通回复、auto 和 requested 的生成参数，且不向聊天发送消息。
