# MaiBot Host 补丁

[response-schema.patch](response-schema.patch) 面向当前 ECS 镜像 `maibot-sakura-support:1.3.14-memory-attribution` 中的 `src/chat/replyer/maisaka_generator_base.py`。

补丁将 `maisaka.replyer.before_request` Hook 返回的 `response_schema` 传给 MaiBot 已有的 `RespFormat(JSON_SCHEMA)`，再交给当前 Replyer 模型。未提供 schema 时，原有生成选项保持不变；无效 schema 或模型接口拒绝请求时，沿用 Host 的生成失败路径。

插件提供的是 MaiBot 的 schema 包装对象，包含 `name`、`strict` 和对象级 `schema`。不新增角色配置、模型别名、二次翻译或自动重试。普通回复也不会被强制改成 JSON。

补丁已固化到 `maibot-sakuratts:1.3.14-integration-20261026` 镜像，并在 ECS 部署测试。构建前从基础镜像提取原文件检查补丁，应用结果与构建输入逐字节一致。部署通过 Compose 切换镜像，不覆盖运行中的容器文件，也不修改角色卡。后续升级仍需在对应 MaiBot 源码中检查补丁，再构建镜像验证。

在匹配的 MaiBot 源码副本中，可用 `git apply --check response-schema.patch` 检查上下文是否匹配。补丁随目标宿主源码维护，不能保证适用于任意 MaiBot 版本。回放工具的 `--validate-only` 会调用真实 `generate_reply_with_context`，以模型替身检查普通回复、auto 和 requested 的生成参数，且不向聊天发送消息。

## SQLite 忙碌时保存消息

[message-storage-busy.patch](message-storage-busy.patch) 面向 `maibot-sakuratts:1.3.14-memory-continuity-20261027` 的 `src/common/utils/utils_message.py`。它处理消息写入与其他数据库写入方竞争时的 `SQLITE_BUSY`，包括语音已送达、历史保存失败的情况。

现有事务边界先回滚并关闭失败的会话，再重做整笔保存，最多尝试三次，两次等待分别为 0.1 秒和 0.2 秒。沿用数据库每次操作的 busy timeout，不重新发送平台消息。最后一次仍忙碌时保留原始异常；非忙碌错误不重试。补丁不回填过去保存失败的历史，也不解决任意长时间占用数据库的问题。

在匹配的源码副本上先执行 `git apply --check message-storage-busy.patch`，再应用并构建镜像。隔离测试入口为 [test_message_storage.py](test_message_storage.py)，需要兼容 MaiBot 的依赖环境、`PYTHONPATH=/MaiMBot`，以及只读挂载到 `/source-config` 的配置副本。测试将配置复制到隔离容器，数据库位于临时目录，不挂载生产数据，不发送聊天消息。

验证覆盖短时写锁恢复、保存后只存在一条带平台消息 ID 和原文的记录、等待期间事件循环响应、持续锁定不伪报成功，以及其他数据库错误不重试。生产切换与回退沿用 [Linux 整合指南](../deploy/linux/README.md#更新与回退) 的版本化镜像流程。
