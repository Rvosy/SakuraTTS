# 模型目录

模型目录包含 `model.json`、`gpt/`、`acoustic/` 和 `frontend/`。`references/` 为可选参考缓存，允许模型完全不带参考条件。旧格式示例见 [model.example.json](../examples/model.example.json)。

`format` 固定为 `sakuratts-model-v1`；`name` 是显示名称；当前 `languages` 必须为 `["ja"]`，`backend.preferred` 只接受 `cuda`。

`gpt`、`acoustic`、`frontend` 和 `references` 的值是相对模型目录的资源子目录。引用必须解析到模型目录内部；绝对路径、越界路径和指向外部的链接会被拒绝。显式提供 `default_reference` 时，它必须在 `references` 中。HTTP 不使用该默认值，按原版请求中的参考音频和文本确定条件。

`acoustic_python` 指向声学和经典前端共用的私有解释器；`main_dictionary` 是可选的 pyopenjtalk-plus 主字典路径。这两项属于安装环境，允许引用模型外部位置；移动到其他机器后需要重新配置。

内部包继续保留原 manifest、文件哈希、官方源码身份与检查点身份。顶层描述只组织加载位置，不替代内部校验。默认使用 FP32；实验声学 FP16 仍需要显式启用。

`sakuratts convert --config OLD --output NEW` 复制旧包并校验输入和输出，目标目录必须不存在。旧 `sakuratts-windows-config-v1` 继续可读；其中外部包路径保留原语义。
