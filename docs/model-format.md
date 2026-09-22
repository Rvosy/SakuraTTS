# 模型目录

模型目录包含 `model.json`、`gpt/`、`acoustic/` 和 `frontend/`。`references/` 为可选参考缓存，允许模型完全不带参考条件。格式示例见 [model.example.json](../examples/model.example.json)，字段和路径检查由 [Model.load](../src/sakuratts/model.py) 定义。

`format` 固定为 `sakuratts-model-v1`；`name` 是显示名称；`languages` 是非空语言名称数组；`backend.preferred` 是建议使用的后端名称，省略时为 `cuda`。当前可执行的公共组合是 `languages: ["ja"]` 与 `backend.preferred: "cuda"`。

元数据可以描述未来的语言或后端。`Model.load` 检查字段和资源路径，`Model.info()` 返回这些声明；执行支持由 `Engine.load` 的后端选择与前端装配判断。`sakuratts capabilities` 列出已实现能力。修改声明不会转换权重或增加新的语言资源，尚未实现的组合会明确报错。

`gpt`、`acoustic`、`frontend` 和 `references` 的值是相对模型目录的资源子目录。引用必须解析到模型目录内部；绝对路径、越界路径和指向外部的链接会被拒绝。显式提供 `default_reference` 时，它必须在 `references` 中。HTTP 不使用该默认值，按原版请求中的参考音频和文本确定条件。

`acoustic_python` 指向声学工作进程的私有解释器；`frontend_python` 指向经典日文 G2P 的私有解释器。旧配置省略后者时沿用 `acoustic_python`，两项也可以显式指向同一个解释器。`main_dictionary` 是可选的 pyopenjtalk-plus 主字典路径。这些字段属于安装环境，允许引用模型外部位置；源码安装搬迁到其他机器后需要重新配置。

读取 `Model` 本身不会按当前整合包改写这些字段。执行时由整合包 `runtime/portable.json` 中的工作进程角色绑定当前位置，声学和前端角色可以共享一份 Python。模型导出、信息查询不依赖整合包是否已安装，也不会把开发机的旧路径当成新安装目录。

Python 可用 `Engine.load(model, backend="cuda")` 覆盖建议后端，CLI 使用 `--backend cuda`。HTTP 服务的选择顺序是显式 `backend` 参数、配置中的 `sakuratts.backend`、原版 `custom.device`、模型建议值；均未指定时使用 `cuda`。模型文件本身不会被覆盖。

内部包继续保留原 manifest、文件哈希、官方源码身份与检查点身份。顶层描述只组织加载位置，不替代内部校验。默认使用 FP32；实验声学 FP16 仍需要显式启用。

`sakuratts convert --config OLD --output NEW` 把旧包复制到临时目录，校验待发布结果后再发布，目标目录必须不存在。旧 `sakuratts-windows-config-v1` 继续可读；其中外部包路径保留原语义。当前转换器仍面向已实现的 Windows CUDA / 日文链路。
