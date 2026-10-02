# 模型目录

模型目录包含 `model.json`、`gpt/`、`acoustic/` 和 `frontend/`。`references/` 为可选参考缓存，允许模型完全不带参考条件。格式示例见 [model.example.json](../examples/model.example.json)，字段和路径检查由 [Model.load](../sakuratts/model.py) 定义。

`format` 固定为 `sakuratts-model-v1`；`name` 是显示名称；`languages` 是非空语言名称数组；`backend.preferred` 是建议使用的后端名称，省略时为 `cuda`。当前后端包括 `cuda`、`cpu`、Windows `directml` 和实验性 Apple `mlx`，具体模型限制见[兼容矩阵](specs/compatibility-matrix.md)；日文资源包声明 `["ja"]`，加入[英文资源](english-frontend.md)后可声明 `["ja", "en"]`。

元数据可以描述未来的语言或后端。`Model.load` 检查字段和资源路径，`Model.info()` 返回这些声明；执行支持由 `Engine.load` 的后端选择与前端装配判断。`sakuratts capabilities` 列出已实现能力。修改声明不会转换权重或增加新的语言资源，尚未实现的组合会明确报错。

`gpt`、`acoustic`、`frontend` 和 `references` 可以使用相对模型目录的路径、绝对路径或指向共享资源的链接。显式提供 `default_reference` 时，它必须在 `references` 中。HTTP 按请求中的参考音频和文本确定条件。

`acoustic_python` 指向 CUDA 声学工作进程的私有解释器；CPU / DirectML / MLX 声学使用主解释器。`frontend_python` 指向经典日文 G2P 的私有解释器，旧配置省略时仍沿用 `acoustic_python`。两项也可以显式指向同一个解释器。`main_dictionary` 是可选的 pyopenjtalk-plus 主字典路径。这些字段属于安装环境，允许引用模型外部位置；源码安装搬迁到其他机器后需要重新配置。

读取 `Model` 本身不会按当前整合包改写这些字段。执行时由整合包 `runtime/portable.json` 中的工作进程角色绑定当前位置，声学和前端角色可以共享一份 Python。模型导出、信息查询不依赖整合包是否已安装，也不会把开发机的旧路径当成新安装目录。

Python 可用 `Engine.load(model, backend="cuda")` 覆盖建议后端，CLI 使用 `--backend cuda`。HTTP 服务的选择顺序是显式 `backend` 参数、配置中的 `sakuratts.backend`、原版 `custom.device`、模型建议值；均未指定时使用 `cuda`。模型文件本身不会被覆盖。

内部包保留原 manifest、文件哈希与来源记录，供追溯和缓存使用；加载不比对哈希，也不要求来源提交一致。文件能否使用由解析结果、张量结构和实际执行后端判断。CPU 默认选择 INT8 GPT 与 FP32 声学，DirectML 默认选择 FP16 GPT 与全图 FP16 声学；CUDA 与 MLX 的默认精度不变。选择设备不会自动转换资源。

`sakuratts convert --config OLD --output NEW` 把旧包复制到临时目录，保留后端建议值，校验待发布结果后再发布，目标目录必须不存在。旧 `sakuratts-windows-config-v1` 继续可读；其中外部包路径保留原语义。原始检查点转换使用 `--backend cpu` 或 `--backend directml` 准备对应公开档位的完整资源，并写入建议后端；CPU 与 AMD 分别发布模型目录。准备环境和缓存用法见 [CPU / DirectML 指南](cpu-amd.md)。
