# SakuraTTS Apple silicon 整合包

解压到可写目录后，运行 `check-runtime.command` 检查 Metal。Python 和推理依赖已在包内，无需安装 Python、Homebrew 或 Git。系统要求见 `bundle-manifest.json` 的 `release.minimum_macos`。

将自己的 V2Pro `.ckpt`、`.pth` 权重放入 `models/`，复制 `configs/tts_infer.example.yaml` 为 `configs/tts_infer.yaml` 并填写权重路径，然后运行 `start-server.command --tts-config configs/tts_infer.yaml`。服务默认监听 `127.0.0.1:9880`，请求格式见项目的 API V2 指南。

完整包带有 `runtime/preparation/`，首次使用会转换原始权重并准备请求中的参考音频。后续请求复用 `cache/`。精简包需要已有的模型目录与参考条件。包内不含角色模型和参考音频。

当前 MLX 路径使用 V2Pro、FP32 与日文前端；V2ProPlus 和 FP16 尚未接入。整体搬迁时保留 `runtime/`、`models/` 和 `cache/`；自定义配置中指向包外的绝对路径需自行更新。

命令行用法：`./sakuratts.command --help`。默认推理不加载 PyTorch；转换和新参考编码由独立准备解释器执行。日志位于 `logs/`，运行检查写入 `cache/runtime-check.json`。

依赖许可、辅助资源来源和 FFmpeg 构建信息在 `licenses/` 与 `runtime/preparation/licenses.json`。代码许可不代替用户模型的分发许可。
