# SakuraTTS Windows / NVIDIA 推理运行包

本包用于桌宠等本地应用，默认没有角色模型、官方发声底模、参考音频或个人配置。模型目录为空。当前是本机验证的预览版，尚未通过其他显卡的兼容验收。

解压到只含英文、数字和空格的路径，例如 `D:/SakuraTTS`。这版 CuPy/NVRTC 编译仍有非 ASCII 头文件路径限制；模型和参考音频路径可以含中文。系统需要兼容的 NVIDIA 驱动，无需系统 Python 或 CUDA Toolkit。

双击 `check-runtime.bat` 检查实际 GPU 运算；结果保存在 `cache/runtime-check.json`。双击 `start-server.bat` 启动 HTTP 服务，默认 `127.0.0.1:9880`；无模型时服务保持未配置状态。

已有 SakuraTTS 转换结果时：

```bat
sakuratts.bat doctor D:/MyModels/example
sakuratts.bat tts D:/MyModels/example --text "こんにちは。" --output hello.wav
start-server.bat D:/MyModels/example
```

运行包自动使用自己的声学解释器，不沿用模型中开发机的解释器路径。Python/CLI 合成需要模型已有参考条件；HTTP 仍按原版请求提供参考音频和转写，已准备的匹配参考可以复用。

`configs/` 附带四个推理档位。桌宠可先试 `--experimental configs/low-vram.json`，需要匹配的 FP16 chunk256 声学包；仅切换配置不会转换模型。`minimum-vram.json` 显存更低，但会增加重载等待，目前只支持 CLI/Python。FP16 听感验收未完成，不传配置仍使用 FP32。

为控制体积，主运行包不含 PyTorch、原版转换源码、HuBERT 或说话人编码器。原始 `.ckpt/.pth` 转换和未准备过的参考音频需要另行配置准备组件；只有主运行包时不能执行这两项工作。`configs/tts_infer.example.yaml` 是供准备组件使用的模板，不会自动加载或下载任何模型。

本包不自动下载依赖或模型。用户自行放入的模型、配置、缓存和日志不属于发行内容。压缩包清单与文件校验在 `bundle-manifest.json`，各依赖许可保留在对应 `.dist-info`、运行组件目录和 `licenses/` 中。
