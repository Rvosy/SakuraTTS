# SakuraTTS Windows / NVIDIA 整合包

本包用于桌宠等本地应用，默认没有角色模型、官方发声底模、参考音频或个人配置。模型目录为空。当前是本机验证的预览版，尚未通过其他显卡的兼容验收。

解压到只含英文、数字和空格的路径，例如 `D:/SakuraTTS`。这版 CuPy/NVRTC 编译仍有非 ASCII 头文件路径限制；模型和参考音频路径可以含中文。系统需要兼容的 NVIDIA 驱动，无需系统 Python 或 CUDA Toolkit。

双击 `check-runtime.bat` 检查实际 GPU 运算；结果保存在 `cache/runtime-check.json`。包含 `start-server.bat` 的包可双击启动 HTTP 服务，默认 `127.0.0.1:9880`；无模型时服务保持未配置状态。SDK/CLI 包省去了 HTTP 依赖和这个启动脚本，使用下文的 `sakuratts.bat` 命令。

完整包带有 `runtime/preparation/`。使用完整 HTTP 包时，将自己的 GPT `.ckpt` 和 SoVITS `.pth` 放进 `models/`，复制 `configs/tts_infer.example.yaml` 为 `configs/tts_infer.yaml`，填写两份权重路径。双击 `start-server.bat` 即可自动转换并启动；第一次会显示前端准备、GPT 转换和 SoVITS 转换进度，后续启动复用缓存。

接口只兼容 GPT-SoVITS API v2，未实现的 V2 功能返回 400。传入自己的参考音频和转写，并显式设置 `parallel_infer=false`（省略或传 `true` 会返回 400）：

```powershell
curl.exe -X POST http://127.0.0.1:9880/tts -H "Content-Type: application/json" --data-raw '{"text":"こんにちは。","text_lang":"ja","ref_audio_path":"D:/Voices/reference.wav","prompt_text":"参考音声です。","prompt_lang":"ja","parallel_infer":false}' --output hello.wav
```

完整支持表、GET / POST 与错误处理示例见 [API v2 使用指南](https://github.com/Rvosy/SakuraTTS/blob/main/docs/api-v2-guide.md)。

第一次使用该参考音频时会自动提取特征，之后复用。参考转写必须与录音一致，当前支持范围是 V2ProPlus、日文。准备过程使用独立 CPU 进程，完成后退出；准备组件只带 CPU 版 PyTorch，主推理服务不加载它。首次准备会占用额外主存和磁盘，转换结果和参考条件保存在包内缓存中。

已有 SakuraTTS 转换结果时也可以直接使用：

```bat
sakuratts.bat doctor D:/MyModels/example
sakuratts.bat tts D:/MyModels/example --text "こんにちは。" --output hello.wav
start-server.bat D:/MyModels/example
```

运行包自动使用自己的声学和前端解释器，不沿用模型中开发机的解释器路径。Python/CLI 合成需要模型已有参考条件；HTTP 仍按原版请求提供参考音频和转写，已准备的匹配参考可以复用。

桌宠可在默认 FP32 基础上加 `--runtime-mode managed`，空闲时退出推理进程树；调用大模型 API 时可先请求 `POST /runtime/wake`。首次原始模型转换也计入唤醒时间，慢机器可加 `--wake-timeout-seconds 900`；默认启动模式仍为 `direct`。

`configs/` 附带四个推理档位。`low-vram.json` 和 `minimum-vram.json` 需要另行准备匹配的 FP16 chunk256 声学包；本次自动转换生成 FP32 资源，仅切换配置不会转换精度。极限档支持 CLI/Python 和 managed HTTP，会增加重载等待。FP16 听感验收未完成，不传配置仍使用 FP32。

如果下载的是不含 `runtime/preparation/` 的精简推理包，原始权重和新参考处理需要补充匹配的准备组件，或者改用完整包。无需填写开发机的 Python、原版源码或辅助模型路径。

本包不自动下载依赖或模型。用户自行放入的模型、配置、缓存和日志不属于发行内容。`runtime/portable.json` 记录发行组合和工作进程路径；压缩包清单与文件校验在 `bundle-manifest.json`，各依赖许可保留在对应 `.dist-info`、运行组件目录和 `licenses/` 中。当前仍只支持 Windows / NVIDIA 日文推理，准备组件里的 CPU PyTorch 不提供 CPU 语音推理。

升级时解压到新目录，再保留自己的 `models/`、配置及参考音频。缓存可以迁移；准备组件或转换逻辑改变时会自动生成新缓存。不要用发行模板覆盖自己的 `configs/tts_infer.yaml`。
