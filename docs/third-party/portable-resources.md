# 整合包辅助资源

整合包不包含角色发声权重和参考音频。准备组件使用以下公共分析模型，包内保存上游声明。

| 资源 | 使用来源与原始项目 | 许可材料 |
|---|---|---|
| chinese-hubert-base | GPT-SoVITS 模型仓库提供的权重；原始项目为 TencentGameMate/chinese-hubert-base | GPT-SoVITS 模型卡声明 MIT；原始模型卡声明 MIT，见 `HuBERT-model-card.md` |
| ERes2NetV2 w24s4ep4 | GPT-SoVITS 模型仓库提供的权重；原始项目为 iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common | 分发模型卡声明 MIT；原始模型卡声明 Apache License 2.0，保留 `ERes2NetV2-model-card.md` 和 `Apache-2.0.txt` |
| fastText lid.176.bin | Facebook AI Research 语言识别模型 | CC-BY-SA 3.0，原始署名与许可位于准备组件 `licenses/` |

使用文件的版本和来源地址见 `runtime/preparation/auxiliary-model-sources.json`。GPT-SoVITS 的转换权重与原始项目文件尺寸不同，不能将原始文件摘要作为分发文件身份。原始作者与上游分发者的声明均保留，不把整个整合包统一标为 MIT。

来源：

- https://huggingface.co/lj1995/GPT-SoVITS/tree/336b2ec4e8d4ac74740798dd40af44e74659ecaf
- https://huggingface.co/TencentGameMate/chinese-hubert-base/tree/fce0375452b1dd6c080ac3248d423d4d037bc831
- https://github.com/TencentGameMate/chinese_speech_pretrain
- https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common

CI 构建的 FFmpeg 7.1.2 启用内置音频解码、PCM、AAC 和 libvorbis Ogg 输出，不启用 GPL 组件；Ogg 与 Vorbis 依赖均以静态库构建，对应源码一并收录。其完整源码归档和配置脚本放在 `licenses/ffmpeg/`；实际编译选项及许可见 `licenses/FFmpeg-build-and-license.txt`。旧本地试验包的 FFmpeg 来源与编译选项不同，不使用本说明替代其实际分发材料。

NVIDIA 运行库遵循对应厂商许可，详见随包 NVIDIA wheel 的许可文件及声学工作进程 `licenses/`。它们不是项目 MIT 许可的一部分。
