# HTTP API 与原版兼容范围

接口以本次对照的 GPT-SoVITS `api_v2.py` 为基准。当前是原生后端的部分兼容实现，支持 Windows / NVIDIA、V2ProPlus 和日文；尚未实现的能力返回明确错误，不静默替换参数。

## 启动

复制 [配置示例](../examples/tts_infer.example.yaml)，填写模型、转换解释器和源码路径：

```powershell
start-server.bat -c configs/tts_infer.yaml
# 同等入口
python api.py -a 127.0.0.1 -p 9880 -c configs/tts_infer.yaml
```

路径按启动工作目录解析，和原版一样；建议使用绝对路径。脚本固定从仓库目录启动。配置提供 `custom.t2s_weights_path` / `vits_weights_path` 时，先转换或复用部署缓存，再加载权重。`sakuratts.model` 可引用已有部署包以跳过完整转换。原始模型和原版配置文件不会被修改。

无参数时仅检查 `configs/tts_infer.yaml`，不会自动发现角色包。没有配置时 HTTP 可以启动，但 `model_loaded=false`，合成会提示配置模型。HTTP 加载 GPU 权重；第一次合成仍可能有内核编译等开销。低级 `Engine.load` 保留延迟加载，实验性的 `staged` 模式只用于低级接口。

## 原版调用方式

| 方法与路径 | 行为 |
| --- | --- |
| `GET/POST /tts` | 返回音频；POST 为 JSON，GET 为查询参数 |
| `GET /set_gpt_weights?weights_path=...` | 更换 GPT，可传原始 `.ckpt` 或转换后的 GPT 目录 |
| `GET /set_sovits_weights?weights_path=...` | 更换 SoVITS，可传原始 `.pth` 或转换后的声学目录 |
| `GET /set_refer_audio?refer_audio_path=...` | 预先准备参考音频的条件；字段拼写沿用原版 |
| `GET /control?command=exit` | 结束独立服务，等待工作线程清理 |
| `GET /control?command=restart` | 清理后按启动配置重新加载；进程 PID 不保证变化 |
| `GET /health`、`GET /models` | 附加诊断接口，查询忙碌和模型状态 |

嵌入 ASGI 应用时，进程控制需要宿主提供回调；默认不退出宿主进程。权重切换不会保存到启动配置，重启恢复配置中的路径。初始模型组合通过配置提供，权重切换接口不承担资源安装。

```json
{
  "text": "こんにちは。今日はいい天気ですね。",
  "text_lang": "ja",
  "ref_audio_path": "D:/Voices/reference.wav",
  "prompt_text": "参考音声です。",
  "prompt_lang": "ja",
  "text_split_method": "cut5",
  "seed": -1,
  "media_type": "wav",
  "streaming_mode": false
}
```

与本次对照的原版 API 一样，`/tts` 仍要求 `ref_audio_path`，调用 `/set_refer_audio` 不会让该字段变成可选。服务没有角色、情绪或参考 ID 的注册前置步骤。已有部署包中的参考文件只作为匹配音频内容和模型身份的缓存，不作为默认选择。

## 参数与已知差异

| 能力 | 当前行为 |
| --- | --- |
| 字段和默认值 | 使用原版字段；`seed=-1`、`text_split_method=cut5`、`parallel_infer=true`、`batch_size=1` |
| 分句 | `cut0` 至 `cut5`，沿用现有原版派生文本处理 |
| 种子 | `-1` 生成随机种子；非负值可复用；NumPy 与 Torch 的同 seed 不保证相同序列 |
| 采样 | Top-k、温度、重复惩罚；`top_p` 当前仅接受 `1`，此前数值边界问题尚未解决 |
| 参考 | 单个原始音频及非空转写，`ja` / `all_ja`；新增音频由独立准备进程编码 |
| 间隔 | `fragment_interval` 控制每片尾部静音，默认 `0.3` |
| 音频 | WAV、RAW；OGG / AAC 需要 PATH 中的 FFmpeg，压缩码流不承诺逐字节等同原版 |
| `streaming_mode=0/false` | 整条请求成功后返回完整音频 |
| `streaming_mode=1/true` | 每片合成完成后立即返回；WAV 先发原版形式的空数据头，后续为 PCM |
| `streaming_mode=2/3` | 尚未实现语义 token 流式，返回 400 |
| 批处理 | 当前 `batch_size=1`；并行及分桶参数在单项批次中没有额外计算效果，不代表已经实现批量并行 |
| 语速、多参考、无转写、超采样 | 尚未实现，明确返回 400 |
| 模型与语言 | Windows V2ProPlus 日文路径；其他版本、中文等仍需实现或验证 |
| 精度 | 默认 FP32；原版 `is_half=true` 尚未对齐，不能直接映射为某个实验 FP16 开关 |

`sample_steps` 对 V2ProPlus 不适用，`overlap_length` / `min_chunk_length` 对模式 0/1 不适用，保留原版字段。参数非法或功能未实现返回 400，Pydantic 类型错误返回 422。合成失败使用原版的 `message: tts failed` / `Exception` 结构。

## 生命周期与输出

终端用一条横线分隔请求，依次显示请求参数、参考音频、输入文本、推理过程和一行完成摘要。文本单独缩进，按终端宽度换行。推理过程保留原版的“提取文本Bert特征”“预测语义Token”“合成音频”阶段名称，下面缩进显示片段与音素数量、进度、停止状态及耗时；日语沿用原版的零 BERT 特征，不加载中文 BERT 模型。多片段请求显示 `[1/2]` 等片段编号，单片段省略编号。模型文件名只在启动和切换权重时显示；每次请求对应的模型、规范化文本和阶段耗时汇总仍写入详细日志。普通消息省略 `INFO`，成功的 HTTP 访问不重复显示；警告和错误保留明确标记。`seed=-1` 时显示本次实际使用的种子，参考音频可复用时统一显示“已准备”。

终端中的 GPT 行随实际采样更新，保留原版常用的 `it`、`it/s`、`EOS` 等标记。`it` 是采样迭代次数，包含最后预测 EOS 的一次，不等同于交给 SoVITS 的有效 Token 数。结束时输出 `T2S Decoding EOS [157 -> 410]` 这样的记录：左侧是参考语义前缀长度，右侧按原版 batch 日志计算，包含参考前缀及所有采样迭代，尚未剔除最后的 EOS。数字来自当前片段。仅因长度限制停止时显示 `STOP` 和警告，不冒充 EOS；EOS 与上限同时出现时保留两种状态。进度不以保护上限计算百分比或预计剩余时间。非终端输出每秒最多记录一次中间进度。取消或失败会关闭进度显示并标记状态，音频编码失败不会打印完成摘要。

完整诊断默认保存到 `logs/sakuratts.log`，包含时间、级别、请求编号、完整路径和模型哈希、输入与规范化文本、参考转写、实际参数、缓存来源、细分停止原因和异常堆栈。单文件达到 10 MiB 后轮转，最多保留 3 份备份。准备进程的原始输出也写入该文件；已知的 `enc_q` 训练模块缺失项不在终端展开，其他未匹配权重和警告仍显示。

`start-server.bat --log-level debug` 可在终端展开排错细节，`--log-file PATH` 可指定诊断文件。级别只控制终端显示，文件始终保留完整诊断；`warning` / `error` 级别不会显示动态进度。多个服务实例应指定不同日志文件。

总耗时包含参考准备，输出 RTF 按包含句间静音的音频时长计算。非流式总耗时不含音频压缩和网络发送；流式总耗时包含片段编码及等待发送队列的时间。日志包含输入文本和本地路径。嵌入 ASGI 时由宿主配置 `sakuratts` 日志级别为 `INFO` 以显示内部进度；低级 SDK 默认保持安静。

一个专用线程负责模型创建、推理、切换和关闭。正在计算或切换时，冲突操作返回 409；这是当前与原版不同的并发约束。模型格式检查失败时保留现有实例；旧 GPU 权重释放后发生的加载失败会使服务处于未加载状态，不能承诺自动恢复旧模型。

完整响应的 `X-SakuraTTS-Status` 区分 `completed` 和 `stopped_at_limit`。`X-SakuraTTS-Request-Ms` 包含本次参考解析和原生推理，音频压缩与网络发送不在其中。达到长度限制不表示内容完整。

流式队列最多保留两个片段，发送慢时阻塞生产。客户端断开后在计算边界取消，工作线程退出前仍保持忙碌。第一片之前失败可以返回 JSON 错误；已开始发送后失败会中断连接，不补成成功音频。模型加载、显存释放、句间 RNG 顺序沿用原生实现。

首次原始参考编码需要配置中的原始检查点、HuBERT / 说话人模型与准备环境；缓存命中后的普通推理不需要 PyTorch。缓存保存音频条件，参考文本特征按当前文本、语言和前端重新计算，修改转写不会误用上一次文本。
