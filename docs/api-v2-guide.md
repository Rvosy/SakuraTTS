# API V2 使用说明

SakuraTTS 提供 GPT-SoVITS API v2 的 GET / POST 合成接口。Windows 的 CPU、CUDA、DirectML 与 Apple silicon 的 MLX 使用同一 HTTP 入口。Windows 日英包支持日语、英语和日英混读，中日英包另支持普通话。Windows 整合包使用 V2ProPlus，Apple MLX 支持 V2Pro 和 V2ProPlus 的日文推理。各设备和模型的实测范围见[兼容矩阵](specs/compatibility-matrix.md)与 [Apple 指南](apple.md)。

这份文档用于客户端接入；部署细节见[快速开始](quickstart.md)，接口实现与生命周期见 [HTTP 服务配置](http-api.md)。对照版本固定为 GPT-SoVITS [`48b1a016`](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/api_v2.py)。

## 启动服务

使用 [Windows 完整整合包](portable-bundle.md)时，将对应后端的配置模板复制为 `configs/tts_infer.yaml`，填写 `custom.t2s_weights_path`（GPT `.ckpt`）和 `custom.vits_weights_path`（SoVITS `.pth`）。双击 `start-server.bat`，或在终端指定配置文件。

```powershell
.\start-server.bat -c configs/tts_infer.yaml
```

[Mac 完整包](apple.md#整合包)使用 `start-server.command`。Windows NVIDIA 源码安装先按[准备指南](setup-windows-nvidia.md)安装依赖，再将[源码配置示例](../examples/tts_infer.example.yaml)复制到 `configs/tts_infer.yaml`，填写权重、官方源码和解释器路径。在仓库根目录运行以下命令。

```powershell
python api_v2.py -a 127.0.0.1 -p 9880 -c configs/tts_infer.yaml
```

默认地址是 `http://127.0.0.1:9880`。终端显示“服务就绪”后访问 `/health`，默认 `direct` 模式下 `model_loaded: true` 表示模型已加载。HTTP 能连通或 `status: "ready"` 只表示服务可以接收请求。浏览器访问 `/docs` 可查看请求字段。不传配置路径时，服务依次查找 `configs/tts_infer.yaml` 和 `GPT_SoVITS/configs/tts_infer.yaml`。两者都缺失时，服务以未加载模型的状态启动。

转换原始权重和编码新参考录音需要准备组件。完整包自带独立解释器、所需官方源码和公共辅助资源，用户提供发声模型和参考录音。精简包可使用已有转换模型和匹配的参考条件，也可补装匹配的准备组件。导入原版 YAML 时，若 `is_half: true`，须另用 `--profile` 或 `sakuratts.profile` 选择 SakuraTTS 档位。[推理档位](inference-profiles.md)

请求中的模型和音频路径都是**服务所在机器上的路径**。它们不是文件上传字段；建议传绝对路径。默认 `direct` 模式在启动时加载模型，第一次请求还可能需要编译内核或准备参考条件。

## 当前支持哪些调用

| 方法与路径 | 当前支持情况 | 使用限制 |
| --- | --- | --- |
| `POST /tts` | JSON 请求，返回音频 | 当前支持的语言、采样与输出参数见下表 |
| `GET /tts` | 查询参数，返回音频 | 与 POST 共用字段、默认值和校验 |
| `GET /set_gpt_weights?weights_path=...` | 切换 GPT 权重 | 支持范围内的原始 `.ckpt` 或转换后的 GPT 目录；先配置初始模型与准备环境 |
| `GET /set_sovits_weights?weights_path=...` | 切换 SoVITS 权重 | 支持范围内的原始 `.pth` 或转换后的声学目录，须匹配当前后端和档位 |
| `GET /set_refer_audio?refer_audio_path=...` | 预先准备参考音频条件 | `/tts` 仍须传参考路径、转写和语言 |
| `GET /control?command=exit` | 结束独立服务 | 等待资源清理；嵌入 ASGI 时需宿主提供控制回调 |
| `GET /control?command=restart` | 重启服务状态 | 恢复启动配置，运行中切换的权重不会写回配置；PID 不保证变化 |

`GET /health` 查询服务与模型状态，`GET /models` 返回当前模型信息，无模型时返回 `[]`。`GET /docs` 和 `GET /openapi.json` 提供接口定义。需要空闲休眠、提前唤醒时，可选用[后台控制模式](http-api.md#可选后台控制模式)。

仓库保留的 `api.py` 启动脚本也启动这套 V2 服务。旧协议的 `/`、`/set_model`、`/change_refer` 返回 404，不能因为启动文件同名就沿用旧客户端协议。

## 发起一次合成

下面的示例只使用 Python 标准库。替换参考路径和对应的真实日文转写后执行：

```python
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

payload = {
    "text": "こんにちは。今日はいい天気ですね。",
    "text_lang": "ja",
    "ref_audio_path": "D:/Voices/reference.wav",
    "prompt_text": "参考音声です。",  # 替换为参考音频中的实际内容
    "prompt_lang": "ja",
    "parallel_infer": False,
    "streaming_mode": False,
    "media_type": "wav",
}
request = Request(
    "http://127.0.0.1:9880/tts",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urlopen(request) as response:
        audio = response.read()
        status = response.headers.get("X-SakuraTTS-Status")
    Path("hello.wav").write_bytes(audio)
    print("hello.wav", status)
except HTTPError as error:
    print(error.code, error.read().decode("utf-8"))
```

GET 调用传同一组参数，用 `urlencode` 处理日文和路径中的特殊字符：

```python
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

params = {
    "text": "こんにちは。",
    "text_lang": "ja",
    "ref_audio_path": "D:/Voices/reference.wav",
    "prompt_text": "参考音声です。",  # 替换为参考音频中的实际内容
    "prompt_lang": "ja",
    "parallel_infer": "false",
    "streaming_mode": "false",
    "media_type": "wav",
}
with urlopen("http://127.0.0.1:9880/tts?" + urlencode(params)) as response:
    Path("hello.wav").write_bytes(response.read())
```

一次只处理一条推理、权重切换或参考准备操作；冲突请求返回 409。非流式成功响应的 `X-SakuraTTS-Status` 若为 `stopped_at_limit`，表示触发生成上限，不能把这段音频当作全文已读完。

## `/tts` 参数支持表

字段与默认值由 [server.SpeechRequest](../sakuratts/server.py) 定义，固定上游对照保存在 [API 快照](../tests/fixtures/gpt_sovits_api_v2.json)。下表说明各字段的使用方式；GET 使用查询字符串，POST 使用 JSON。

### 文本、语言与参考

| 字段 | 默认值 | 当前行为 |
| --- | --- | --- |
| `text` | `null` | 必须传非空目标文本 |
| `text_lang` | `null` | `ja`、`all_ja`、`en`、`zh`、`all_zh` 或 `auto`。处理中文或英文的段落需要对应前端资源 |
| `ref_audio_path` | `null` | 必须传服务端可读的参考音频路径 |
| `aux_ref_audio_paths` | `null` | 多参考未实现；`null` / 空数组可用，非空数组返回 400 |
| `prompt_lang` | `null` | 同上，独立于目标语言生效 |
| `prompt_text` | `""` | 当前必须传参考音频的非空转写；无转写推理返回 400 |

每次请求可以单独选择目标语言和参考录音的语言。`auto` 使用原版语言检测分段，遇到当前模型前端缺少的处理器会返回 HTTP 400。普通话用 `zh` 或 `all_zh`，韩语和粤语没有专用模式。参见[普通话请求示例](chinese-frontend.md)与[日英混合处理](english-frontend.md)。

### 分句与采样

| 字段 | 默认值 | 当前行为 |
| --- | --- | --- |
| `text_split_method` | `"cut5"` | 支持 `cut0`～`cut5`；其他值返回 400 |
| `top_k` | `15` | 已接入采样，必须 ≥ 1 |
| `top_p` | `1` | 核采样阈值，范围为 `(0, 1]`。值越小，可供采样的候选越少；`1` 保留全部候选，再按温度和 `top_k` 采样 |
| `temperature` | `1` | 已接入采样，必须为有限正数 |
| `repetition_penalty` | `1.35` | 已接入重复惩罚，必须为有限正数 |
| `seed` | `-1` | `-1` 随机；非负整数固定种子。同 seed 不保证与原版 Torch 输出相同 |
| `batch_size` | `1` | 当前只支持单项批次，其他值返回 400 |
| `batch_threshold` | `0.75` | 保留字段；当前单项批次中不改变批次划分 |
| `split_bucket` | `true` | 非流式按文本长度排序推理后恢复音频顺序；按句流式自动关闭 |
| `parallel_infer` | `true` | 接受；当前单项批次按顺序执行 |

### 音频与流式

| 字段 | 默认值 | 当前行为 |
| --- | --- | --- |
| `speed_factor` | `1` | 非流式支持 `0.5`–`100`；非 `1` 值需要 FFmpeg，以保音高方式调整整段输出速度。流式只支持 `1` |
| `fragment_interval` | `0.3` | 每片尾部的静音秒数，必须 ≥ 0 |
| `media_type` | `"wav"` | 支持 `wav` / `raw`；`ogg` / `aac` 需 FFmpeg 分别提供 `libvorbis` / `aac` 编码器 |
| `streaming_mode` | `false` | `0` / `false` 返回完整音频；`1` / `true` 按句返回；`2` / `3` 返回 400 |
| `sample_steps` | `32` | 保留字段；当前 V2ProPlus 模型不使用它 |
| `super_sampling` | `false` | 超采样未实现，`true` 返回 400 |
| `overlap_length` | `2` | 保留字段；当前模式 0 / 1 不使用它 |
| `min_chunk_length` | `16` | 保留字段；当前模式 0 / 1 不使用它 |

`batch_threshold`、`sample_steps`、`overlap_length` 和 `min_chunk_length` 可随原版请求一起传入，但修改它们不会启用新的能力。表外字段会被忽略。

`speed_factor=0.9` 轻微放慢，整段时长约为原来的 `1/0.9`。调速使用 FFmpeg 的 [atempo](https://ffmpeg.org/ffmpeg-filters.html#atempo)，在模型生成后处理整段音频，包括片尾静音。超过 `2` 倍时分级处理。`1` 保留原始 PCM。完整响应的 `X-SakuraTTS-Speed-Factor` 标明倍率，`X-SakuraTTS-Audio-Seconds` 是调整后的时长。缺少 FFmpeg、超出倍率范围或请求流式调速时，在推理和唤醒前返回 400。

按句流式需要客户端持续读取响应：每句话生成完成才发送这一片，尚不支持语义 Token 流式。WAV 模式先发送空数据长度的 WAV 头，再发送 PCM；直接保存后可能无法作为普通完整 WAV 播放。需要保存完整文件时使用 `streaming_mode=false`。RAW 为单声道 16 位小端 PCM，不包含采样率等文件头信息。

## 切换权重和参考

在服务空闲时依次调用。下面的路径需要替换为服务端实际文件；新的权重必须属于当前支持的模型范围：

```python
import json
from urllib.parse import urlencode
from urllib.request import urlopen

base = "http://127.0.0.1:9880"

def call(path, **params):
    with urlopen(base + path + "?" + urlencode(params)) as response:
        print(json.load(response))

call("/set_gpt_weights", weights_path="D:/Voices/character.ckpt")
call("/set_sovits_weights", weights_path="D:/Voices/character.pth")
call("/set_refer_audio", refer_audio_path="D:/Voices/character.wav")
```

成功返回 `{"message": "success"}`。两次权重切换是独立操作，应在都成功后再合成。`/set_refer_audio` 的参数名是 `refer_audio_path`；`/tts` 使用 `ref_audio_path`，并继续传入对应的 `prompt_text` 和 `prompt_lang`。

切换只影响当前服务会话。固定使用另一套权重时，编辑 YAML 后重启。原始权重切换需要准备组件。参考条件可缓存；修改参考转写或语言后会重新计算文本特征。

## 出错时如何判断

| 结果 | 含义与处理 |
| --- | --- |
| `400`，含 `error: "unsupported_feature"` | 请求使用未实现的计算能力；按支持表修改请求 |
| 其他 `400` | 缺少必填项、参数值非法、文件或资源准备失败、合成或编码失败；读取 `message`，有 `Exception` 时一并保留 |
| `409` | 正在推理、准备参考或切换权重；等待当前操作结束后再发请求 |
| `422` | 字段类型不正确，例如 `seed="abc"`；检查 JSON / 查询参数及响应中的 `detail` |
| `404` / `405` | 路径不存在或 HTTP 方法不支持；确认使用上表中的 V2 路由和方法 |
| 流式连接中断 | 首片之前失败可返回 JSON；开始发送音频后失败只能中断连接，客户端应将已收片段标为不完整 |

参数未实现通常可直接根据 400 修正调用；无法连接服务则应先查看启动终端和 `logs/sakuratts.log`。模型或配置加载失败可能使服务无法启动，不属于 HTTP 参数错误。

## 验证范围

设备、模型、数值失败和待验收项集中在[模型与功能验证矩阵](specs/compatibility-matrix.md)。接口与编排测试、CUDA 实测、ASR 和人工听音分别记录结果。
