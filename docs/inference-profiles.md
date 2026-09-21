# 推理档位

当前提供四档：FP32、FP16 标准、FP16 低显存、FP16 极限。对应此前实验的 A、C、E、H。FP16 标准是常用推荐候选；不传实验选项时，运行时仍沿用 FP32 默认。FP16 的人工听音验收尚未完成。

| 档位 | 配置文件 | 适用情况 |
|---|---|---|
| A．FP32 | [fp32.json](../examples/fp32.json) | 保留当前 FP32 路径与精度 |
| C．FP16 标准 | [fp16.json](../examples/fp16.json) | 日常使用，兼顾速度与显存 |
| E．FP16 低显存 | [low-vram.json](../examples/low-vram.json) | 更少显存，接受每片段额外加载 |
| H．FP16 极限 | [minimum-vram.json](../examples/minimum-vram.json) | 显存优先，接受约 3 秒的重载等待 |

## 峰值显存

| 档位 | 完整请求观测峰值 |
|---|---:|
| FP32 | 2148.6 MB |
| FP16 标准 | 758.2 MB |
| FP16 低显存 | 584.1 MB |
| FP16 极限 | 465.1 MB |

## 空闲显存

| 档位 | 最后请求后空闲 |
|---|---:|
| FP32 | 1253.1 MB |
| FP16 标准 | 569.4 MB |
| FP16 低显存 | 387.0 MB |
| FP16 极限 | 106.5 MB |

## 请求耗时

| 档位 | 热短句 | 热长句 |
|---|---:|---:|
| FP32 | 0.289 s | 3.290 s |
| FP16 标准 | 0.234 s | 2.677 s |
| FP16 低显存 | 1.009 s | 3.560 s |
| FP16 极限 | 3.055 s | 5.481 s |

数值来自 RTX 5060 8 GB、Windows WDDM、Sakura V2ProPlus 日文单请求，使用准备好的中性参考、`cut0` 分句设置。MB 为十进制单位；显存是同一时刻主进程与声学 worker 的 Dedicated Usage 之和，空闲取请求后约 1 秒的稳定值。耗时为持续采样下的热请求中位数；FP32 与 FP16 来自不同轮次，输出长度也有差异。数据不是任意模型、输入或显卡的保证。详见 [FP32／FP16 对照](research/low-vram-20260921.md)及[声学错峰实测](research/acoustic-session-staging-20260921.md)。

## 选择与使用

A 需要 FP32 声学包。C、E、H 都需要已转换且通过筛查的 FP16 chunk256 声学包；选择档位不会自动转换权重。这些文件是 `--experimental` 的执行选项，模型路径另行传入。

```powershell
sakuratts tts MODEL --experimental examples/fp16.json --text "こんにちは。" --output outputs/hello.wav
```

将 `MODEL` 换成准备好的模型目录或配置文件。选择其他档位时，更换 `--experimental` 后的文件；A 与三个 FP16 档需要各自匹配的声学包。

Python 使用相同文件：

```python
import json
from pathlib import Path
from sakuratts import Engine

options = json.loads(Path("examples/fp16.json").read_text(encoding="utf-8"))
with Engine.load("MODEL", experimental=options) as engine:
    audio = engine.synthesize("こんにちは。")
    audio.save("outputs/hello.wav")
```

A、C、E 可经 `serve MODEL --experimental FILE` 启动 HTTP 服务；H 的总体 `policy="staged"` 当前仍不支持 HTTP，使用 Python Engine 或 `tts` CLI。现有 YAML 的 `is_half` 兼容限制仍需遵循 [HTTP API](http-api.md)。

C 保留模型权重，语义生成完成后释放 GPT 请求状态；E 进一步让 latent／vocoder Session 错峰；H 再让 GPT／声学模型错峰。三个 FP16 档均关闭 Prefill query 分块，在已测对应文本上的音频逐采样一致。`cut5` 是独立的文本分句选项，会改变停顿与生成过程，不再单列为一个档位；E、H 在片段较多时会有更多重载等待。

## 旧配置迁移

- 原 `examples/low-vram.json` 对应 C，现在改用 `examples/fp16.json`；新的 `low-vram.json` 对应 E。
- 原 `examples/minimum-vram.json` 对应旧 F，现在的同名文件对应 H；旧 F 仅在[研究配置](../research/experiments/configs/fp16-staged-resident-acoustics.json)中保留，供复现历史数据。
- `low-vram-acoustic-staged.json` 和 `minimum-vram-acoustic-staged.json` 分别合并到 `low-vram.json`、`minimum-vram.json`。
- Prefill query128 组合仅保留[研究配置](../research/experiments/configs/fp16-staged-acoustics-prefill128.json)与测量记录，不作为当前档位提供。
