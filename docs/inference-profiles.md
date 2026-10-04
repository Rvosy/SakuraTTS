# 推理档位

公共入口通过 `backend` 选择设备，通过 `profile` 选择预设。后端名为 `cpu`、`cuda`、`directml`、`mlx`；具体模型范围和验证状态见[兼容矩阵](specs/compatibility-matrix.md)。`sakuratts capabilities` 可在不加载计算库的情况下列出可用组合。

| 档位 | CPU | AMD DirectML | NVIDIA CUDA | Apple MLX（实验） |
|---|---|---|---|---|
| `int8` | 默认且唯一档位：ORT INT8 GPT，4 线程；FP32 声学，8 线程 | 不提供 | 不提供 | 不提供 |
| `fp16` | 不提供 | 默认且唯一档位：GPU GPT + 全图 FP16 声学，KV 容量 1280 | FP16 GPT + FP16 分块声学，释放 GPT 请求状态 | 尚未实现 |
| `fp32` | 不提供 | 不提供 | FP32，保留模型 | FP32 decode / 声学，CPU FP64 Prefill |
| `low-memory` | 不提供 | 不提供 | FP16，声学 Session 错峰 | FP32，释放 GPT 请求状态 |
| `minimum-memory` | 不提供 | 不提供 | FP16，GPT / 声学及声学 Session 错峰 | FP32，GPT / 声学错峰 |

预设的具体选项由 [profiles.py](../sakuratts/profiles.py) 定义。CPU / DirectML 的详细参数见[设备指南](cpu-amd.md)。MLX 使用原生 V2Pro / V2ProPlus 包，公共转换可从原始权重生成；声学 encoder 在 CPU 执行，flow / decoder 在 Metal 执行。Apple 真机与整合包验收见 [Apple 指南](apple.md)。

```powershell
sakuratts tts MODEL_AMD --backend directml --profile fp16 --text "こんにちは。" --output outputs/hello.wav
sakuratts serve MODEL_CPU --backend cpu --profile int8 --runtime-mode managed
```

```python
from sakuratts import Engine

with Engine.load("MODEL_AMD", backend="directml") as engine:
    audio = engine.synthesize("こんにちは。")
```

YAML 使用 `sakuratts.backend`、`sakuratts.profile` 和 `sakuratts.runtime_options`。命令行或 Python 显式参数覆盖 YAML 对应字段，`runtime_options` 覆盖预设，显式 `experimental` 再覆盖同名选项。省略 `profile` 时 CPU 选择 `int8`、DirectML 选择 `fp16`；CUDA 与 MLX 保持原有默认行为。加载按所选实现与资源执行，声学精度需符合显式预设。`minimum-memory` 用于低级 Engine、`tts` 或 `managed` HTTP，默认 `direct` HTTP 不接受错峰加载。

加载已有模型包时，选择档位不会改写包内模型。HTTP 配置提供原始 GPT / SoVITS 权重和准备环境时，首次加载及权重切换按所选档位自动准备资源。CUDA 的三个 FP16 档共用 FP16 分块声学缓存，与 FP32 分开保存；转换期间在临时目录内完成图检查，失败不发布缓存。CPU 默认需要 GPT INT8 附加资源与 FP32 整图声学包；DirectML 默认需要 FP16 GPT、对应容量的静态 Decode 图与 FP16 声学包。运行时不要求设备执行报告或哈希匹配。降精度可能改变采样序列和音频长度，准备和试听方法见[CPU / AMD 指南](cpu-amd.md)。

CPU 模式不加载 GPU 执行器。DirectML 把 GPT Transformer 与声学模型放在 GPU，文本处理、embedding 和采样仍使用 CPU。默认 GPT 线程数为 4，DirectML 声学的 CPU 部分为 2 线程。两条路径都允许显式调整线程、容量和驻留策略；其他 CPU / AMD 候选仅保留在[历史实验](../research/notes/cpu-amd-precision-listening-20260927.md)，不再作为公开档位。

## NVIDIA 既有档位与实测

CUDA 当前提供四档：FP32、FP16 标准、FP16 低显存、FP16 极限。对应此前实验的 A、C、E、H。FP16 标准是常用推荐候选；不传实验选项时，运行时仍沿用 FP32 默认。FP16 的人工听音验收尚未完成。

| 档位 | 配置文件 | 适用情况 |
|---|---|---|
| A．FP32 | [fp32.json](../examples/fp32.json) | 保留当前 FP32 路径与精度 |
| C．FP16 标准 | [fp16.json](../examples/fp16.json) | 日常使用，兼顾速度与显存 |
| E．FP16 低显存 | [low-vram.json](../examples/low-vram.json) | 更少显存，接受每片段额外加载 |
| H．FP16 极限 | [minimum-vram.json](../examples/minimum-vram.json) | 显存优先，接受每片段重载等待 |

## 2026-09-21 实测

### 峰值显存

| 档位 | 完整请求观测峰值 |
|---|---:|
| FP32 | 2148.6 MB |
| FP16 标准 | 758.2 MB |
| FP16 低显存 | 584.1 MB |
| FP16 极限 | 465.1 MB |

### 空闲显存

| 档位 | 最后请求后空闲 |
|---|---:|
| FP32 | 1253.1 MB |
| FP16 标准 | 569.4 MB |
| FP16 低显存 | 387.0 MB |
| FP16 极限 | 106.5 MB |

### 请求耗时

| 档位 | 热短句 | 热长句 |
|---|---:|---:|
| FP32 | 0.289 s | 3.290 s |
| FP16 标准 | 0.234 s | 2.677 s |
| FP16 低显存 | 1.009 s | 3.560 s |
| FP16 极限 | 3.055 s | 5.481 s |

数值来自 RTX 5060 8 GB、Windows WDDM、Sakura V2ProPlus 日文单请求，使用准备好的中性参考、`cut0` 分句设置。MB 为十进制单位；显存是同一时刻主进程与声学 worker 的 Dedicated Usage 之和，空闲取请求后约 1 秒的稳定值。耗时为持续采样下的热请求中位数；FP32 与 FP16 来自不同轮次，输出长度也有差异。其他模型、输入和设备需分别测量。详见 [FP32／FP16 对照](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/low-vram-20260921.md)及[声学错峰实测](https://github.com/Rvosy/SakuraTTS/blob/main/research/notes/acoustic-session-staging-20260921.md)。

## 选择与使用

A 需要 FP32 声学包。C、E、H 都需要 FP16 分块声学包，默认块长 256。使用原始权重的 HTTP 配置可通过 `sakuratts.profile` 自动准备，已有部署包则需自行选择匹配资源。下面的 JSON 文件是 `--experimental` 的执行选项，模型路径另行传入。

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

A、C、E 可经 `serve MODEL --experimental FILE` 启动 HTTP 服务。H 的总体 `policy="staged"` 可用于 Python Engine、`tts` CLI，或显式启用的 `managed` HTTP 模式；默认 `direct` 仍拒绝 H。YAML 的 `is_half=true` 尚未映射到实验档位；使用上述 JSON 显式选择。

```powershell
sakuratts serve MODEL --experimental examples/minimum-vram.json --runtime-mode managed
```

控制模式默认空闲 60 秒后退出整个推理进程树，释放进程级 GPU 资源。H 档醒着时也按片段错峰加载权重，活动显存较低，但会增加每片段等待；提前唤醒只准备运行环境，`preparation="runtime_init"`，不会提前把两组权重都加载。需要连续短句速度时可把命令中的配置换为 `examples/fp16.json`，保留标准档执行速度，再通过空闲休眠减少长期显存驻留。控制模式的 HTTP 测量单列在[后台驻留与提前唤醒](background-runtime.md)。

C 保留模型权重，语义生成完成后释放 GPT 请求状态；E 进一步让 latent／vocoder Session 错峰；H 再让 GPT／声学模型错峰。三个 FP16 档均关闭 Prefill query 分块，在已测对应文本上的音频逐采样一致。`cut5` 是独立的文本分句选项，会改变停顿与生成过程，不再单列为一个档位；E、H 在片段较多时会有更多重载等待。

## 旧配置迁移

- 原 `examples/low-vram.json` 对应 C，现在改用 `examples/fp16.json`；新的 `low-vram.json` 对应 E。
- 原 `examples/minimum-vram.json` 对应旧 F，现在的同名文件对应 H；旧 F 仅在[研究配置](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/configs/fp16-staged-resident-acoustics.json)中保留，供复现历史数据。
- `low-vram-acoustic-staged.json` 和 `minimum-vram-acoustic-staged.json` 分别合并到 `low-vram.json`、`minimum-vram.json`。
- Prefill query128 组合仅保留[研究配置](https://github.com/Rvosy/SakuraTTS/blob/main/research/experiments/configs/fp16-staged-acoustics-prefill128.json)与测量记录，不作为当前档位提供。
