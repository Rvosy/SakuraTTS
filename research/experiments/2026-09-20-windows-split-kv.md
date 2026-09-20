# Windows GPT split-KV 注意力候选

RTX 5060 8 GB、Sakura V2ProPlus 上，FP32 split-KV 256 将固定回放长句的完整请求中位数从 3045.15 ms 降到 2007.75 ms，减少 34.07%；短句从 168.94 ms 增至 178.42 ms，慢 5.61%。使用与官方捕获一致的 CUDA 参考后，baseline 和候选各 15 个请求的官方回放检查全部通过。默认仍为 `baseline`。

首次 FP32 256 运行曾出现未找到根因的 Prefill 异常，FP16 split-KV 也未通过相对 FP16 baseline 的同精度严格检查。上述完整请求通过仅覆盖本模型、四类日文和固定随机回放，尚无内容或听感验收。完整数据与来源哈希见 [JSON](data/2026-09-20-windows-split-kv.json)，早期失败和参考使用错误的记录未覆盖。

## 实现与边界

Decode 按 KV 位置分块，每个 head 的每块由一个 CUDA block 计算局部最大值、指数和及未归一化加权 V，再用第二个 kernel 按全局最大值合并。统计和合并均使用 FP32；无效块写入零和、零加权 V，尾部缓存不参与 softmax。FP32、FP16 使用相同 KV 布局，Prefill、采样与停止规则未改。

`CUDAGPT.load(..., attention="split-kv", attention_chunk_size=256)` 显式启用候选；块长也可选 512。Engine、CLI、benchmark 和生命周期 Harness 已透传选择。`--gpt-attention` 默认 `baseline`，不按输入自动切换实现。

容量 2048 时，原 Decode 工作区为 26,628 bytes；256 分块为 44,036 bytes，增加 17 KiB；512 分块为 35,332 bytes，增加 8.5 KiB。额外数组跨层复用，随请求状态和 Graph 一起释放重建。这些数字只计数组 `nbytes`，本轮没有重新测全引擎显存峰值。

## 固定 token 历史

四类输入共 1182 步，每类先预热一次；baseline 和 512 各计时三次，256 各计时五次。下表是 Prefill 加全部 Decode 的中位数，包含逐步 logits 回传，不含前端、采样或声学。使用同一模型、参考 archive、capture 与容量，执行器源码 SHA-256 均为 `30666267ba9d92969ecf5bbbdf14879f47e197b6c9ea78726972165ed370f358`。

| 输入 / 步数 | FP32 baseline ms | FP32 256 重测 ms | FP32 512 ms |
|---|---:|---:|---:|
| 短句 / 44 | 92.76 | 90.93 | 91.43 |
| 长句 / 676 | 2333.64 | 1299.95 | 1663.97 |
| 多句 / 259 | 633.12 | 509.68 | 602.99 |
| 标点 / 203 | 460.65 | 388.18 | 446.23 |

baseline、512 和 256 重测均通过官方 `atol=1e-4, rtol=1e-5`，候选对同精度 baseline 也通过。短句在其他请求之后、释放重建以及 Graph/eager 的结果逐位相同。256 重测取 `fp32-256-repeat5`，没有用它替换首次失败。

首次 `fp32-256` 的标点输入在第 0 步 Prefill 就相对 baseline 出现最大差 `0.0002770424`；完整历史相对官方最大差 `0.0039818287`，65,337 个 logits 元素超差，原状态为 `numerical_screen_failed`。其余三类通过。随后原顺序的两次和五次重复、把标点放在最前的运行均通过；额外 192 组 Prefill/首个 Decode 检查也全部通过。检查覆盖 baseline、256、512、不同请求和状态重建，但未重现首次异常。不能据此把原因归到请求顺序、额外 scratch 或某个 BLAS 算法，也不能宣布问题已经修复。

FP16 256 的固定长历史为 1211.74 ms，但四类对 FP16 baseline 的严格比较均失败，最大绝对误差为 0.07959–0.12010，RMS 为 0.01675–0.01757。旧 Harness 只把相对官方的宽松 FP16 工程筛查纳入汇总，因此原报告仍有 `engineering_passed=true`；这个字段不足以支持 attention 同精度验收。新 Harness 已增加同精度严格检查，本报告将 FP16 split-KV 保留为未通过的候选。

独立 attention 边界验证有 104 例，采用 NumPy FP64 的 QK、稳定 softmax 和加权 V 作为参考，覆盖 FP32/FP16、256/512 分块、1553/2048 容量、块边界两侧和写成 NaN 的无效尾 KV。全部通过各自预设容差：FP32 为 `1e-4/1e-5`，FP16 为 `1e-3/1e-3`。最大差分别为 `4.77e-7` 和 `6.10e-5`；该检查不替代完整 GPT 的同精度验收。

## 原始文本到完整 PCM

`fp32-correct-reference-baseline` 与 `fp32-correct-reference-256` 各在新进程中完成 15 个请求：首个短句、四类各三次热请求，以及空闲卸载后的首次和热重载。两轮均使用 resident 策略、CUDA Graph、容量 2048 和 FP32 声学；`runtime-validation.json` 载入的参考数组与官方捕获逐项相同，固定回放同一批随机采样数组与声学噪声。目标音素和 token 只用于请求完成后的比较，没有注入生成。持续资源采样关闭，WAV/JSON 写盘不计入请求时间。

| 输入 / 完整 PCM 秒 | baseline ms | split-KV 256 ms | 变化 |
|---|---:|---:|---:|
| 短句 / 2.02 | 168.94 | 178.42 | +5.61% |
| 长句 / 27.30 | 3045.15 | 2007.75 | -34.07% |
| 多句 / 10.62 | 932.13 | 793.92 | -14.83% |
| 标点 / 8.38 | 691.01 | 612.60 | -11.35% |

两轮生成的 token、音频长度相同，因此这组时间可以比较相同工作量。所有请求的音素、采样 token、语义 token、停止、PCM 长度和官方原 FP32 数值容差检查都通过，两份状态均为 `completed`，运行期间源码未变化。短句没有获得加速；三次热请求也不足以覆盖长期抖动。这里未测试自然随机生成、其他模型或流式首包延迟，也不把容差通过等同于 PCM 逐字节相同。

## 早期回放的参考错误

早期 `fp32-full-baseline` 与 `fp32-full-256` 使用相同 CPU 参考，15 个配对 token 全部相同、PCM 最大差 1 LSB，原容差外样本数为零。该组长句为 3144.66→2006.35 ms，短句为 164.02→162.33 ms；这些数据保留为同 CPU 参考下的比较，不与主表两轮混用。

对官方捕获，早期两轮的每次热长句均有 14 个 PCM 样本超差、最大差 5 LSB；多句均有 170 个超差、最大差 6 LSB。后续查明 `runtime.json` 指向 CPU 准备的 `ge/ge512`，官方捕获使用 CUDA 参考，已有 `reference_arrays_equal=false` 未被旧 Harness 拒绝。因此这组数据不能用于同官方输入的验收；两份原报告的 `replay_validation_failed` 保留。Harness 已将参考数组一致性纳入运行前检查，主表改用修正参考后的两次独立运行。

## 复现

固定历史分别运行 baseline、256 和 512，使用新的输出目录；`--compare` 指向同精度 baseline 的结果目录：

```powershell
.venv-windows-runtime\Scripts\python.exe research/tools/windows_gpt_precision.py `
  --gpt models/windows-sakura/gpt `
  --reference models/windows-sakura/references/中性 `
  --captures outputs/windows-precision/captures.json `
  --precision fp32 --attention split-kv --attention-chunk-size 256 `
  --repeats 5 --compare outputs/windows-split-kv/fp32-baseline `
  --output outputs/windows-split-kv/new-fixed-256
```

同官方输入的完整回放使用 `research/tools/windows_nvidia_benchmark.py`，参数为 `--config models/windows-sakura/runtime-validation.json --gpt-precision fp32 --gpt-attention split-kv --gpt-attention-chunk-size 256 --repeats 3 --no-memory-sampler --skip-reference-switch --skip-random --idle-ms 0 --replay-captures outputs/windows-precision/captures.json --output 新目录`。baseline 只把 attention 改为 `baseline`。`runtime-validation.json` 引用 CUDA 准备的参考；数据文件保留早期两轮及后续验证的逐项结果和来源哈希。
