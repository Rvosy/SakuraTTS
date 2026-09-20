# V2Pro 固定条件声学对照

日期：2026-09-19。平台为 Apple M4、16 GiB 统一内存，MPS / FP32；模型为“朱雀院红叶”V2Pro。

官方和 Lite 在相同语义 token、音素、参考条件及显式噪声下，从参考条件 `ge` 到声码器输入的全部已捕获结果逐位一致。最终波形有小幅数值差异，仍在实验预设容差内。这缩小了两条固定样例的排查范围，不能据此认定漏句、发音和音色已经通过验收。

## 对齐方法

`research/tools/sovits_fixed_conditions.py` 分别在两个独立进程中运行官方和 Lite，只加载 SoVITS。输入取自官方诊断记录 `runs/20260919T102853.549769Z-official-mps/`，包括目标语义 token、目标音素、参考频谱和说话人 embedding。没有加载 GPT、BERT、CNHuBERT 或外部说话人模型。

官方提交为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`，Lite 为 `6c049397142f4c9147a85f86b6ba37546e93a188`。SoVITS 权重 SHA-256 为 `f8bd92196175f435ae9df2bc01e87b0119a5d94c0af81ada4573a9720536ab38`。加载后检查了全部 678 个 checkpoint 张量，均与原权重相同。官方构造器保留的 `enc_q` 缺失权重单独记录，它不参与当前 decode。

官方运行从 flow 实际收到的参数中捕获最终 `ge`，再把这个数组注入 Lite。Lite 另用相同参考频谱和 embedding 重算一次 `ge`，该结果也与官方逐位一致。

噪声由独立 CPU FP32 随机生成器产生，seed 为 `20260919`。官方使用并保存完整噪声数组，Lite 重放该数组；不是用相同 seed 假定两端的随机数相同。两端均检查 decode 只调用一次 `randn_like`，`noise_scale=0.5`、`speed=1.0`。原官方 trace 未捕获噪声，其旧波形仅留作来源记录，不参与本次数值比较。

## 结果

| 固定样例 | 语义 token | 音素 | 输出样本 / 时长 | 波形最大绝对误差 | 波形 RMS 误差 | 波形相对 L2 |
|---|---:|---:|---:|---:|---:|---:|
| 日文 `ja-reported-intro` | 120 | 58 | 153,600 / 4.80 s | 2.2592023e-5 | 5.4344005e-7 | 3.5771338e-6 |
| 中文 `zh-reported-greeting` | 146 | 56 | 186,880 / 5.84 s | 1.3008714e-5 | 5.2391108e-7 | 3.4653985e-6 |

采样率为 32 kHz。以下 14 个阶段的全部数组逐位一致：`ge`、`ge_projected`、`quantized`、`ssl_encoded`、`text_encoded`、`mrte`、`encoder_hidden`、`mean`、`log_scale`、`mask`、`noise`、`flow_input`、`flow_output`、`decoder_input`。第 15 个阶段 `waveform` 存在上表差异。所有数组均为有限值，两例的全部阶段都满足预设的 `atol=1e-4, rtol=1e-5`，没有调整阈值。

Lite 加载器在 CPU 上折叠声码器的 weight norm，再把模型传到 MPS；官方保留 forward hook，在执行设备上计算归一化。差异首次出现在声码器输出，与该实现差别吻合。不过本轮没有独立切换归一化设备，因此它仍是原因候选，不能写成已经证实的因果结论。

## 资源与质量边界

该 harness 使用 hooks、同步和 CPU 数组复制，结果中的 `diagnostic_seconds` 只用于说明诊断成本，不能作为正常声学生成速度或端到端加速证据。本轮未测内存峰值，也未比较 NVIDIA 显存。

权重审计发现两个可继续验证的生命周期成本：官方 `enc_q` 有 54,560,256 字节参数，当前 decode 不依赖它；参考条件分支中 `ref_enc`、`sv_emb`、`prelu` 和 `ge_to512` 合计 88,525,824 字节参数，当前固定参考下可以尝试预计算结果后释放。参数字节只说明候选规模，实际节省要由独立端到端对照测量。Lite 本来就没有构造 `enc_q`，不能把它列为尚未采用的 Lite 优化。

输出为未裁头、未归一化、未补静音的完整 Float32 WAV。四个 WAV 均与各自 NPZ 中的 `waveform` 逐位相同，样本数、采样率、NPZ/WAV/源码快照 SHA-256 已独立核对。本轮没有 ASR 或人工试听；其他实验的试听结论不能转移给这些新波形。

## 证据与复现

目录相对于 `/Users/beyondpower/Documents/Projects/SakuraTTS-References`：

- `runs/20260919T111545.846735Z-sovits-fixed-official-mps/`
- `runs/20260919T111827.205483Z-sovits-fixed-lite-mps/`

各目录保存 `result.json`、中日两份中间数组、完整 Float32 WAV 和运行时源码快照。JSON 保存原始命令、源提交及文件哈希、checkpoint 哈希、参数分组字节数和逐阶段误差。CPU 模型审计记录分别为 `runs/20260919T110747.492433Z-sovits-fixed-official-cpu-model-check/` 与 `runs/20260919T110808.136250Z-sovits-fixed-lite-cpu-model-check/`。更早一次官方 CPU 审计因 `HParams` 不能直接 JSON 序列化而失败，原目录保留；修复序列化后新建了上述成功记录。

在项目根目录执行，每次运行新建证据目录：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-official-macos/bin/python" -u research/tools/sovits_fixed_conditions.py \
  --references "$REF" \
  --official-trace-run "$REF/runs/20260919T102853.549769Z-official-mps" \
  --backend official --device mps
"$REF/.venv-macos/bin/python" -u research/tools/sovits_fixed_conditions.py \
  --references "$REF" \
  --official-trace-run "$REF/runs/20260919T102853.549769Z-official-mps" \
  --backend lite --device mps \
  --official-conditions "$REF/runs/20260919T111545.846735Z-sovits-fixed-official-mps"
```

第二条命令引用本轮已经保存的官方条件；如要配对新的官方运行，应替换 `--official-conditions`。比较失败时，新 harness 将状态写为 `numerical_mismatch` 并返回退出码 1；既有结果不回写。
