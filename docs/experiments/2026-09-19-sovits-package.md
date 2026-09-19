# V2Pro 声学解码权重包

日期：2026-09-19。本轮为“朱雀院红叶”V2Pro 增加独立 FP32 解码权重包。转换器使用 PyTorch 和固定官方源码，输出为 NumPy NPZ 与 JSON；声学独立运行时尚未实现。

## 包含哪些权重

`scripts/convert_sovits.py` 按官方非流式 decode 的实际依赖导出权重。当前 checkpoint 有 678 个张量，其中 650 个进入解码包：

| 模块 | 张量数 | FP32 原始字节 |
|---|---:|---:|
| `enc_p` | 235 | 56,584,192 |
| `flow` | 124 | 53,623,296 |
| `dec` | 290 | 59,697,152 |
| quantizer 的 `embed` | 1 | 3,145,728 |
| 合计 | 650 | 173,050,368 |

`enc_p.ssl_proj` 参与目标声学编码，保留在包中。根级 `ssl_proj` 用于从参考 SSL 特征提取语义 token，不参与已提供 codes 的 decode，因此没有放进解码包。

其余排除项在 manifest 中逐张量列出原因：`ref_enc`、`sv_emb`、`prelu`、`ge_to512` 用于准备参考条件；量化器的 `inited`、`cluster_size`、`embed_avg` 用于初始化或训练更新，decode 只读取 `embed`。训练后验网络 `enc_q` 原本就不在该 checkpoint 中，不能把它记为转换文件的缩减收益。

## 布局与参考条件

所有导出数组保留原始键名和轴顺序。Conv1d 权重为 `[out, in/groups, kernel]`，ConvTranspose1d 为 `[in, out/groups, kernel]`，Embedding 为 `[vocabulary, features]`；相对位置 embedding 的独立布局也写入 manifest。每个张量记录原始精度、shape、FP32 字节数与内容 SHA-256。

131 个 weight norm 模块保留原 `weight_g`、`weight_v` 和实际 `dim=0`，本轮没有进行归一化折叠。所选权重原来均为 FP16，转成 FP32 可精确表示其值。保存后重新打开 NPZ，已检查 650 个数组的全部字节与转换前一致。

解码包要求调用方提供 `[1, 1024, 1]` 的 `ge` 和 `[1, 512, 1]` 的 `ge512`。它们必须按官方表达式从同一 SoVITS 权重准备，manifest 使用 checkpoint SHA-256 绑定身份，并列出参考音频、预处理、说话人编码器、精度和准备源码等条件依赖。原模型路径仅作来源记录；校验 harness 由调用方显式传入 checkpoint 路径，再按哈希核对，不依赖固定绝对路径识别模型。

创建或更换参考音频时仍需要原模型及参考准备流程。原始 checkpoint 没有修改，也没有宣称只凭这个解码包就能处理任意新参考条件。显式噪声、`noise_scale` 和语速属于解码输入；当前数值验证范围只覆盖 `noise_scale=0.5`、`speed=1.0`、单请求和非流式路径。

## 包大小与当前验证边界

生成目录相对于参考目录为 `models/converted/20260919T113249.439600Z-f8bd92196175-sovits-decode-fp32/`。

- `weights.npz`：173,247,422 字节。
- 全包文件合计：173,621,870 字节，包含 manifest、转换脚本快照和官方源码许可。
- 原始 checkpoint：134,946,419 字节，权重为 FP16。

FP32 解码包仍比原 FP16 checkpoint 大，不能用排除张量数量宣布安装体积下降。这个实验首先分离日常解码所需权重和参考准备所需权重；最终发行包还需核算精度、参考资源和运行库。

转换已经完成，已检查官方模型可接受 checkpoint 的 schema、所有导出数组有限、归档后逐字节相同，以及原 checkpoint 转换前后的 SHA-256 不变。

随后使用 `harness/sovits_package_replay.py` 完成纯 CPU 重载对照，线程数为 2。先保存官方两条样例的输出与实际 `ge/ge512/noise`，然后删除六个参考准备或训练模块和三个 codebook 训练缓冲，将剩余张量全部清零，再从 NPZ 严格重载 650 个张量。中、日两例的 `quantized`、`mean`、`log_scale`、`mask`、`flow_input`、`flow_output`、`decoder_input` 和完整波形全部逐位相同，最大绝对误差均为 0。

原始对照保存在 `runs/20260919T113924.613659Z-sovits-package-replay-cpu/`，包含两例官方/重载后的数组、逐阶段误差、源码快照和命令。日文输出 153,600 样本，中文 186,880 样本，采样率均为 32 kHz；进程正常退出，退出码为 0。该校验采用官方 CPU 算子作为转换对照，没有实现自有声学算子，也没有进行新的 ASR、试听、正常计时或显存测量。

## 复现

在项目根目录执行，转换器每次创建新包：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-official-macos/bin/python" scripts/convert_sovits.py \
  --references "$REF" \
  --checkpoint "$REF/models/suzakuinmomiji/voice/models/朱雀院红叶_e8_s38928.pth"
"$REF/.venv-official-macos/bin/python" harness/sovits_package_replay.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T113249.439600Z-f8bd92196175-sovits-decode-fp32" \
  --checkpoint "$REF/models/suzakuinmomiji/voice/models/朱雀院红叶_e8_s38928.pth" \
  --official-trace-run "$REF/runs/20260919T102853.549769Z-official-mps" \
  --threads 2
```

原始权重 SHA-256 为 `f8bd92196175f435ae9df2bc01e87b0119a5d94c0af81ada4573a9720536ab38`，官方源码固定在 `48b1a0169a28582a8984402f82cf438d3bfa6aca`。源码 MIT 许可不等于模型可再分发许可；这里仍按用户提供的本地模型处理。
