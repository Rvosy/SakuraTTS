# 自有 GPT 沿自身生成历史的对照

日期：2026-09-19。两条原始中日文回归输入通过：自有 GPT 和采样模块使用官方实际采样噪声，每一步回填自己抽到的 token，最终生成序列、停止条件、返回索引和声学语义切片均与官方相同。没有用官方 token 强制引导后续计算。

## 输入与检查

沿用固定“朱雀院红叶”V2Pro、参考音频、官方文本条件及采样参数。GPT 使用显式 CPU FP64 Prefill 和 MLX FP32 Decode，采样使用 NumPy FP32；Top-k 15、Top-p 1、temperature 1、repetition penalty 1.35。自有进程没有导入 PyTorch，运行环境也未安装 PyTorch。

先在官方 MPS 诊断中保存每一步实际使用的指数噪声和过滤后概率。捕获函数保留固定官方实现的 `empty_like(probs).exponential_(1)` 及 `argmax(probs / noise)` 表达式。新增回归检查确认捕获前后的 token、CPU RNG 状态和函数恢复一致；实际官方两份 WAV 也与原始对照逐字节相同。捕获需要 CPU 复制和同步，仅用于诊断，不参与正常计时。

自有生成器读取相同的 phones、BERT 特征和参考语义前缀。之后只将自身采样结果传给 `decode()`。保存的官方 token 仅用于比较；若发生第一次分歧，此后的不同历史不再进行数值验收，报告会分别记录已比较和跳过的步骤，整体仍判生成不一致。

| 检查 | 日文原始样例 | 中文原始样例 |
|---|---:|---:|
| 自行采样步骤 | 121 | 147 |
| 同历史 logits 最大绝对误差 | `3.43323e-5` | `3.05176e-5` |
| 最终声学语义 token 数 | 120 | 146 |
| 第一次生成分歧 | 无 | 无 |
| EOS 条件 | sampled 和 argmax 同时为 EOS | sampled 和 argmax 同时为 EOS |

全部 268 步的 logits 满足原 `atol=1e-4, rtol=1e-5`；概率满足 `atol=1e-6, rtol=1e-5`。过滤集合、采样 token、撤回 EOS 后的历史、返回 idx 和 `[-idx:]` 语义切片均要求完全相同，不能用数值容差放过离散差异。释放后 MLX active/cache 均为 0。

## 结论的范围

这项检查补上了固定历史回放之外的反馈循环，但仍共享官方实际随机数。它没有证明 NumPy 与 PyTorch 同种子的 RNG 流相同，也没有验证其他 seed、Top-p 小于 1、无参考、流式和长句自由生成。已知的三项 Top-p 边界差异继续保留，见[采样实验](2026-09-19-sampling-semantics.md)。

本轮没有运行自有声学模块、生成新的自有音频、执行 ASR 或新增试听。用户已确认正常的是此前指定的官方与 `official_text` 候选四份样音，不能将该反馈直接记为自有完整引擎通过。

## 证据与复现

以下目录相对于 `SakuraTTS-References/`：

- `runs/20260919T114052.042551Z-official-mps/`：实际采样噪声、概率、完整 trace、官方 WAV 和正常退出记录。
- `runs/20260919T114420.173465Z-audio-run-comparison/`：捕获插桩前后两份官方 WAV 相同。
- `runs/20260919T115124.217760Z-native-gpt-generation/`：首次自有历史生成结果。
- `runs/20260919T115555.145914Z-native-gpt-generation/`：补齐 EOS 条件比较和分歧后跳过统计后的复跑，全部检查通过。
- `runs/20260919T121654.122637Z-native-gpt-generation/`：将生成循环移入 `src/sakuratts/generation.py` 后复跑，全部检查通过。
- `runs/20260919T122138.464428Z-native-source-migration-check/`：移植前后的全部 logits、token、history 和 semantic 逐位相同。

各目录保留实际命令、输入哈希、源码快照和每步 logits、完整 token 历史、语义切片。音频比较器通过原始语言、文本和重复序号匹配早期没有 `case_id` 的记录，没有改写旧清单。

源代码入口 `generate_semantic()` 只接收参考前缀和模型输入，不接收目标 token；正常调用使用自身 NumPy RNG。共享噪声与逐步 observer 为诊断选项，普通调用不保留每步 logits。容量不足仍由模型明确报错，不通过提前停止来隐藏未完成的文本。当前入口明确拒绝尚有兼容缺口的 Top-p 小于 1。

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-official-macos/bin/python" -u harness/run_reference_process.py \
  --references "$REF" --backend official --device mps --repeat 1 \
  --diagnostic --capture-sampling-noise
"$REF/.venv-mlx-macos/bin/python" harness/native_gpt_generation.py \
  --references "$REF" \
  --official-run "$REF/runs/20260919T114052.042551Z-official-mps" \
  --package "$REF/models/converted/20260919T104221.749176Z-010197bfc30b-gpt-fp32"
```

第一条命令会创建新目录；复现新采样记录时，将第二条的 `--official-run` 换成实际输出目录。
