# 首层 attention 的 softmax 数值差异

日期：2026-09-19。前一轮已经把 `ja-punctuation` 的 MRTE 超差追到 SSL/text 编码器的累计舍入差异；统一 FP64 LayerNorm 虽然修复了该例，却使日文长句波形退化，因而被否决。详见 [逐层数值对照](2026-09-19-encoder-layer-numerics.md)。本轮继续拆解首次出现差异的 SSL 第 0 层 attention，未修改生产运行时。

## 固定输入与展开验证

`harness/attention_node_diagnosis.py` 直接读取官方逐层捕获中的 `ssl.0.input`，形状为 FP32 `[1,192,422]`，不重新生成语义 token。模型仍是朱雀院红叶 V2Pro，官方提交为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`，权重使用原 FP32 转换包。该层为两头、每头 96 维、相对位置窗口 4；输入没有 padding，mask 全为 1。

官方 MPS 与 MLX CPU 分别保存 20 个节点，覆盖 q/k/v、query 缩放、普通分数、相对位置分数及变换、mask 后分数、softmax、两路 value 加权、求和及输出投影。MLX 另外保存一份逐节点重放结果：每个算子的直接输入全部取自官方节点，以区分上游传播误差与该算子自身差异。

官方诊断展开的输出和 attention 概率均与真实模块逐位相同，真实输出也逐位复现历史层捕获。MLX 展开输出与真实 `multihead()` 逐位相同。两轮都正常退出，因此下面的差异不是诊断展开改变公式或执行路径后制造的现象。

## 首次差异发生在 softmax

q/k/v、query 除以 `sqrt(96)`、普通与相对位置分数、相对位置变换、相加及 mask 后分数全部逐位相同。首个差异是对相同分数执行 softmax。其最大绝对误差为 `2.98023224e-7`，RMS 为 `2.67503134e-9`。

| 节点 | 累计最大绝对误差 | 累计 RMS | 官方同输入下的最大误差 |
|---|---:|---:|---:|
| mask 后分数 | 0 | 0 | 0 |
| softmax 概率 | 2.98023224e-7 | 2.67503134e-9 | 2.98023224e-7 |
| 普通 value 加权 | 7.62939453e-6 | 3.37272083e-7 | 0 |
| 相对位置 value 加权 | 9.53674316e-7 | 5.82507912e-8 | 0 |
| 两路结果相加 | 8.58306885e-6 | 3.43890258e-7 | 0 |
| 输出投影 | 9.53674316e-6 | 7.42821444e-7 | 0 |

softmax 之后的全部算子，只要直接输入换成对应官方数组，结果也都逐位相同。这说明本例首层输出的差异来自 softmax 概率及其向后传播，未发现该层投影、相对位置索引或矩阵乘法的独立差异。累计输出误差与前一轮记录完全相同。

20 个累计节点和 20 个同输入节点都通过原有 `atol=1e-4, rtol=1e-5`，没有调整容差。这只解释一个固定输入上的首层差异；其他层及完整声学链仍有累计效应，原十例中的 MRTE 失败继续保留。这里也没有对其他语言、模型版本、padding 或 mask 边界作验证。

## 证据与下一步

证据保存在参考目录下：

- 官方节点：`runs/20260919T131028.986460Z-attention-nodes-official/`。
- MLX 节点及逐节点同输入结果：`runs/20260919T131046.245358Z-attention-nodes-mlx/`。

每轮保存命令、模型与输入身份、源码快照、全部节点 NPZ 和比较结果。源码及数组 SHA-256 已核验；MLX 进程没有导入 PyTorch。`diagnostic_seconds` 包含捕获与 CPU 复制，不作为正常运行速度指标。本轮没有新增 ASR、人工试听或内存收益结论。

复现官方捕获：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-official-macos/bin/python" harness/attention_node_diagnosis.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-layer-run "$REF/runs/20260919T124415.513327Z-encoder-layer-diagnosis-official-none" \
  --backend official
```

在 MLX 环境使用相同参数，将 `--backend` 改为 `mlx`，并通过 `--official-attention-run` 指向新生成的官方节点目录。

下一步可以只针对 softmax 做同分数对照，比较现有实现与统一的高精度归约候选，再决定是否值得完整回归。高精度计算不保证更接近官方 MPS 的 FP32 结果。任何候选都需要应用到所有相应 attention，重新检查十例全部阶段及波形；不能只让这一处或 `ja-punctuation` 通过就采纳。
