# 日文标点样例的 MRTE 误差来源

日期：2026-09-19。继续调查[十例声学回归](2026-09-19-expanded-acoustic.md)中 `ja-punctuation` 的单个 MRTE 超差元素。阈值仍为 `atol=1e-4, rtol=1e-5`，没有改生产编码器或替换输出坐标。

## 同输入对照

`research/tools/mrte_numerical_diagnosis.py` 读取官方和 MLX CPU 编码器已经保存的 SSL、text 输出，并组合成四组输入。参考条件 `ge512` 完全相同。官方路径直接调用固定提交的 MRTE 模块，通过 hooks 保存内部结果；MLX 路径使用现有 Conv1d、attention 算子，并核对诊断展开与原 `multihead` 调用逐位一致。

官方 MRTE 用官方 SSL/text 输入时，输出与原 gold 逐位相同。MLX CPU 用原 native SSL/text 输入时，也逐位复现原完整链路中的 MRTE 失败。因此这些阶段边界可以用来分解原问题。

对官方模块替换输入后，在失败坐标 `[0,21,125]` 得到：

| SSL 来源 | text 来源 | 官方 MRTE 输出 | 相对原官方结果的误差 |
|---|---|---:|---:|
| 官方 | 官方 | 2.3635818958 | 0 |
| 官方 | MLX CPU | 2.3635127544 | -6.91414e-5 |
| MLX CPU | 官方 | 2.3635163307 | -6.55651e-5 |
| MLX CPU | MLX CPU | 2.3634321690 | -1.49727e-4 |

该坐标允许误差为 `1.23636e-4`。同时使用两个 native 编码结果时，官方 MRTE 自己也在同一位置超差。单独替换任一输入时，整个 MRTE 输出仍通过，但两者的影响并非简单相加；交叉注意力包含 softmax。

MLX CPU 与官方 MRTE 在四组相同输入上，11 个内部阶段的 44 项比较全部通过原定容差。MRTE 最终输出的同输入最大绝对误差为 `1.90735e-5` 至 `2.28882e-5`。当前失败不能归为只出现在 MLX MRTE 中的公式或权重布局偏差；上游 SSL/text 的小差异已经足以触发它。

## 差异主要经注意力分支传到输出

进入 MRTE 前，SSL 输出最大绝对误差为 `6.07967e-6`，text 为 `7.30157e-7`，二者都通过阶段容差。给官方 MRTE 注入这些差异后，query 最大变化约 `2.71797e-5`，注意力分数约 `4.19617e-5`，概率约 `6.70552e-6`，attention 输出约 `8.82149e-5`。它们各自仍在当前容差内，最后投影到 MRTE 输出时出现单元素超差。

为分清注意力分支和 SSL 残差分支，将官方模块保存的 FP32 分支差值，使用原 `c_post` 的第 21 行权重在 FP64 下投影到时间位置 125。分解结果为：

| 来源 | 对该输出差值的贡献 |
|---|---:|
| attention 输出变化 | -1.48093e-4 |
| SSL 残差变化 | -2.03740e-6 |
| 残差相加的舍入差异 | -6.46783e-7 |
| 最后投影的舍入差异 | +1.05037e-6 |
| 合计 | -1.49727e-4 |

主要变化来自交叉注意力分支。这是对现有失败坐标的诊断分解，没有据此改动该坐标或对这条输入设置专门路径。中间数组、完整四组输出和分解表保存在官方诊断 run 中的 `attribution.json`。

## FP64 MRTE 不能消除上游差异

还用独立 NumPy FP64 实现同一 MRTE：所有 1×1 投影直接作仿射矩阵运算，按四个头计算缩放点积、稳定 softmax 和 value 加权，再执行残差与最后投影。它读取同一份原始 FP32 权重，整个诊断不导入 PyTorch 或 MLX。

该 FP64 计算与官方在四组相同输入的 44 项内部比较也全部通过。但输入换成 native SSL/text 后，相对于原官方管线，在同一坐标的误差变为 `-1.69487e-4`，仍然失败。只提高 MRTE 内部精度不会修复输入已经携带的差异；本轮没有采用新的 MRTE 数学路径。

本轮将下一步调查范围缩小到 MRTE 之前的 SSL 和 text 编码器，需要按层、按同输入继续分解累积舍入。没有证据支持改变模型、关闭特征、按样例切换后端或放宽阈值。原十例 run 仍为 `numerical_mismatch`；最终波形十例通过、MRTE 一项失败这两个事实同时保留。

## 证据与复现

参考目录下的三轮结果均退出 0，状态 `diagnosis_completed` 表示诊断矩阵执行完整，不表示所有被比较的管线通过。源码快照和四组数组的 SHA-256 均已核验。

- 官方 MPS：`runs/20260919T123543.843892Z-mrte-diagnosis-official-mps/`。
- MLX CPU：`runs/20260919T123620.493233Z-mrte-diagnosis-mlx-cpu/`。
- NumPy FP64：`runs/20260919T123620.494668Z-mrte-diagnosis-numpy-cpu/`。

官方输入来自 `runs/20260919T122232.246358Z-sovits-fixed-official-mps/`，native 输入来自 `runs/20260919T122652.881973Z-mlx-sovits-complete-gpu/`。先运行官方捕获：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-official-macos/bin/python" research/tools/mrte_numerical_diagnosis.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T122232.246358Z-sovits-fixed-official-mps" \
  --native-acoustic "$REF/runs/20260919T122652.881973Z-mlx-sovits-complete-gpu" \
  --backend official --device mps
```

在 MLX 环境里使用相同参数，将 `--backend` 改为 `mlx` 或 `numpy`，设备改为 `cpu`，并增加 `--official-mrte-run` 指向刚生成的官方 MRTE 目录，即可复现同输入对照。`--case` 默认为 `ja-punctuation`。

这三轮保存中间阶段，会增加复制和同步；并发进行的两项 CPU 诊断也不适合计时比较。这里不报告速度或资源收益，没有新增 ASR、人工试听或音质结论。
