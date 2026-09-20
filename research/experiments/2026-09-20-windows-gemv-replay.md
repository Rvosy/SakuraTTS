# Windows 选择性 GEMV 完整固定历史回放

日期：2026-09-20。Windows 11 / RTX 5060 8 GB，Sakura V2ProPlus，GPT FP16 baseline attention。

[独立 GEMV 探针](2026-09-20-windows-gemv.md)中通过三个采样点的形状，放进完整模型后得到不同结果：**仅替换 FFN-out 时，全部记录 logits 与 cuBLAS 逐位一致；仅替换最终 Output 时，全部通过原严格容差；替换 Attention output 或三项组合时，四类历史均失败。** 这说明少数真实输入上的单算子通过，不能代替完整历史检查。

本轮四个开发实验到此冻结。生产 GPT 继续使用现有 cuBLAS，Windows 预览版发布优先。没有修改生产内核，也没有将本轮结果解释为音质验收或稳定提速。

## 执行与输入

`research/tools/windows_gemv_replay.py` 复用 `windows_gpt_precision.replay`，逐一注入官方 capture 中的原始 token。每条历史产生一行 Prefill logits 和全部 Decode logits；最后一个已采样 token 不再喂入 Decode。短句、长句、多句、标点分别为 44、676、259、203 行，共 1182 行、1178 次 Decode。参考条件、文本、KV 容量 2048 和 baseline attention 均保持不变。

每轮先运行新加载的 FP16 cuBLAS 模型，再运行另一个新加载的候选模型。候选沿用已测 `warp4` 内核：HALF 输入与权重、FP32 累积，Transformer 输出仍为 HALF，最终 logits 仍为 FP32。可选项仅为 `attention_output`、`ffn_out`、`output`，QKV 和 FFN-in 不允许替换。

路由按当前模型的实际权重指针匹配，且只在 `_decode_graph_body` 内启用。Prefill 的最终单行 Output 也保留 cuBLAS，不能只凭 batch=1 判断替换。内核提前编译，既有 Decode 首次预热后再捕获 CUDA Graph；热图重放不经过 Python 形状选择。移除路由前先释放图和请求缓冲，恢复原方法，再关闭模型。

每轮每个模型保存 warm、一次热回放、全部 case 的 eager 回放，以及首例经过其他请求后和释放重建后的结果，共 14 个完整观察数组。全部候选观察值直接与同 case、同 phase 的 cuBLAS 数组比较，避免把多次近似相等误当作可传递关系。

## 数值结果

严格条件保持 `atol=1e-4, rtol=1e-5`。下表的 max / RMS 覆盖完整 logits 矩阵，包含相同的 Prefill 首行；“首个失败”按 Decode step 编号，0 留给 Prefill。逐步 argmax 仅比较每行最大值位置，不表示候选采样或停止行为已通过。

| 替换范围 | 输入 | 最大绝对误差 | RMS | 超差元素数 | 逐步 argmax 一致 | 首个失败 step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 三项组合 | 短句 | 0.054290295 | 0.014672166 | 42601 | 44 / 44 | 2 |
| 三项组合 | 长句 | 0.11767483 | 0.017917489 | 685008 | 676 / 676 | 1 |
| 三项组合 | 多句 | 0.087272644 | 0.01721128 | 259727 | 258 / 259 | 3 |
| 三项组合 | 标点 | 0.091982841 | 0.016790058 | 204483 | 203 / 203 | 1 |
| Output | 短句 | 8.5830688e-06 | 1.6172675e-06 | 0 | 44 / 44 | 无 |
| Output | 长句 | 1.1444092e-05 | 1.7007027e-06 | 0 | 676 / 676 | 无 |
| Output | 多句 | 1.1444092e-05 | 1.6340971e-06 | 0 | 259 / 259 | 无 |
| Output | 标点 | 1.5258789e-05 | 1.6932687e-06 | 0 | 203 / 203 | 无 |
| Attention output | 短句 | 0.054293156 | 0.014672176 | 42601 | 44 / 44 | 2 |
| Attention output | 长句 | 0.11767769 | 0.017917483 | 685007 | 676 / 676 | 1 |
| Attention output | 多句 | 0.087272644 | 0.017211285 | 259725 | 258 / 259 | 3 |
| Attention output | 标点 | 0.091982841 | 0.016790055 | 204482 | 203 / 203 | 1 |
| FFN-out | 短句 | 0 | 0 | 0 | 44 / 44 | 无 |
| FFN-out | 长句 | 0 | 0 | 0 | 676 / 676 | 无 |
| FFN-out | 多句 | 0 | 0 | 0 | 259 / 259 | 无 |
| FFN-out | 标点 | 0 | 0 | 0 | 203 / 203 | 无 |

三项组合和 Attention output 单项在多句的第 127 步出现 argmax 变化。组合候选从 cuBLAS 的 token 81 变为 792：cuBLAS 两者 logits 为 `12.75822735`、`12.75514412`，候选为 `12.78459549`、`12.80219936`。其他步骤 argmax 相同，仍有大量元素超出原容差。这里没有运行候选采样，不能将 argmax 结果当作实际生成 token。

原 harness 继承的 `argmax_equal` 是整张矩阵展平后的最大值比较。本轮原始 JSON 保持原样；独立审计另算每行 argmax，表格与证据 JSON 使用逐步结果，避免把原字段的 `true` 误读为所有步骤相同。

## 误差随 Decode 的变化

三项组合中，短句的第一步最大误差约 `5.72e-6`，第二步开始严格失败；多句第一步约 `4.77e-6`，第三步开始失败。长句和标点在第一步已分别出现 `0.01805`、`0.01765` 的最大误差。Prefill 首行均逐位一致。

误差没有随步数单调增加。组合候选的最大误差分别出现在短句 step 14、长句 step 53、多句 step 131、标点 step 146；长句 step 53 的最大误差为 `0.11767483`，最后一步为 `0.04520798`。长句按 step 1–16、17–64、65–256、257–675 分段的 RMS 分别约 `0.01316`、`0.01995`、`0.01572`、`0.01877`。

Attention output 单项已经足以产生相近差异。该结果没有定位到某个具体层或 HALF 舍入点，因此不能把所有误差都归因于前一轮发现的某个中点。FFN-out 单项在全部观察中逐位一致，最终 Output 单项的误差则始终较小；两者的通过范围仍限于这四条固定历史。

## 生命周期与证据完整性

四轮共保存 112 个数组、29072 行、29798800 个有限 FP32 值。CPU 审计重算全部数组哈希和 56 对同 phase 数值检查；56 对 Prefill 首行均逐位一致。每轮两种模型的 warm / 热回放、graph / eager、跨请求及释放重建均逐位一致，共 80 对非自身生命周期比较。三次后续运行的新鲜 cuBLAS 基线与第一轮的 42 对同名数组也逐位一致。

每个模型记录 4 次用于图预热 / 捕获的 Python body 调用和 1178 次 eager body 调用。97 个 Decode 权重各有 1182 次成功 Python dispatch，每个权重还有 14 次 Prefill 调用保留 cuBLAS。这些是主机入队和建图计数；CUDA Graph 重放不再经过 Python，不能把这些数字称为 GPU 实际执行次数。

模型归档、checkpoint、参考条件、capture、源码和候选权重身份均保留哈希。全部 32 个“最终回放 logits 对官方 FP32 logits”的比较仍未通过原严格容差，原失败继续记录。FFN-out / Output 相对同精度 cuBLAS 的通过不会覆盖这些既有失败。

## 计时边界

每个 case 仅做一次 warm 和一次热计时。以下为热 Decode 时间，单位 ms，每格依次为 cuBLAS / 候选；Prefill 单独记录。计时含固定历史的逐 token 调用和 FP32 logits 回传，不含采样、前端、声学、模型加载、编译或结果写盘。

| 替换范围 | 短句 | 长句 | 多句 | 标点 |
| --- | ---: | ---: | ---: | ---: |
| 三项组合 | 71.93 / 77.14 | 2365.81 / 2295.32 | 603.80 / 587.59 | 424.86 / 424.02 |
| Output | 70.59 / 70.63 | 2345.37 / 2316.73 | 592.81 / 587.57 | 413.72 / 411.72 |
| Attention output | 75.32 / 78.63 | 2294.98 / 2263.66 | 588.20 / 595.80 | 408.25 / 429.41 |
| FFN-out | 69.27 / 79.30 | 2313.54 / 2271.03 | 582.89 / 596.32 | 402.99 / 433.15 |

各轮均固定先测基线再测候选。单次热结果有快有慢，尚不能确认稳定收益，更不能由此宣称完整 TTS 请求加速。声学显存和音频质量没有在本轮测量。

## 运行与验收记录

默认或 `--check-only` 仅做 CPU 准入，不导入 CUDA 后端。实际回放需要显式 `--run-gpu`，输出目录必须是新目录。例如三项组合：

```powershell
.venv-windows-runtime/Scripts/python.exe -B research/tools/windows_gemv_replay.py `
  --gpt models/windows-sakura/gpt `
  --reference models/windows-sakura/references-cuda/中性 `
  --captures outputs/windows-precision/captures.json `
  --output outputs/windows-gemv-replay/new-all-shapes `
  --shapes attention_output ffn_out output --repeats 1 --run-gpu
```

单项实验分别将 `--shapes` 设为 `output`、`attention_output`、`ffn_out`，并指定不同新目录。真实四类输入的 CPU 准入通过；相关 20 项 CPU 测试通过，其中新增 10 项覆盖路由、Prefill 隔离、dtype、异常清理、完整历史及逐观察值检查。

实机三项组合、Attention output 单项均以返回码 1、`numerical_check_failed` 结束；Output 和 FFN-out 单项以返回码 0、`completed` 结束。所有运行保留 `quality_accepted=false`。本轮没有运行候选完整 TTS 请求、ASR 或人工听音。

[证据 JSON](data/2026-09-20-windows-gemv-replay.json)保留各轮状态、误差随步数变化的摘要、argmax 变化、调度记录、计时、身份和哈希。完整逐步序列、原始结果、NPZ 和独立 CPU 审计位于未跟踪的 `outputs/windows-gemv-replay`；哈希用于识别这些本地归档，不表示克隆仓库即可获得模型和原始数据。
