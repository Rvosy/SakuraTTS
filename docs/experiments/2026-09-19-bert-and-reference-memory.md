# 保留官方输出的参考资源与 BERT 裁剪

日期：2026-09-19。范围：Apple M4、MPS / FP32、“朱雀院红叶”V2Pro，中文 BERT 始终启用。该实验仍依赖 PyTorch，用来验证可迁移的计算和生命周期改动，不是最终轻量运行包。

## 改动与数值依据

第一项改动提前完成官方参考语义、频谱、参考文本特征和说话人 embedding，之后释放 CNHuBERT 与 ERes2Net。目标文本改变时仍重新计算目标前端与 GPT；没有复用依赖目标文本的前缀 KV。参考准备结果保存到新实验目录，原音频和权重未修改。

第二项改动是 `src/sakuratts/bert_features.py`。实际中文 BERT 是 24 层、隐藏维度 1024；官方只使用 `hidden_states[-3]`，即第 22 层输出。新实现只加载 embedding 和前 22 层，不计算后两层或 MLM head。输入 token、CLS / SEP 去除、word2ph 展开及 FP32 均保持不变。

| 独立 BERT 检查 | 实测 |
|---|---|
| CPU / MPS hidden `[1,32,1024]` | 均与原模型逐元素一致 |
| CPU / MPS 音素特征 `[1024,56]` | 均与原模型逐元素一致 |
| FP32 唯一参数字节 | 1,302,182,432 → 1,197,121,536，少 100.19 MiB（8.07%） |
| MPS 正常 forward，1 次预热后 5 次中位数 | 21.217 → 18.309 ms；只含已分词的 BERT forward |
| MPS 加载后 allocated | 少约 100.19 MiB |
| MPS 加载后 driver | 本轮没有下降，不能把参数减少量直接视为进程总占用收益 |
| BERT 单模块卸载并 empty_cache | 两条路径的 allocated 均为 0 |

独立 BERT 证据：`../SakuraTTS-References/runs/20260919T103517.823052Z-bert-equivalence/`。CPU 数值证据在 `runs/20260919T103346.439902Z-bert-equivalence/`；该 CPU 计时与其他实验并发，只采用数值结果。

## 端到端正常运行

使用相同原始两条回归文本、模型、参考、seed 和采样参数，每种语言运行 4 次。基线与组合优化共 16 个 WAV 的实际文件哈希逐对相同，包含每次开头与结尾的全部样本。该结论表示优化保持既有官方输出；没有替代人工对内容和音色的验收。

| 指标 | 官方普通路径 | 参考释放 + BERT 裁剪 |
|---|---:|---:|
| 日文热运行中位数，3 次 | 2.494 s | 2.058 s |
| 中文热运行中位数，3 次 | 3.099 s | 2.487 s |
| 最后中文请求后的 MPS allocated | 2650.76 MiB | 1953.19 MiB |
| 同一边界 MPS driver | 3903.33 MiB | 2803.33 MiB |
| 全部模型卸载、GC、empty_cache 后 allocated | 9.17 MiB | 9.01 MiB |
| 同一卸载边界 driver | 183.33 MiB | 107.33 MiB |

请求结束时 allocated 少了 **697.58 MiB（26.32%）**。这些是统一内存边界，不能称为峰值或 NVIDIA 显存。热运行样本仍少，且有顺序运行的频率和后台负载漂移，只作为探索性速度结果，不作正式加速承诺。

优化把参考准备移到了请求之前，耗时 1.426 秒且含保存文件。因此首次 `infer` 不能直接与基线比较；本轮没有证明冷启动更快。RSS 也没有稳定下降，不能将 MPS allocated 的下降表述为整机同等内存收益。

原始证据（相对于参考目录）：

- `runs/20260919T103238.622081Z-official-mps/`：普通路径，正常计时。
- `runs/20260919T103819.533577Z-official-mps/`：组合优化，正常计时。
- `runs/20260919T104146.445379Z-lifecycle-bert-comparison/comparison.json`：配置与 16 个 WAV 的独立哈希复核，逐次热时间和资源边界。

## 单独的内存采样

新增 `--sample-memory`，在独立运行中轮询 MPS allocated、driver 和进程 RSS，保存原始 `memory-samples.csv`。请求间隔设为 10 ms，但实际最大采样间隙约 0.30–0.31 秒，可能受 Python 执行和调度影响。下面只能称为采样观察到的最大值，是实际峰值的下界；不能声称捕获了全部瞬态分配。

| 从加载到卸载的采样最大值 | 普通路径 | 组合优化 |
|---|---:|---:|
| MPS allocated | 2959.14 MiB | 2526.75 MiB |
| MPS driver | 7435.34 MiB | 6451.33 MiB |
| 进程 RSS | 2520.95 MiB | 2793.02 MiB |

两条路径各生成中日文 2 次，均保持已有官方 WAV 哈希。优化路径的 allocated 最大观测值出现在参考准备阶段，生成阶段的最大观测值更低。driver 瞬态明显高于请求结束边界，下一步需要定位它对应的声学临时空间或运行库工作区。仅看结束快照会低估这种压力。

采样结果在 `runs/20260919T104335.444675Z-official-mps/` 与 `runs/20260919T104438.424564Z-official-mps/`。这些运行的时间包含采样开销，不混入上一节速度比较。PyTorch、Metal 和进程计数口径不同，不能相加。

## 验证与复现

结构测试覆盖源配置不变、非支持 decoder 架构拒绝、包含 padding / token_type 的 BERT 中间层一致性。诊断测试另覆盖“重复惩罚原地改变 logits 后才判断 argmax EOS”的停止语义。

早期源码快照曾平铺保存，无法直接按相对路径执行；当前已保留 `source/harness/cases/` 和 `source/src/sakuratts/`。旧证据不改写，使用其记录的项目源码版本和命令复现。旧两条诊断的末步停止结论未受采样前 argmax 记录问题影响，后续诊断同时保存采样前后值。

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-official-macos/bin/python" -m unittest discover -s tests -v
"$REF/.venv-official-macos/bin/python" harness/bert_equivalence.py --references "$REF" --frontend-json "$REF/runs/20260919-frontend-audit/official-frontend.json" --devices mps --repeat 5 --warmup 1 --threads 4
"$REF/.venv-official-macos/bin/python" harness/reference_smoke.py --references "$REF" --backend official --device mps --repeat 4 --prepare-reference --prune-bert
# 内存轮询另跑，不采用该轮的性能时间。
"$REF/.venv-official-macos/bin/python" harness/reference_smoke.py --references "$REF" --backend official --device mps --repeat 2 --prepare-reference --prune-bert --sample-memory
```

原 checkpoint 和完整开发环境仍保留，当前没有实现安装包体积下降。更广语料、参考切换、持久缓存加载、其他模型、FP16 和 CUDA 继续单独验证。
