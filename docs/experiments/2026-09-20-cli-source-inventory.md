# 补齐 CLI 的 FP64 Prefill 源码记录

日期：2026-09-20。检查独立运行包时发现，日文 CLI 在推理前遍历已导入模块保存源码哈希，而 `gpt_prefill.py` 到第一次 FP64 Prefill 才导入。此前的 CLI 记录因此漏掉这个实际执行文件。

当前 CLI 固定使用 CPU FP64 Prefill，修复直接把该文件加入请求前的源码清单。没有提前执行 Prefill，也没有改变计算、加载顺序或计时范围。旧报告保留原样；本次补录不意味着旧清单已经完整。

用新源码快照分别运行 NumPy seed 0、1 的原始日文三句，两个真实 CLI 进程均退出 0。新增哈希与冻结文件一致，清单中其余哈希也逐项相符；文本、生成计数和停止摘要、参数、参考身份及 WAV 均与上一轮相同 seed 的完整权重路径一致，没有导入 Torch、Transformers 或中文前端。

本轮验证的是记录完整性和输出回归。CLI JSON 没有完整 token 序列，不据此声称逐 token 比较；没有新增 ASR、人工试听或性能结论。

证据目录：`SakuraTTS-References/runs/20260919T174043.687462Z-cli-prefill-source-inventory/`。其中 `run.py` 保存可复现调度，`source/` 保存执行源码，`processes.json` 保存两条完整命令、退出码与进程耗时，`comparison.json` 为 `passed`。基线为 `20260919T171011.567386Z-bound-runtime-release-suite/cli/full-{0,1}.wav`；再次生成应使用新目录。
