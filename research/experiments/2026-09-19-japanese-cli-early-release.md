# 普通日文 CLI：声学前释放 GPT 状态的回归

日期：2026-09-19。模型和环境沿用[普通 CLI 随机生成实验](2026-09-19-japanese-cli-free-sampling.md)。本轮只运行一次 seed 0，没有重跑 ASR 或新增试听。

CLI 现在显式向 `synthesize_prepared` 传入 `release_gpt_state=True`，在语义生成完成后、声学准备开始前释放 GPT 请求 KV；权重继续保留。公开 API 的默认值仍是 `False`。输出 JSON 用 `runtime_policy.release_gpt_state_before_acoustic=true` 记录实际策略，其他采样参数未改。

这次仍从原始三句日文生成，使用干净的 `.venv-japanese-macos`、相同的前端/参考/GPT/SoVITS 包、NumPy seed 0 和高精度计算设置。旧 WAV 与 JSON 只在生成后用于比对，没有把历史目标特征、token 或随机噪声传入模型。

子进程实际退出码为 0。25 项离线检查全部通过：全文、规范化结果、58 个音素和语言分段相同，模型包及参考身份、字典、依赖、采样参数均相同，未导入 Torch、Transformers 或中文模块。sampled/semantic token 数仍为 115/114，返回索引为 114，停止原因仍为 `argmax_eos` 和 `sample_eos`。旧普通 CLI 没有保存 token ID 序列，因此这里不声称逐个 token ID 相等。

新 WAV 与 `20260919T143508.441566Z-japanese-cli-free-sampling/seed0-first/speech.wav` 的整个文件逐字节相同，SHA-256 均为：

```text
b023ea2a447538028b9b034c9cc7f5d273ae9710fb4701c88d4803934188ecbe
```

波形主体为 4.56 秒，含追加尾静音的 PCM 为 4.86 秒。完整请求耗时 1.518 秒，父进程墙钟 1.730 秒；这是单次入口复核，不用于估计速度收益。本轮没有测量内存峰值。

此前对原 seed 0 WAV 的 ASR 转写可以通过字节身份关联到新 WAV：`こんにちは。今日は良い天気ですね。よろしくお願いします。` 这不是一次新识别，也不能替代 wa、音色或自然度的人工评价。新 CLI 记录中的 `quality` 保留生成时的 `not_run`，关联关系另存在验证结果的 `prior_asr` 字段中。原 WAV、原 CLI JSON 与此前 ASR 结果的哈希保持不变。

证据目录为：

```text
/Users/beyondpower/Documents/Projects/SakuraTTS-References/runs/20260919T151137.527299Z-japanese-cli-early-release
```

`source/` 保存本次实际执行的 15 个源文件；`prepared.json` 保存其哈希和旧结果身份；`process.json` 保存完整命令、环境覆盖项、真实退出码与墙钟时间，stdout/stderr 分别保存。`speech.wav` 和 `speech.json` 是新输出，`result.json` 保存逐项比较与既有 ASR 的关联。复现时使用 `process.json` 中的命令，并为 `--output` 选择新的路径；CLI 会拒绝覆盖已有 WAV 或 JSON。

主要源码 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `tools/synthesize_japanese.py` | `6b079b49ffcd66207176feb973ecda793a98b1498187ad6f368c6c68a9690bc7` |
| `src/sakuratts/_internal/synthesis.py` | `32eabcf8c16661a4a7c640fedf1d368610a3f07f9cfb33d5757ee13582c82452` |
| `src/sakuratts/backends/mlx/gpt.py` | `390c0796b43b65c8f1c2f9bcc3724db9e4cf1cf2daf1f95403a14bad55439e29` |
