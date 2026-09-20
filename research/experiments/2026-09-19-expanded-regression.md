# 参考资源释放与 BERT 裁剪的扩展回归

日期：2026-09-19。10 条固定输入的优化前后 WAV 逐字节相同，原始输入未修改。范围包括用户报告的两条多句文本、中日文各一条短句、长句、标点文本，以及中英、日英混合文本。

这验证了当前 V2Pro 模型、参考与 seed 下，预先准备参考条件、释放 CNHuBERT / ERes2Net 和裁剪 BERT 无依赖计算没有改变输出。它不证明扩展文本的内容、音色已经通过试听，也不涉及自有 MLX GPT 的数值问题。

## 固定输入与输出

`research/tools/compare_audio_runs.py` 检查实际文件字节及 SHA-256，同时核对模型与参考哈希、官方提交、输入文本、采样参数、seed、精度和设备。两份基线分别提供 8 条普通语料和 2 条混合文本；第一份在进入混合文本时因资源缺失失败，已完成的 8 条记录和失败状态都保留。

路径均相对于 `SakuraTTS-References/`：

| 证据 | 路径 |
|---|---|
| 官方前 8 条诊断结果 | `runs/20260919T104548.448908Z-official-mps/` |
| 官方混合文本诊断结果 | `runs/20260919T110311.350332Z-official-mps/` |
| 组合优化，10 条完整请求 | `runs/20260919T110419.798581Z-official-mps/` |
| 离线逐文件对照，10 / 10 相同 | `runs/20260919T111055.358397Z-audio-run-comparison/` |

基线带逐步诊断插桩，候选是正常请求，因此不使用两者时间计算速度收益。候选只运行一轮，长句 RTF 分别为日文 1.20、中文 1.55，提示长句速度仍需要继续改进；这些数值不代表稳定热运行性能。之前同条件多次热运行与独立内存采样见 [资源实验](2026-09-19-bert-and-reference-memory.md)。

复现波形对照，无需加载模型：

```sh
TTS_REF_DIR=/Users/beyondpower/Documents/Projects/SakuraTTS-References
python3 research/tools/compare_audio_runs.py --references "$TTS_REF_DIR" \
  --baseline "$TTS_REF_DIR/runs/20260919T104548.448908Z-official-mps" \
  "$TTS_REF_DIR/runs/20260919T110311.350332Z-official-mps" \
  --candidate "$TTS_REF_DIR/runs/20260919T110419.798581Z-official-mps"
```

每个生成目录的 `result.json` 保存实际命令与源代码快照。组合优化参数是 `--prepare-reference --prune-bert`，两条混合输入的语言参数沿用官方 `auto`。

## 英文前端资源

首次混合文本先遇到 `ModuleNotFoundError: No module named 'wordsegment'`，补齐包后又遇到 NLTK 资源缺失。自动下载被解析到 `198.18.0.15` 的安全检查拒绝，原异常留存在 `runs/20260919T105904.169242Z-english-resource-discovery/`。

`tools/prepare_english_resources.py` 从固定的官方 `nltk_data` 提交 `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a` 获取 cmudict 和 tagger 归档，按官方索引与预设 SHA-256 校验；其余已有语言资源逐文件校验后复制到参考目录的 `models/nltk_data`。遇到内容不同的同名文件会报错，不覆盖。没有关闭 NLTK 安全检查或改动上游 Python 源码。

准备结果、来源、文件哈希及 CPU G2P 检查在 `runs/20260919T110121.436768Z-english-resource-prepare/`。探针将 NLTK 搜索路径限制在该目录，`Please check the audio.` 得到 14 个音素，随后两条官方混合文本完成生成。`wordsegment==1.3.1` 已加入现有官方开发环境，旧环境记录保持原样；这不是最终运行包的依赖方案。

## 下一项验证

继续定位日文长句的自有 GPT 数值超差，并用相同语义、音素、音色条件和显式噪声对照声学计算。参考条件已能提前保存，下一步检查声学模型中仅在准备阶段使用的参数能否释放，以及这会怎样影响空闲常驻和请求峰值。
