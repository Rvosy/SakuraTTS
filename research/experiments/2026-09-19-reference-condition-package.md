# 最小参考条件包与独立重载

日期：2026-09-19。已有官方单参考 V2Pro 准备结果已导出为独立包，包含五个必要数组和身份清单。新进程从另一目录加载后，五个数组的形状、类型和全部字节均与原产物相同；12 项错误身份、损坏文件及对齐检查全部按预期拒绝。加载器只依赖 NumPy。

该包用于后续“原始目标文本 + 离线准备参考包”的请求入口。本轮只验收打包与重载，没有执行目标文本生成、ASR 或人工试听，也没有实现无 Torch 的新参考音频准备。

## 包内内容

原准备目录为 `SakuraTTS-References/runs/20260919T112618.136695Z-official-mps/`，官方提交为 `48b1a0169a28582a8984402f82cf438d3bfa6aca`。原运行已记录正常退出，使用“朱雀院红叶”V2Pro、日文参考及 MPS / FP32。导出器只读取原产物，原 checkpoint、音频和历史目录未修改。

| 数组 | 类型与形状 | 原始字节 |
|---|---|---:|
| `reference_phones` | int64 `[47]` | 376 |
| `prompt_semantic` | int64 `[113]` | 904 |
| `reference_bert` | FP32 `[1024, 47]` | 192,512 |
| `ge` | FP32 `[1, 1024, 1]` | 4,096 |
| `ge512` | FP32 `[1, 512, 1]` | 2,048 |
| 合计 | | 199,936 |

NPZ 归档为 201,222 字节，加上 JSON 清单共 208,106 字节，约 203.23 KiB。当前日文参考 BERT 全零，仍直接保留原 FP32 数组，没有增加特殊压缩表示。

原 `prepared-reference.npz` 中的频谱、16 kHz 说话人波形及 20,480 维 embedding 留在历史诊断目录。普通生成已经读取最终的 `ge` / `ge512`，无需再次加载这些中间结果。原两个 NPZ 共 1,338,878 字节；新包范围更小，但这不是整个运行包或常驻内存减少的测量。

## 身份与加载行为

`tools/prepare_reference_package.py` 复核原 GPT checkpoint、SoVITS checkpoint 和参考音频的当前哈希与历史记录相同，然后绑定：

- 两份模型与参考音频的 SHA-256、原始参考文本、语言、官方提交。
- 补点后的 prompt 和规范化文本、准备精度及历史记录中的执行设备和版本信息。
- 两个原始 NPZ、原准备 JSON、完成记录、退出记录和当时准备脚本快照的哈希。
- 新归档哈希、各数组类型 / 形状 / 字节数 / 内容哈希，以及导出器和加载器源码哈希。

固定官方提交中的相关源码与符号表在导出时解析并记录哈希。清单明确区分这个来源与历史执行时逐文件捕获：旧实验并没有保存每个实际导入的官方源文件哈希。

旧记录也缺少 CNHuBERT / ERes2Net 权重与配置、日文词典和 Nani 资源、pyopenjtalk 版本、音频解码与重采样实现的完整历史身份。这些字段保存在 `unrecorded_historical_resources`，没有用当前资源的哈希补写成历史事实。后续重新准备参考时应当直接捕获这些身份。

`src/sakuratts/_internal/reference_condition.py` 的 `PreparedReference.load()` 必须接收调用方的 GPT 与 SoVITS checkpoint 哈希。可以再指定原参考文本、语言、音频哈希、官方提交和包清单哈希；给定值不匹配时直接报错。这里的 `reference_text` 是原参考转写，不是目标文本，也不是补点后的 prompt。

加载器始终校验归档与五个数组，不从 provenance 路径读取旧机器上的文件，不导入任何计算后端。返回数组设为只读。当前只接受这一 V2Pro 条件结构，并校验参考 BERT 与音素数量、`ge` / `ge512` 的维度。

## 独立验证

`research/tools/reference_package_replay.py` 只复制 `manifest.json` 与 `conditions.npz` 到新目录，另起 Python 进程加载，再与历史数组逐字节对照。新进程实际退出码为 0，未导入 Torch、Transformers、MLX 或 ONNX Runtime。

12 个负例覆盖：错误 GPT / SoVITS / 音频 / 官方提交身份、缺失模型身份、错误参考文本 / 语言 / 清单哈希、归档字节损坏、重新计算归档哈希后仍与原数组哈希不同、缺失必要数组，以及参考 BERT 与音素长度不一致。每项保留实际报错文本。生成的负例包只位于新的验收目录中。

本轮没有运行模型，也没有测量合成速度、GPU 占用或新的语音质量。代码与数据格式没有绑定 MLX 或 MPS；实际重载发生在 macOS，Windows 重载仍待验证。

## 产物与复现

路径相对于 `SakuraTTS-References/`：

- 包：`models/converted/20260919T135540.094161Z-v2pro-reference/`。
- 导出记录：`runs/20260919T135540.094161Z-reference-condition-export/`。
- 独立重载、拒绝结果、源码快照及日志：`runs/20260919T135548.471879Z-reference-condition-reload/`。

包清单 SHA-256 为 `4f39d040310fa9de86fa18c3ba5b976349ae66ba0d69ec25d03528cf326d13a3`。在项目根目录执行以下命令会生成新的包及证据目录：

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" tools/prepare_reference_package.py \
  --references "$REF" \
  --prepared-run "$REF/runs/20260919T112618.136695Z-official-mps" \
  --gpt-checkpoint "$REF/models/suzakuinmomiji/voice/models/朱雀院红叶-e15.ckpt" \
  --sovits-checkpoint "$REF/models/suzakuinmomiji/voice/models/朱雀院红叶_e8_s38928.pth"

"$REF/.venv-mlx-macos/bin/python" research/tools/reference_package_replay.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T135540.094161Z-v2pro-reference" \
  --prepared-run "$REF/runs/20260919T112618.136695Z-official-mps"
```

下一步将参考包接入原始目标文本入口，沿既有 GPT 和声学对照检查完整请求。新增或修改参考音频仍需独立准备工具；可以先让允许依赖 Torch 的工具沿官方计算生成条件并退出，普通运行时只加载包。
