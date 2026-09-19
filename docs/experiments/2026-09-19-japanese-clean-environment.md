# 独立日文运行环境

日期：2026-09-19。按日文优先的范围，在参考目录新建 `.venv-japanese-macos`，使用 Python 3.11.15 和 `requirements-mlx-japanese.txt`。旧官方、Lite 和累计研发环境均未改动。

安装前先执行只接受 wheel 的 dry-run，再执行正式安装；两次退出码均为 0，`pip check` 通过。环境中没有 Torch、Transformers、tokenizers、OpenCC、PyPinyin 或 jieba-fast，安装也没有带入中文 BERT / G2PW 资源。日文仍保留原 OpenJTalk、用户词典、Sudachi、Nani ONNX 与真实语言识别能力。

## 实际体积

已安装 distribution 清单共 29 项，含 pip 和 setuptools。记录到的文件逻辑大小共 707,773,683 字节，约 674.99 MiB。主要部分如下：

| 安装项 | 文件逻辑大小 |
|---|---:|
| SudachiDict-core | 207.41 MiB |
| mlx-metal | 199.59 MiB |
| pyopenjtalk-plus | 109.08 MiB |
| onnxruntime | 78.78 MiB |
| numpy | 32.83 MiB |
| pip + setuptools | 22.92 MiB |

这些数字包含包内数据和生成文件，未包含 Python 基础解释器、外部语言识别模型、独立用户词典、GPT / SoVITS 权重、参考包、系统动态库与下载缓存。它不是完整安装包体积，也不是最终最小运行库大小。MLX / Metal 占用仅属于当前 Mac 验证环境，Windows 分发须重新选型和测量。

此次收窄语言范围不改变模型权重。日文的零 BERT 特征来自官方日文语义，不是在中文模式下关闭必要特征。日文入口明确拒绝未支持语言，中文已有独立代码和实验保留，待后续阶段恢复。

## 验证与证据

安装记录位于 `SakuraTTS-References/runs/20260919T141553.840721Z-japanese-clean-env/`：

- `requirements.txt` 与项目中的日文依赖清单逐字节相同。
- `setup.json`、`execution.json`、原始日志：环境创建、dry-run 和安装命令及退出码。
- `dry-run.json`、`install.json`：解析后的包、版本、下载来源与 wheel 哈希。
- `installed-files.json`：逐 distribution、逐文件的大小清单。
- `pip-check.json`、`result.json`：依赖检查、解释器身份与禁止依赖的实际可用性。

随后使用这个环境和它自己的主词典运行四条日文原始文本的完整诊断，记录在 `141941.619153Z-native-text-speech-diagnostic`。目标音素和特征、自有生成历史、停止及波形对照通过，进程退出码为 0；详细生成与正常计时结果见对应的日文整链记录。没有借用旧环境的 Python 包来补齐缺失依赖。

可以在新的空目录复现安装：

```sh
<已安装的Python3.11> -m venv <新日文环境目录>
<新日文环境目录>/bin/python -m pip install --only-binary=:all: -r requirements-mlx-japanese.txt
<新日文环境目录>/bin/python -m pip check
```

模型、词典和参考包的身份与路径由生成 Harness 单独固定；安装成功本身不代表合成或语音质量验收。
