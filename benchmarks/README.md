# 基准入口

`python benchmarks/run.py MODEL --text "こんにちは。" --output outputs/benchmark.json`
通过公共 Engine 连续生成三次完整音频，分别保存首轮与后续请求数据。用 `--repeats` 调整次数，用 `--experimental options.json` 显式选择实验参数。耗时包含请求内发生的模型加载。首段延迟与音质另行测量，计量定义见[基准协议](../docs/specs/benchmark-protocol.md)。

在 Git checkout 中运行 `python research/compare_official.py --help` 查看 Windows 官方回放、生命周期和资源对照参数。固定文本在 `benchmarks/cases/`。

细分数值诊断与历史实验在 `research/tools/`，原始证据在 `research/experiments/`。新的运行结果写入被 Git 忽略的 `outputs/`、`results/` 或 `artifacts/`。
