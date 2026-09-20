# 开发与迁移

先读与改动相关的 `docs/specs/`、`docs/adr/` 和测试。依赖声明在 `pyproject.toml`；`requirements/` 是此前实测环境的冻结快照，不能当作额外功能声明。运行、转换与私有声学解释器仍分开安装。

```powershell
python -m pip install -e ".[japanese,dev,server]"
python -m unittest discover -s tests
python benchmarks/run.py models/mika --text "こんにちは。" --output outputs/benchmark.json
python scripts/build_preview.py --output dist/preview-0.1.0a1
```

单元测试使用小型资源和替身，不等于 GPU 或音质验收。CUDA smoke 使用本机准备好的模型及运行环境；MLX 验证需要 Mac。

## 旧目录对应关系

| 原位置 | 当前位置 |
| --- | --- |
| `nvidia.py` 的实现 | `backends/cuda/engine.py`；原导入保留兼容入口 |
| `cuda_*` / `ort_*` / `mlx_*` | 对应 `backends/cuda/`、`onnx/`、`mlx/` |
| 文本、日文、中文、G2PW 模块 | `frontend/` |
| `tone_sandhi` / `zh_normalization` | `frontend/_vendor/`，保留许可与来源 |
| worker、参考和采样模块 | `_internal/` |
| `harness/` | `research/tools/`，常用入口提到 `benchmarks/` |
| `docs/experiments/` | `research/experiments/` |
| GPT / ONNX 导出和 Windows 参考准备 | `_internal/conversion/`，由 `sakuratts convert` 编排 |
| 其余模型与资源脚本 | `tools/` |
| 根目录 `requirements-*.txt` | `requirements/*.txt` |

旧 `sakuratts synthesize --config ...` 和 `sakuratts.nvidia.NVIDIAEngine` 继续可用。内部模块没有逐个添加别名；仓库内调用方与测试已迁移，外部研究脚本需按表更新。

新实验输出写到被 Git 忽略的 `outputs/`、`results/` 或 `artifacts/`。`research/experiments/data/` 只保留此次迁移前已经引用的证据，避免丢失数值失败与性能数字的来源。
