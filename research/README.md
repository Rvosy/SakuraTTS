# 研究归档

这里保存推理实现的来源、测量工具和失败记录，供维护者复现。普通使用从仓库首页和 `docs/` 进入；wheel、源码包及默认产品测试均不依赖此目录。

| 目录 | 内容 |
| --- | --- |
| `notes/` | 性能、内存、兼容性总结和早期路线记录 |
| `experiments/` | 逐次实验记录、已引用 JSON 和复现配置 |
| `tools/` | 专项诊断、回放和依赖实验数据的打包工具 |
| `tests/` | 实验工具回归测试 |

在安装开发依赖的 Git checkout 中运行：

```powershell
python research/run_tests.py
python research/compare_official.py --help
```

普通完整请求基准使用 `sakuratts benchmark`，固定语料在 `benchmarks/cases/`。实验输出写入被 Git 忽略的 `outputs/`、`results/` 或 `artifacts/`。

报告中的设备、模型和绝对路径属于当时记录。归档不会使数值失败变成通过；重跑需要准备对应平台、官方源码和模型资源。
