# SoVITS 共用一次归档加载

日期：2026-09-19。自有声学入口原先分别调用 encoder、flow 和 decoder 的 `load()`，每次都读取清单、校验整个归档并打开 NPZ。现在由 `SoVITSPackage` 管理一次上下文，三个模块读取各自的张量。组件单独加载的入口仍可使用。

包读取只依赖 NumPy，不负责设备放置。张量流式交给对应模块，不额外保留整包 CPU 副本；退出上下文后关闭归档。紧凑包仍按逐张量存储声明恢复并校验原 FP32 内容，格式、模型版本、dtype 和实际张量集合均检查。

## 正确性

`harness/sovits_package_loading.py` 从已提交的 `a67074a` 取出原三组件加载源码，在独立新进程中与当前路径比较。同一紧凑包、CPU encoder / GPU flow+decoder、FP64 softmax 累积模式，全部 650 个运行权重的形状与内容哈希相同。

| 诊断计数 | 原三次加载 | 共享加载 |
|---|---:|---:|
| 整归档 SHA-256 | 3 | 1 |
| NPZ 打开 | 3 | 1 |
| 存储声明验证 | 3 | 1 |
| 实际张量读取 | 650 | 650 |

原实现并没有把每个张量解压三次；此次消除的是整归档重复校验、打开与清单验证，不能把收益解释为减少三份完整运行权重。

完整声学对照另用原 FP32 包运行十例：120 个阶段与改动前的 `133516.751269Z-encoder-softmax-mlx-fp64-runtime` 逐位相同，120 项官方容差检查通过。固定样例、采样历史与声学噪声没有变。该路径采用已经验证的显式 softmax 累积选项，不新增精度修改，也没有新试听。

新增四项小包测试检查紧凑张量恢复与归档关闭、归档损坏、未声明张量、展开后的错误身份。测试只需 NumPy，无模型资源。

## 加载成本

诊断和正常计时使用不同进程。正常路径不安装计数钩子、不复制权重到 CPU；每种策略各五个新进程，交替先后顺序。时间包括清单、完整 SHA、NPZ 读取、张量验证和设备构造，截止设备同步。它不包括 Python 导入，也不是清除 OS 文件缓存后的磁盘冷启动。

同一紧凑包的构造中位数为 **231.0→155.2 ms，减少 32.8%**。五次原始值分别为 243.9、221.4、231.0、221.1、233.0 ms，与 169.8、155.2、155.4、154.5、154.0 ms。收益仅限声学加载，不代表请求生成提速。

| 五个进程的边界中位数 | 原路径 | 共享路径 |
|---|---:|---:|
| 加载后 RSS | 240.36 MiB | 236.36 MiB |
| 加载后 MLX active | 165.15 MiB | 165.15 MiB |
| 释放并清缓存后 RSS | 74.55 MiB | 71.33 MiB |
| 释放后 MLX active / cache | 0 / 0 | 0 / 0 |

活动权重不变，几 MiB 的 RSS 差异不作为主要优化收益。以上为 Apple 统一内存的进程 / MLX 统计，不是 CUDA 显存。此轮尚未折叠 WeightNorm；相关候选默认关闭，另行测量。

## 证据

相对于 `SakuraTTS-References/runs/`：

- `20260919T135854.508721Z-sovits-package-loading`：两次权重诊断、十个正常新进程、快照、计数、逐权重哈希和实际退出码，全部通过。
- `20260919T140014.436577Z-encoder-softmax-mlx-fp64-runtime`：十例 120 阶段和同链数组，实际退出码 0。
- `135935.484030Z` 的路径拼写错误和 `135951.203952Z` 的不同包身份拒绝均保留，未运行模型，不计入性能或对照成功。

```sh
REF=/Users/beyondpower/Documents/Projects/SakuraTTS-References
"$REF/.venv-mlx-macos/bin/python" harness/sovits_package_loading.py \
  --package "$REF/models/converted/20260919T123227.630582Z-20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32-lossless-storage"
"$REF/.venv-mlx-macos/bin/python" harness/encoder_softmax_candidate.py \
  --references "$REF" \
  --package "$REF/models/converted/20260919T115416.309917Z-f8bd92196175-sovits-decode-fp32" \
  --official-conditions "$REF/runs/20260919T122232.246358Z-sovits-fixed-official-mps" \
  --candidate mlx-fp64 --runtime \
  --equivalence-reference "$REF/runs/20260919T133516.751269Z-encoder-softmax-mlx-fp64-runtime"
"$REF/.venv-mlx-macos/bin/python" -m unittest discover -s tests -p test_sovits_package.py -v
```
