# 日文运行包不再携带安装工具

日期：2026-09-20。完整目录隔离通过后，按逐文件清单检查分发内容，发现基础 Python 和 venv 各有一份 pip / setuptools，基础 Python 还带有 ensurepip 的安装轮子。

## 排除范围

固定的日文源码与 CLI 不导入这些工具；已安装依赖的元数据也没有把它们列为无条件运行依赖。第三方源码中的相关引用位于 MLX 扩展构建助手、NumPy 开发工具与测试、ORT 的其他模型示例等路径，未进入当前日文调用链。

导出器现在默认不复制 pip、setuptools、pkg_resources、`_distutils_hack`、对应的 `.dist-info` 和 `distutils-precedence.pth`。捆绑基础 Python 时还排除 ensurepip 及三个 pip 入口。保留的日文依赖、distribution 元数据、词典、Nani 和四个模型 / 条件包不变；没有修改源环境，也没有删掉历史完整运行目录。

`--include-install-tools` 保留完整环境的导出选择。默认生成的是推理目录，变更依赖应在开发环境完成后重新导出。省略规则只作用于明确的安装工具路径，不按文件大小或是否在单次 trace 中出现来任意删文件。默认的 `venv --without-pip` 创建方式继续正常工作。

启动探针确认新环境找不到 pip、setuptools、pkg_resources、`_distutils_hack`、ensurepip；没有遗留 `.pth` 自动导入。清单区分源环境的 29 个 distribution 与新目录的 27 个运行依赖，避免把未复制的工具继续写成已安装。

## 输出与体积

在新目录中，沿用上一轮严格沙箱，再禁用旧完整运行目录。实际负探针确认 11 个旧文件和网络被拒，涵盖旧源码、环境、原模型、历史实验、旧运行包及 Darwin 用户 cache / temp。四条原始日文使用同一参考与 NumPy seed 0，实际 CLI 全部退出 0。

每例 14 项比较通过，包含完整文本记录、音素、生成摘要、参数 / 精度 / 策略、源码、依赖、词典与四包身份、WAV 参数及全部 WAV / PCM 字节。推理源码和资源哈希保持；少掉的文件全部位于声明的安装工具范围内。共同文件中只改变了导出器本身和随安装位置变化的 venv 配置 / 激活脚本，后者经路径替换对照确认。

| 口径 | 字节数 | MiB |
| --- | ---: | ---: |
| 旧完整目录，含清单 | 1,148,959,761 | 1095.733 |
| 新推理目录，含清单 | 1,123,935,698 | 1071.869 |
| 净减少 | 25,024,063 | 23.865 |
| 省略的工具文件本身 | 24,025,189 | 22.912 |

工具文件共有 1960 个，旧 payload 的 5658 项降为 3698 项；清单也随条目减少而变小。净减少已包含导出器变更、路径长度和清单大小的差异。新目录约 1.047 GiB，是逻辑文件大小，仍不含系统库、驱动服务、文件系统分配开销或压缩率。

四例生成增加 1,977,282 bytes 的 WAV / JSON，单独留在 `output/`，不计入安装体积；包内 home / cache / tmp 没有新增文件。系统服务 I/O 没有完整追踪，不声称系统缓存为零。

本轮不改变模型驻留、计算或精度，没有提出显存、RSS 或速度收益。严格隔离下四例请求耗时依次为 4.847、1.819、4.657、2.871 s；各一次新进程运行，仅保留观察，不与正常热基准混用。未新增 ASR 或人工试听，既有概率与 MRTE 超差不受此改动影响。

## 证据

新推理目录：`SakuraTTS-References/deployments/20260919T175654.361087Z-japanese-inference-runtime/`。

实验目录：`SakuraTTS-References/runs/20260919T175654.361087Z-runtime-install-tools/`。`config.json`、`run.py` 保存输入与调度，逐项 `*.process.json` 保存命令和真实退出码，`exporter.py` 是执行时的源码快照。`result.json` 为 `passed`，列出所有省略项、共同文件变化、原始计时及逐例检查；`absent-install-tools.json` 和 `negative-probe.json` 分别证明工具缺失与实际隔离。重新生成应使用新目录。

独立复核结果保存在 `install-tools-dependency-audit.json`、`install-tools-result-audit.json` 和 `install-tools-audio-review.json`。审计核对了分发记录中的文件归属、体积算术、全部文件大小及链接、33 个小源码哈希，重新运行缺失工具与隔离探针；四例的 11 组元数据和 3 项音频检查也重新计算通过。没有为复核重跑模型，大权重哈希沿用执行后的全量校验。
