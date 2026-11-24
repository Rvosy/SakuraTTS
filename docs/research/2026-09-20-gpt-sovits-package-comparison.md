# GPT-SoVITS 普通包与 NVIDIA 50 系包依赖对比

日期：2026-09-20。研究对象是 FlowerCry 在 ModelScope 上保存的两份 `GPT-SoVITS-v2pro-20250604` 归档；结论绑定文件 SHA256，不把文件名里的日期视为精确源码版本。

## 对 SakuraTTS 的建议

先维护一套现代 NVIDIA 推理运行包，不照搬上游“普通版 / 50 系版”的划分。当前 wheel 和源码包本来就不捆绑 CUDA 运行环境，更不需要按显卡代际生成两个 Python 包。

若后续提供离线整合环境，建议先以 Turing 及更新架构作为候选范围，即 GTX 16、RTX 20/30/40/50 系列。这里说的是依赖支持范围，不是已经完成的产品支持承诺：项目已有记录只验证了 RTX 5060，其他代际仍需实际测试。

是否增加旧卡环境，应取决于是否明确支持 GTX 10 等 Pascal 显卡，以及统一依赖方案能否通过回归。不要仅因上游提供两个下载链接，就提前维护两套包。即使需要两套运行环境，SakuraTTS 代码、模型和参考资源也应共用，仅拆分运行库。

## 两份归档

| 项目 | 普通包 | NVIDIA 50 系包 |
| --- | ---: | ---: |
| 文件名 | `GPT-SoVITS-v2pro-20250604.7z` | `GPT-SoVITS-v2pro-20250604-nvidia50.7z` |
| 下载字节 | 8,185,086,602 | 8,835,144,925 |
| 下载大小 | 7.623 GiB | 8.228 GiB |
| 归档内文件长度之和 | 14,128,680,597 bytes / 13.158 GiB | 14,964,402,944 bytes / 13.937 GiB |
| 普通文件数，含空文件 | 50,612 | 51,648 |
| 顶层 site-packages 的 dist-info 数 | 246 | 245 |

50 系包下载量多 619.944 MiB，包内文件总量多 797.007 MiB。`torch/` 增加 635.586 MiB，`onnxruntime/` 增加 130.498 MiB；两者合计占未压缩增量约 96%。目录计量不包含各自位于同级的 dist-info，也不是实际磁盘分配量。

两份归档中，共有路径里有 40,578 个文件的大小和 CRC 一致，8,741 个不同；普通包另有 1,293 个独有文件，50 系包另有 2,329 个。共有的非 `runtime/` 文件大小和 CRC 全部一致。普通包额外保留了 BigVGAN demo、filelists 等文件，50 系包额外保留 `GPT_SoVITS/text/G2PWModel_1.1.zip`。这些是附带内容差异，不能据此推导两种模型格式或两套推理算法。

## Python 依赖差异

下面列出全部 dist-info 名称或版本差异，已逐个读取 METADATA 核对；还需结合下一节的二进制构建差异阅读。

| 依赖 | 普通包 | NVIDIA 50 系包 |
| --- | --- | --- |
| torch | 2.0.0+cu118 | 2.7.0+cu128 |
| torchaudio | 元数据 2.0.0；version.py 为 2.0.0+cpu | 2.7.0+cu128 |
| torchvision | 元数据 0.15.1；version.py 为 0.15.1+cpu | 未包含包目录或 dist-info |
| ctranslate2 | 3.24.0 | 4.6.0 |
| pytorch-lightning | 2.1.3 | 2.4.0 |
| sympy | 1.11.1 | 1.13.3 |
| gradio | 4.44.1 | 4.24.0 |
| gradio-client | 1.3.0 | 0.14.0 |
| urllib3 | 2.4.0 | 1.26.13 |

两边同版本的主要依赖包括 NumPy 1.23.4、Transformers 4.43.0、librosa 0.9.2、SciPy 1.9.3、soundfile 0.12.1、pyopenjtalk 0.3.4、x-transformers 2.0.2 和 onnxruntime-gpu 1.19.2。

不能把全部差异都归因于 50 系显卡。Torch/CUDA/cuDNN 是明确的 GPU 运行栈变化；Gradio、urllib3 等差异只是这两份归档的实际状态，没有证据表明它们是支持 50 系的必要条件。SakuraTTS 也不需要为了分发推理运行库而引入这些 UI、训练或 ASR 依赖。

## Python 与 GPU 二进制构建

| 项目 | 普通包 | NVIDIA 50 系包 |
| --- | --- | --- |
| CPython | 3.9.13 / Windows x64 | 3.9.13 / Windows x64 |
| Torch 的 CUDA 构建 | 11.8 | 12.8 |
| Torch 捆绑 CUDA runtime | 11.8.89 | 12.8.57 |
| Torch 的 cuDNN | 8 系列 | 9.7.1.26 |
| ORT Python 包版本 | 1.19.2 | 1.19.2 |
| ORT `version_info.py` | CUDA 11.8.89 / cuDNN 8 | CUDA 12.2.128 / cuDNN 9 |
| ORT CUDA provider DLL 大小 | 471,534,136 bytes | 607,457,312 bytes |

Python 版本来自 `python39.dll` 的版本资源，两包该 DLL 的 SHA256 也相同。Torch CUDA 构建来自实际 `torch/version.py`；CUDA runtime 和 50 系 Torch 的 cuDNN 版本来自对应 DLL 的版本资源。普通包 cuDNN 8 的 PE 版本资源仍包含旧的 `6.5.0` 文本，不能用它确定真实 cuDNN 小版本，因此此处只确认其 8 系列 ABI。

ORT 的差异已用 DLL 导入表交叉确认：普通包导入 `cublas64_11.dll`、`cublasLt64_11.dll`、`cudart64_110.dll`、`cufft64_10.dll` 和 `cudnn64_8.dll`；50 系包对应导入 cuBLAS 12、CUDA runtime 12、cuFFT 11 和 `cudnn64_9.dll`。这不是同一文件改名，而是同一个 ORT 版本的两种构建。只有 `pip freeze` 或 requirements 无法表达完整区别。

[ORT 官方 CUDA provider 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)也分别列出 1.19.x 的 CUDA 11.8/cuDNN 8 和 CUDA 12.x/cuDNN 9 构建，并明确 cuDNN 8 与 9 不兼容。50 系包的 ORT 构建版本为 CUDA 12.2，不必为了与 Torch 的 12.8 一致而把它误写成 12.8；实际 DLL 属于 CUDA 12 主版本系列，运行兼容还取决于库与驱动。

[PyTorch 2.7 官方发布说明](https://pytorch.org/blog/pytorch-2-7/)将 Blackwell 支持和 CUDA 12.8 构建作为新增能力。这与两份包里 Torch 2.0/cu118 和 2.7/cu128 的变化一致，解释了支持新显卡所需的主要运行栈升级；不能进一步推断新版运行栈只适用于 50 系。

还有几处说明这两份包不宜直接当作 SakuraTTS 的依赖锁文件：

- 两包根目录 `requirements.txt` 的 SHA256 完全相同：`3b9e3a82029cc701c6fedd1e4d4a01fec2dc7fcb58ec83b634fd3fcbd6580fb3`，实际安装环境却有上述差异。
- 50 系包的 `torch/lib/cudnn64_9.dll` 为 9.7.1.26，`ctranslate2/cudnn64_9.dll` 为 9.1.0.70，哈希不同。不能按文件名直接去重，也没有在 Windows 中检查具体启动顺序下实际加载了哪份。
- 两包都保留 `torch-directml 0.2.0.dev230426`；它的 METADATA 要求 Torch 2.0.0、torchvision 0.15.1，与 50 系包的实际环境不一致。这是静态依赖冲突，不能等同于已经复现 CUDA 推理失败。
- Torch METADATA 中还含仅适用于 Linux 的 NVIDIA 依赖条件。本次 Windows CUDA 结论来自实际版本文件和 DLL，没有把 Linux 条件依赖算入 Windows 环境。

## 对应到 SakuraTTS

检查的项目基线为 `9759794e2ce6b214fc05285f1c5ddc2b715debff`。普通运行依赖和实现表明：

- 主环境是 Python 3.11、CuPy 14.2.0、CUDA runtime 12.9.79、NVRTC 12.9.86、cuBLAS 12.9.2.10。GPT 自定义核经 `cp.RawKernel` 编译，代码没有固定到 `sm_120`。
- 声学工作进程使用 CPython 3.9 / ORT CUDA 1.19.2；准备工具复制 cuDNN 9 系列 DLL。安装说明指定准备组件使用 `nvidia-cudnn-cu12==9.26.0.51`。
- 普通推理不依赖 Torch；Torch / torchaudio 属于转换、参考准备和开发对照环境。不能直接把上游 Torch 整合包的分发边界套到产品运行包上。

50 系包的 ORT CUDA provider SHA256 为 `5351fb347f6978075c6db4cf18b2c1352949f569e352a475dab667ae06be61df`，与项目已有 [Windows 清点摘要](../../research/experiments/data/2026-09-20-windows-runtime-inventory.json) 保存的值完全一致。这说明已有记录中的声学组件正是该 CUDA 12/cuDNN 9 构建；它不表示整套运行环境与上游相同，SakuraTTS 使用的 CUDA 12.9 与 cuDNN 9.26 组件另行准备。

[NVIDIA cuDNN 9.26.0 支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.26.0/reference/support-matrix.html)列出 CUDA 12.x 构建支持 Turing 7.5、Ampere、Ada、Hopper 和 Blackwell。它没有列入 Pascal；GTX 16 是 Turing，不能和 GTX 10 一起归为相同兼容档位。本轮另存了对应版本的官方文档快照。

若将来需要 GTX 10，可先验证一套同时覆盖 Pascal 和 Blackwell 的依赖组合，再判断是否需要独立 legacy 包。例如 [cuDNN 9.10.2 的矩阵](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html)仍列有 Pascal/Maxwell，同时也有 Blackwell；[CUDA 12.9 发布说明](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)说明 CUDA 12.x 仍支持为旧架构构建。它们说明“支持 50 系”与“只能支持 50 系”不同，并不证明降级 cuDNN 后 SakuraTTS 的数值、性能和全部算子已经可用。

在发布前，单一现代包至少要在承诺范围的较老代表卡和 50 系上验证首次 JIT、GPT、ORT CUDA provider 实际执行、完整 WAV、重复请求及目录搬迁。驱动下限要按实际 NVRTC/PTX 和 GPU 要求确定，不能只引用 CUDA 12.x 通用的最低驱动；NVRTC JIT 有自己的驱动约束。

可复用的现有依据是 [Windows 安装说明](../setup-windows-nvidia.md)、[运行依赖清单](https://github.com/Rvosy/SakuraTTS/blob/9759794e2ce6b214fc05285f1c5ddc2b715debff/requirements-windows-runtime.txt)、[GPT 实现](https://github.com/Rvosy/SakuraTTS/blob/9759794e2ce6b214fc05285f1c5ddc2b715debff/src/sakuratts/cuda_gpt.py)、[声学组件准备工具](https://github.com/Rvosy/SakuraTTS/blob/9759794e2ce6b214fc05285f1c5ddc2b715debff/scripts/prepare_ort_worker_runtime.py)及 [Windows 运行环境清点](../../research/experiments/2026-09-20-windows-runtime-inventory.md)。本轮没有在 Windows 或 GPU 上重跑这些验证。

## 下载、校验与证据

两份文件已完整下载，SHA256 均与 ModelScope API 返回值一致：

| 文件 | SHA256 |
| --- | --- |
| 普通包 | `bd60d0796553ff05d8568136e199c13e0dc22ebe2ed24273134e34ed6f215cd6` |
| 50 系包 | `97b4edcd451c42357db7e26e6c1c877ca5d85144fe97beaff6d7005d35bee008` |

来源：[普通包](https://modelscope.cn/models/FlowerCry/gpt-sovits-7z-pacakges/file/view/master/GPT-SoVITS-v2pro-20250604.7z)、[50 系包](https://modelscope.cn/models/FlowerCry/gpt-sovits-7z-pacakges/file/view/master/GPT-SoVITS-v2pro-20250604-nvidia50.7z)。API 分别记录文件 revision `f28dff5a72bea299249afee5f3a87e7bece58d8d`、`73078800531a976476f70fd018d8e6d8aeaa9c31`。

本地证据保存在被 Git 忽略的 `outputs/gpt-sovits-package-comparison/`：

- 两个完整 `.7z`、`modelscope-files.json`、分段下载和 SHA256 校验记录。
- `*.listing.txt`、`standard-files.json`、`nvidia50-files.json` 和 `listing-comparison.json`：完整归档文件目录、大小、CRC 与依赖目录对比。
- `extracted/`：普通包 508 个、50 系包 505 个实际提取文件，共约 1.08 GiB，包含全部顶层 dist-info 的 METADATA/WHEEL、关键版本文件和 DLL。7-Zip 两次选择性解压均报告 `Everything is Ok`；日志里的处理文件数包含固实块解码所经过的文件，不等于最终写出的文件数。
- `standard-analysis.json`、`nvidia50-analysis.json`、`verified-summary.json`：逐个读取的 Python 元数据、文件 SHA256、PE DLL 导入表和版本资源。
- `sources.json` 和 HTML 快照：官方资料的 URL、下载字节与 SHA256；辅助分析脚本也保留在同目录。

本轮完成的是完整下载校验、归档目录比较、选择性解压和静态依赖分析。没有安装或执行上游 Windows 程序，没有在不同显卡上运行推理，也没有全量解压所有模型。本轮仅新增这份研究记录，未修改产品依赖或打包实现。
