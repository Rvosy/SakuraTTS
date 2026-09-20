# Windows 运行依赖体积与重复文件

日期：2026-09-20。基线提交 `d323b5072ac57caad3e792eebe4ef415e9aa9553`。本轮只读取现有文件，没有卸载依赖、修改 DLL、启动 GPU 或重新生成语音。中文说明按 `humanizer-zh` 校订。

两部分日常运行环境合计 **4.314 GiB**。其中五个 CUDA DLL 各保存了两份，SHA256 相同，共有 **828.58 MiB** 内容重复。它们适合优先验证共用文件的打包方式，但这个数字还不是已经可以删除的体积。

## 计量范围

新增 [只读清点脚本](../tools/windows_runtime_inventory.py)，逐文件记录逻辑字节、SHA256、目录分类和文件身份。相同路径被多个输入目录覆盖时只计一次；硬链接另算一次文件内容，不把已共享的空间算作可节省体积。符号链接和 Windows reparse point 不跟随，单独报告。本轮 9,965 个文件中没有输入范围内的硬链接别名或跳过项。

运行目录来自 [安装说明](../../docs/setup-windows-nvidia.md) 和 `models/windows-sakura/runtime.json`。该配置选择 `gpt`、`sovits-onnx-v1`、`frontend-classic` 及 CPU 准备的五组 `references`。不计旧声学导出、plus 前端资源包、CUDA 对照参考和准备日志。

配置文件 SHA256 为 `9ab16715d46cea5e4a389c30c0f1ac0202d4211b7ec72bca5cc32d046d0a0c42`；固定运行依赖清单 SHA256 为 `332e882a0e13bbe041e7795b421c53197f6baa07c80b1d1bb7f118079719183b`。

| 范围 | 实际目录 | 逻辑字节 |
| --- | --- | ---: |
| 主运行环境 | `.venv-windows-runtime` | 1,562,831,265 |
| CPython 3.9 / ORT CUDA worker | `data/windows-ort-runtime` | 3,069,419,804 |
| 两个运行环境合计 | | **4,632,251,069** |
| 外部共享 CPython 3.11.15 | uv 管理的基础解释器目录 | 74,324,992 |
| GPT 包 | `models/windows-sakura/gpt` | 318,767,118 |
| SoVITS 包 | `models/windows-sakura/sovits-onnx-v1` | 251,347,840 |
| 经典日文前端资源 | `models/windows-sakura/frontend-classic` | 262,567,159 |
| 五组参考条件 | `models/windows-sakura/references` | 1,385,885 |
| 所选模型与资源合计 | | **834,068,002** |
| CuPy 编译缓存 | `C:/Users/Rvosy/.cupy/kernel_cache` | 435,497 |

不含缓存的上述范围共 5,540,644,063 bytes，约 5.160 GiB。主运行环境中的测试、元数据、字节码及后来安装的测量依赖都按现状计入；本轮没有裁剪它们。worker 比准备清单的 3,069,259,906 bytes 多 159,898 bytes，因为这里也计入清单本身和导入日志等目录文件。

基础 Python 从 `pyvenv.cfg` 定位。uv 的 `cpython-3.11-windows-x86_64-none` 是 junction，实际扫描其 `cpython-3.11.15-windows-x86_64-none` 目标。共享解释器单列，未混入主 venv，也未假定用户机器已经安装。CuPy cache 是本机已有缓存快照，未清空或重新预热。

这些是未压缩文件长度，不是压缩下载包、磁盘分配空间、加载后的 CPU 内存或显存。范围外仍有 editable 安装所用的仓库源码、运行配置、系统 DLL、驱动和可能的驱动缓存；本轮没有遍历整个用户缓存或开发环境。因此 5.160 GiB 也不是完整安装包的验收数字。

## 重复内容与下一步

以下文件分别位于主环境的 `Lib/site-packages/nvidia/*/bin` 和 worker 的 `cuda` 目录。逐对字节长度与 SHA256 均相同，文件身份不同，属于两份实际文件。

| DLL | 每份字节 | 重复份数 |
| --- | ---: | ---: |
| `cublasLt64_12.dll` | 668,673,536 | 1 |
| `cublas64_12.dll` | 102,518,272 | 1 |
| `nvrtc64_120_0.dll` | 89,832,960 | 1 |
| `nvrtc-builtins64_129.dll` | 7,217,664 | 1 |
| `cudart64_12.dll` | 583,680 | 1 |
| 合计 | **868,826,112** | |

`configure_cuda()` 在主进程搜索已安装的 NVIDIA 包，在 worker 优先加入解释器旁的 `cuda`。worker 准备工具还记录了本地 DLL、许可和文件哈希。哈希相同只说明内容重复，未验证删掉一侧后能否满足两种加载路径、独立启动和搬迁要求。

最小候选是在新导出目录内为这五组文件建立硬链接。`configure_cuda()`、`ort_process.py` 的工作进程入口和现有相对路径都可保留；准备工具在复制后核对大小与 SHA256，再让另一份导出路径指向包内文件。不能把新包硬链接到用户现有 wheel 或开发环境，以免更新一份时影响另一份。manifest 继续记录每条路径的哈希，另记共享关系；许可和 CuPy JIT 头文件不变。

这种方式只减少支持硬链接的文件系统上的重复存储，逻辑路径总长度仍为 4.314 GiB。普通复制、归档或安装器可能把它恢复成两份，必须在最终安装目录再次用本脚本复核 `unique_file_bytes`。已安装文件应当按不可变文件处理，升级时替换单独的文件身份，避免原位改写共享内容。

如果最终分发方式不能保留硬链接，再考虑单独的 `shared-cuda` 目录。届时需要 `configure_cuda()` 显式接受包内 DLL 路径并保留搜索句柄，让主进程和 ORT worker 从同一处加载；worker 启动时传递由新包位置计算的路径，不能存开发机绝对路径。`prepare_ort_worker_runtime.py` 的复制计划和 manifest 也要区分 worker 私有 DLL 与共用 DLL。主环境 NVIDIA 包内的头文件仍需保留，不能将整个包目录删除。这比硬链接试验涉及更多加载和打包行为，本轮未实施。

两种候选都要在新目录断开旧环境访问，运行经典日文前端、完整请求、重复请求和 worker 重启，再将整个候选安装到另一位置复测。加载路径检查需覆盖主进程与 worker 的原生 DLL，单纯成功 `import` 不能证明 CUDA provider 和 JIT 都可用。只有这些检查通过后，才能把约 829 MiB 从“待验证可节省”改成安装收益。

即使完全共用这五个 DLL，现有两个运行环境按逻辑文件长度仍约 3.505 GiB。其他大项包括 worker 的 cuDNN 1,144,448,608 bytes、ORT CUDA provider 单文件 607,457,312 bytes、cuFFT 276,121,088 bytes；主环境还带有 Sudachi 词典 217,466,039 bytes 和 plus OpenJTalk 包约 111 MB。它们需要依赖与功能验证，不能从文件名或本次短清单判断可以删除。

主环境的 `nvrtc64_120_0.alt.dll` 有 89,899,520 bytes，SHA256 与普通版本不同，而且属于 NVIDIA wheel 的 `RECORD`。本轮没有把它当作重复文件或临时备份。经典版与 plus 的 OpenJTalk 主字典也不相同，不能用少量共享字典文件推断两个前端等价。

先复现共用 DLL 的收益，再考虑 ORT 构建裁剪和按前端配置导出依赖，比直接以 200 MB 为安装包目标更容易验证。与 Genie 比较前还需统一是否包含 CUDA/cuDNN、Python、词典、模型，以及报告的是压缩包还是安装后大小。

## 复现与验证

在仓库根目录运行；输出必须是新路径，且位于被测目录之外：

```powershell
.venv-windows-runtime\Scripts\python.exe -B harness\windows_runtime_inventory.py `
  --root 'runtime:main=.venv-windows-runtime' `
  --root 'runtime:ort-worker=data/windows-ort-runtime' `
  --root 'shared_python:cpython311=C:/Users/Rvosy/AppData/Roaming/uv/python/cpython-3.11.15-windows-x86_64-none' `
  --root 'model:gpt=models/windows-sakura/gpt' `
  --root 'model:sovits=models/windows-sakura/sovits-onnx-v1' `
  --root 'model:frontend=models/windows-sakura/frontend-classic' `
  --root 'model:references=models/windows-sakura/references' `
  --root 'cache:cupy=C:/Users/Rvosy/.cupy/kernel_cache' `
  --output outputs/windows-runtime-inventory-new/inventory.json `
  --summary-output outputs/windows-runtime-inventory-new/summary.json
```

实际全量证据在 `outputs/windows-runtime-inventory-20260920/inventory.json`。随源码保存的 [JSON 摘要](data/2026-09-20-windows-runtime-inventory.json) 保存各目录、组件计量、最大的 30 个文件与 30 个重复组，以及全量文件的大小和 SHA256。它没有把未列出的重复组隐去后重新计算总数。

局部测试实际运行 `python -B -m unittest discover -s tests -p test_windows_runtime_inventory.py -v`：重叠路径、真实硬链接与独立副本、不同分类重叠拒绝三项通过。符号链接用例因 Windows 当前用户没有创建符号链接权限而跳过。全量扫描完成，并复核了已枚举文件的身份、大小和修改时间；这不是锁定文件系统的快照，后续运行新增的缓存不会自动补入本轮结果。

本轮只验证清点工具与现有文件，不提供删减运行依赖后的合成、搬迁或音质结论。
