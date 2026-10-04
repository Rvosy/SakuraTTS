# Mac 整合包构建与验收

用户使用方法见 [Apple 指南](apple.md)。本页说明如何装配 Apple silicon 日文包。Windows 的 NVIDIA 与 CPU/AMD 组合继续按 [Windows 整合包指南](portable-bundle.md)构建；不同系统使用同一份产品源码，各自携带匹配的解释器与计算依赖。

## 运行组件

发行组合由 [macos-mlx-ja.toml](../packaging/recipes/macos-mlx-ja.toml)选择，依赖版本来自 [pyproject.toml](../pyproject.toml)。主环境包含 MLX、CPU ONNX 日文前端及可选 HTTP 服务；准备组件单独包含 CPU PyTorch、原版源码与公共分析资源。包内不收录用户权重、参考、缓存和开发环境。

recipe 的 `minimum_macos` 是构建目标。构建器检查 wheel 的平台标签，以及 Mach-O 的架构、最低系统版本和动态库依赖；不会把开发机上自动选中的 macOS 26 wheel 当成 macOS 14 构建。系统版本目标仍须在对应系统上实测。

先在单独目录准备运行依赖，例如用 uv 明确选择 macOS 14 wheel：

```sh
uv venv --python 3.11 build-inputs/runtime
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip install \
  --python build-inputs/runtime/bin/python --python-platform aarch64-apple-darwin \
  '.[mlx,japanese,server]'
```

这一步可联网获取构建输入；后续装配只读取本地文件。缓存齐全时可加 `--offline`。准备依赖同样按目标系统解析，实际闭包由 [build_preparation.ROOT_REQUIREMENTS](../scripts/build_preparation.py)及安装包元数据决定。可从 `.[convert,japanese]` 的准备环境补齐该清单，再交给构建器检查。

`--python-base` 必须是可搬迁的独立 CPython，采用 `bin/python3.X`、`lib/python3.X` 布局；不能是 venv、系统 Python 或 Homebrew framework。构建器按清单选择标准库与安装包文件，不复制旧 venv 的绝对路径。准备环境中的 Numba 使用包内 workqueue；不携带依赖外部 SoX、FFmpeg 动态库的可选 TorchAudio 插件。参考解码使用独立 FFmpeg 可执行文件和 SoundFile。

`--ffmpeg` 需要 arm64 独立构建，支持项目使用的音频编码器。直接复制 Homebrew FFmpeg 通常会保留 `/opt/homebrew` 动态库依赖，构建检查会拒绝这种输入。版本、编译选项与许可写入发行包的 `licenses/FFmpeg-build-and-license.txt`。

## 离线装配

先用 `scripts/build_preview.py` 构建经过筛选的 wheel，取出 ZIP 中 `dist/` 下的 wheel。再装配准备组件与完整包：

```sh
python scripts/build_preparation.py --target macos-arm64 --minimum-macos 14.0 \
  --python-base LOCAL_STANDALONE_CPYTHON \
  --site LOCAL_PREPARATION_SITE_PACKAGES \
  --official-source LOCAL_SUPPORTED_GPT_SOVITS_SOURCE \
  --language-model LOCAL_LID_176_BIN \
  --output dist/macos-preparation --audit outputs/macos-preparation-inputs.json
python scripts/build_portable.py --recipe packaging/recipes/macos-mlx-ja.toml \
  --python-base LOCAL_STANDALONE_CPYTHON \
  --main-site LOCAL_RUNTIME_SITE_PACKAGES \
  --ffmpeg LOCAL_STANDALONE_FFMPEG --wheel LOCAL_PRODUCT_WHEEL \
  --preparation dist/macos-preparation \
  --output dist/SakuraTTS-macOS-AppleSilicon --audit outputs/macos-inputs.json
```

构建脚本需要 `packaging`。`--plan-only` 只检查输入；输出目录必须不存在。省略 `--preparation` 可生成只使用预制模型与参考的精简包。Windows 的 `--vc-runtime` 不用于 macOS。

主包和准备组件分别保存依赖版本、源码身份、文件清单和 SHA256。来源绝对路径只进入包外 audit。Apple 包使用 `runtime/main/bin/python3` 和 `.command` 启动器，保留可执行权限；缓存位于包内。安装路径可以包含中文和空格。

辅助资源归属见 [auxiliary-model-sources.json](../packaging/auxiliary-model-sources.json)。准备组件的 `licenses.json` 保留尚待补齐的权重许可材料；公开发布前仍须完成所携第三方二进制及辅助权重的分发材料核对。

## 验收与压缩

使用新的完整包和空缓存，运行 [首次使用验收](../scripts/verify_portable_first_use.py)：

```sh
python scripts/verify_portable_first_use.py --bundle dist/SakuraTTS-macOS-AppleSilicon \
  --backend mlx --gpt LOCAL_V2PRO_CKPT --sovits LOCAL_V2PRO_PTH \
  --reference LOCAL_REFERENCE_AUDIO --prompt-text '参考音声です。' \
  --output outputs/macos-first-use
```

验收解释器需要 `psutil`；服务、推理和准备使用包内解释器。检查原始转换、新参考、重复 PCM、重启缓存、direct/managed 和进程退出。搬迁后用相同参数加 `--reuse-cache` 验证已有缓存，不清除用户数据。脚本本身不限制网络和文件读取；隔离条件由外部测试环境提供。旧系统、其他硬件和音质需分别验收。

Mac 使用系统可解压的 `tar.gz`，保存启动器与解释器的可执行权限：

```sh
python scripts/archive_portable.py --bundle dist/SakuraTTS-macOS-AppleSilicon \
  --format tar.gz --output dist/macos-release
```

压缩仅包含发行清单的文件，先核对输入哈希，压缩后逐文件核对内容。测试生成的模型缓存、日志和音频不会进入归档。产物包含压缩包、SHA256 和压缩报告。实际构建、验收与未验证范围见 [2026-10-04 记录](../research/notes/macos-portable-20261004.md)。
